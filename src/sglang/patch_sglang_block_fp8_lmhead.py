#!/usr/bin/env python3
"""Pixion patch: block-FP8 (128x128) lm_head loading for SGLang v0.5.20 lineage.

Two fixes, each guarded by unique anchors + post-patch compile() gate:

1. srt/layers/vocab_parallel_embedding.py
   VocabParallelEmbedding.__init__ passes its own vocab row-parallel loader to
   quant_method.create_weights(...). Fp8LinearMethod attaches that same loader
   to block-FP8 weight_scale_inv, whose rows are vocab/128 (1940 for vocab
   248320 / group 128). Loading lm_head.weight_scale_inv then fails with
   "org_vocab_size=248320 ... loaded_weight.shape[output_dim]=1940".
   Fix: when tp_size == 1 and the scale shape matches the block grid exactly,
   install a strict-shape direct loader on that parameter.

2. srt/layers/quantization/modelopt_quant.py
   ModelOptMixedPrecisionConfig resolves layer quant entries by suffix and only
   maps "*.lm_head" to the "lm_head" checkpoint entry. The NEXTN MTP draft head
   is named "model.shared_head.head"; without a candidate the MTP worker would
   build an UNQUANTIZED lm_head for an FP8 checkpoint.
   Fix: add "lm_head" as a candidate for "*shared_head.head" prefixes.

Idempotent (marker scan) and fail-fast (anchor count, compile gate).
File discovery uses importlib find_spec first so editable installs (official
images ship sglang editable at /sgl-workspace/sglang/python) are handled.
Exit 0 on success (patched or already patched), non-zero otherwise.
"""

import glob
import importlib.util
import pathlib
import sys
import sysconfig


def find_file(rel: str) -> pathlib.Path:
    cands: list[pathlib.Path] = []
    try:  # canonical: resolves editable installs and .pth redirects
        spec = importlib.util.find_spec("sglang")
        if spec is not None and spec.submodule_search_locations:
            cands.extend(pathlib.Path(p) for p in spec.submodule_search_locations)
    except Exception:
        pass
    for key in ("purelib", "platlib"):
        try:
            cands.append(pathlib.Path(sysconfig.get_paths()[key]) / "sglang")
        except Exception:
            pass
    for pat in (
        "/opt/*/lib/python3*/site-packages/sglang",
        "/usr/local/lib/python3*/site-packages/sglang",
        "/usr/lib/python3*/site-packages/sglang",
        "/sgl-workspace/*/python/sglang",
    ):
        cands.extend(pathlib.Path(p) for p in glob.glob(pat))
    seen: set[pathlib.Path] = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        p = c / rel
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    raise FileNotFoundError(f"cannot locate sglang/{rel} in {sorted(map(str, seen))}")


def patch(path: pathlib.Path, anchor: str, replacement: str, marker: str) -> None:
    src = path.read_text()
    if marker in src:
        print(f"[pixion] already patched: {path}")
        return
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"[pixion] anchor count {n} != 1 in {path}; aborting")
    new = src.replace(anchor, replacement)
    try:
        compile(new, str(path), "exec")
    except SyntaxError as exc:
        raise SystemExit(f"[pixion] patched source fails syntax gate: {exc}")
    path.write_text(new)
    print(f"[pixion] patched: {path}")


def main() -> int:
    # --- 1. vocab_parallel_embedding.py: block-FP8 scale loader ---
    vpe = find_file("srt/layers/vocab_parallel_embedding.py")
    vpe_anchor = (
        "            params_dtype=params_dtype,\n"
        "            weight_loader=self.weight_loader,\n"
        "        )\n"
    )
    vpe_replacement = vpe_anchor + (
        "\n"
        "        # pixion: a block-quantized (128x128) weight_scale_inv for a vocab-\n"
        "        # parallel module has vocab_size/128 rows; the vocab row-parallel\n"
        "        # loader registered by create_weights would assert. For TP=1 with an\n"
        "        # exact block-grid shape, install a strict-shape direct loader.\n"
        "        _pixion_scale = getattr(self, \"weight_scale_inv\", None)\n"
        "        if (\n"
        "            _pixion_scale is not None\n"
        "            and self.tp_size == 1\n"
        "            and tuple(getattr(_pixion_scale, \"shape\", ()))\n"
        "            == (\n"
        "                (self.num_embeddings_padded + 127) // 128,\n"
        "                (self.embedding_dim + 127) // 128,\n"
        "            )\n"
        "        ):\n"
        "            def _pixion_block_scale_loader(param, loaded_weight, *args, **kwargs):\n"
        "                if tuple(loaded_weight.shape) != tuple(param.data.shape):\n"
        "                    raise RuntimeError(\n"
        "                        \"pixion block-FP8 vocab scale shape mismatch: \"\n"
        "                        f\"{tuple(loaded_weight.shape)} != {tuple(param.data.shape)}\"\n"
        "                    )\n"
        "                param.data.copy_(loaded_weight.to(param.data.dtype))\n"
        "                return param\n"
        "\n"
        "            # BasevLLMParameter.weight_loader is a read-only property\n"
        "            # returning _weight_loader; assign the backing field when so.\n"
        "            if isinstance(\n"
        "                getattr(type(_pixion_scale), \"weight_loader\", None),\n"
        "                property,\n"
        "            ):\n"
        "                _pixion_scale._weight_loader = _pixion_block_scale_loader\n"
        "            else:\n"
        "                _pixion_scale.weight_loader = _pixion_block_scale_loader\n"
    )
    patch(vpe, vpe_anchor, vpe_replacement, "_pixion_block_scale_loader")

    # --- 2. modelopt_quant.py: shared_head.head -> lm_head candidate ---
    mo = find_file("srt/layers/quantization/modelopt_quant.py")
    mo_anchor = (
        "        if prefix.endswith(\".lm_head\"):\n"
        "            candidates.append(\"lm_head\")\n"
    )
    mo_replacement = mo_anchor + (
        "        # pixion: NEXTN shared head is named model.shared_head.head;\n"
        "        # resolve it to the lm_head quantization entry.\n"
        "        if prefix.endswith(\"shared_head.head\"):\n"
        "            candidates.append(\"lm_head\")\n"
    )
    patch(mo, mo_anchor, mo_replacement, "# pixion: NEXTN shared head")

    print("[pixion] patch script done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
