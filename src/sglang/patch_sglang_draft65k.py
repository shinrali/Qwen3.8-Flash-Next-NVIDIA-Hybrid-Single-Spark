#!/usr/bin/env python3
"""Pixion patch: block-FP8 reduced (hot) MTP draft vocab for SGLang NEXTN/EAGLE-v2.

One fix, anchored + compile-gated:

srt/speculative/eagle_worker_v2.py  EAGLEWorkerV2.init_lm_head()
   The non-eagle3 token-map branch reduces the draft head with
   ``head.data = head.data[self.hot_token_id]``. For a block-FP8 (128x128)
   target lm_head that row-gathers raw FP8 rows WITHOUT remapping
   weight_scale_inv, i.e. it silently produces a mathematically wrong
   65K-row head. This patch adds an opt-in path: when the env var
   SGLANG_MTP_DRAFT_HEAD points at a pre-extracted reduced-vocab safetensors
   (FP8 block weights + weight_scale_inv, row order == token-map order),
   the draft model's lm_head module is replaced with a block-FP8 head
   loaded from that file instead of row-gathering the target FP8 tensor.

   Enable with:
     --speculative-token-map /tmp/hot65536.pt   (int64 d2t map, torch.save)
     SGLANG_MTP_DRAFT_HEAD=/model/<matched-65k-fp8-head>.safetensors

Also (re)writes srt/speculative/pixion_draft_head.py with the builder.

Idempotent (marker scan) and fail-fast (anchor count, compile gate).
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


def write_helper(dirpath: pathlib.Path) -> None:
    helper = dirpath / "pixion_draft_head.py"
    if helper.exists() and "pixion-draft65k-v1" in helper.read_text():
        print(f"[pixion] already present: {helper}")
        return
    helper.write_text(HELPER_SRC)
    try:
        compile(HELPER_SRC, str(helper), "exec")
    except SyntaxError as exc:
        raise SystemExit(f"[pixion] helper source fails syntax gate: {exc}")
    print(f"[pixion] wrote: {helper}")


EAGLE_ANCHOR = (
    "        else:\n"
    "            if self.hot_token_id is not None and head is not None:\n"
    "                head = head.clone()\n"
    "                self.hot_token_id = self.hot_token_id.to(head.device)\n"
    "                head.data = head.data[self.hot_token_id]\n"
    "\n"
    "            # Share the embedding and lm_head\n"
    "            self.draft_runner.model.set_embed_and_head(embed, head)\n"
    "            maybe_share_target_lm_head()\n"
)

EAGLE_REPLACEMENT = (
    "        else:\n"
    "            _pixion_draft_head = None\n"
    "            if self.hot_token_id is not None and head is not None:\n"
    "                import os as _pixion_os\n"
    "\n"
    "                _pixion_draft_head_path = _pixion_os.environ.get(\n"
    "                    \"SGLANG_MTP_DRAFT_HEAD\", \"\"\n"
    "                )\n"
    "                if _pixion_draft_head_path:\n"
    "                    # pixion-draft65k-v1: block-FP8 reduced head from file\n"
    "                    from sglang.srt.speculative.pixion_draft_head import (\n"
    "                        build_pixion_reduced_lm_head,\n"
    "                    )\n"
    "\n"
    "                    _pixion_draft_head = build_pixion_reduced_lm_head(\n"
    "                        target_lm_head, self.hot_token_id, _pixion_draft_head_path\n"
    "                    )\n"
    "                    self.hot_token_id = self.hot_token_id.to(head.device)\n"
    "                else:\n"
    "                    head = head.clone()\n"
    "                    self.hot_token_id = self.hot_token_id.to(head.device)\n"
    "                    head.data = head.data[self.hot_token_id]\n"
    "\n"
    "            # Share the embedding and lm_head\n"
    "            self.draft_runner.model.set_embed_and_head(embed, head)\n"
    "            if _pixion_draft_head is not None:\n"
    "                self.draft_runner.model.lm_head = _pixion_draft_head\n"
    "            maybe_share_target_lm_head()\n"
)

HELPER_SRC = '''"""pixion-draft65k-v1: block-FP8 reduced (hot) draft head for EAGLE/NEXTN.

Builds the draft model lm_head from a pre-extracted reduced-vocab safetensors
instead of row-gathering the (block-FP8) target lm_head weight, which would
break the 128x128 scale grid. Row order of the file MUST equal the d2t order
of --speculative-token-map (hot_token_id), same contract as the vLLM
VLLM_MTP_DRAFT_HEAD runtime.
"""

import copy
import logging

import torch
from safetensors.torch import safe_open

logger = logging.getLogger(__name__)


def build_pixion_reduced_lm_head(target_head, hot_token_id, path):
    """Return a ParallelLMHead-shaped module with len(hot) rows, block-FP8."""
    if target_head is None:
        raise RuntimeError(
            "pixion draft65k: SGLANG_MTP_DRAFT_HEAD requires the target "
            "lm_head module, but it is None"
        )
    weight = getattr(target_head, "weight", None)
    scale = getattr(target_head, "weight_scale_inv", None)
    if weight is None or scale is None:
        raise RuntimeError(
            "pixion draft65k: target lm_head is not block-FP8 "
            "(missing weight and/or weight_scale_inv); only set "
            "SGLANG_MTP_DRAFT_HEAD for block-FP8 lm_head checkpoints"
        )
    device = weight.device
    n_hot = int(hot_token_id.numel())

    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        w_keys = [k for k in keys if k.endswith("weight")]
        s_keys = [k for k in keys if k.endswith("weight_scale_inv")]
        if len(w_keys) != 1 or len(s_keys) != 1:
            raise RuntimeError(
                f"pixion draft65k: {path} must contain exactly one *weight "
                f"and one *weight_scale_inv, got {keys}"
            )
        w = f.get_tensor(w_keys[0])
        s = f.get_tensor(s_keys[0])

    if w.dim() != 2 or int(w.shape[0]) != n_hot:
        raise RuntimeError(
            f"pixion draft65k: head rows {tuple(w.shape)} != d2t map size {n_hot}"
        )
    if w.shape[1] != weight.shape[1]:
        raise RuntimeError(
            f"pixion draft65k: head dim {w.shape[1]} != target hidden {weight.shape[1]}"
        )
    want_scale = ((n_hot + 127) // 128, (int(w.shape[1]) + 127) // 128)
    if tuple(s.shape) != want_scale:
        raise RuntimeError(
            f"pixion draft65k: scale grid {tuple(s.shape)} != {want_scale}"
        )
    if w.dtype != weight.data.dtype:
        raise RuntimeError(
            f"pixion draft65k: file weight dtype {w.dtype} != target "
            f"{weight.data.dtype}; row-gather fallback would be the only "
            f"option but it is invalid for block-FP8"
        )

    target_rows = int(weight.shape[0])
    mod = copy.copy(target_head)
    # torch.nn.Module's shallow copy shares its registration dictionaries.
    # Detach them before replacing the draft parameters, otherwise pop/register
    # below also mutates the target model's full-vocabulary lm_head.
    mod._parameters = target_head._parameters.copy()
    mod._buffers = target_head._buffers.copy()
    mod._modules = target_head._modules.copy()
    mod._non_persistent_buffers_set = target_head._non_persistent_buffers_set.copy()
    for holder in (mod._parameters, mod._buffers):
        holder.pop("weight", None)
        holder.pop("weight_scale_inv", None)
    mod.register_parameter(
        "weight", torch.nn.Parameter(w.to(device=device), requires_grad=False)
    )
    mod.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(s.to(device=device), requires_grad=False),
    )
    mod.num_embeddings = int(w.shape[0])
    mod.num_embeddings_padded = int(w.shape[0])
    mod.org_vocab_size = int(w.shape[0])

    if mod._parameters is target_head._parameters:
        raise RuntimeError("pixion draft65k: draft and target parameter maps still alias")
    if int(target_head.weight.shape[0]) != target_rows:
        raise RuntimeError(
            "pixion draft65k: target lm_head was mutated while installing draft head"
        )

    logger.warning(
        "pixion draft65k: reduced MTP draft head installed from %s: "
        "%d of %d rows, dtype %s, scale grid %s (%d MiB)",
        path,
        w.shape[0],
        weight.shape[0],
        w.dtype,
        tuple(s.shape),
        (w.numel() + s.numel() * 2) // (2**20),
    )
    return mod
'''


def main() -> int:
    eagle = (
        pathlib.Path(sys.argv[1])
        if len(sys.argv) > 1
        else find_file("srt/speculative/eagle_worker_v2.py")
    )
    patch(eagle, EAGLE_ANCHOR, EAGLE_REPLACEMENT, "pixion-draft65k-v1")
    write_helper(eagle.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
