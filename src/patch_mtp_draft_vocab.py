#!/usr/bin/env python3
"""Build-time patch (qwen38-flash-dgx): reduced draft vocabulary for the MTP drafter.

vLLM shares the target model's lm_head with the MTP draft (llm_base_proposer._maybe_share_lm_head),
so every draft step scores all 248,320 vocabulary rows. With
VLLM_MTP_DRAFT_VOCAB=<ids.npy> the draft scores only those rows while the target's lm_head is
untouched. VLLM_MTP_DRAFT_HEAD may supply a matching BF16 or block-FP8 reduced head. The logits
of every other id are -inf, so the proposer's argmax/sampling code is unchanged.
The target still verifies every drafted token, so outputs are identical to full-vocabulary
drafting; only the acceptance rate can move (down, when the target wants an id outside the set).

usage: patch_mtp_draft_vocab.py <path to vllm/models/qwen3_8_flash_next/nvidia/mtp.py>
Inert unless VLLM_MTP_DRAFT_VOCAB is set at runtime.
"""
import sys

TARGET = sys.argv[1]
MARK = "qwen38-flash-dgx: reduced draft vocabulary"
FP8_REDUCED_MARK = "reduced FP8 draft head initialized during model construction"
HOOK = '''

# --- qwen38-flash-dgx: reduced draft vocabulary (VLLM_MTP_DRAFT_VOCAB=<ids.npy>) -------------
import os as _dv_os
from vllm.logger import init_logger as _dv_init_logger

_dv_logger = _dv_init_logger(__name__)


def _dv_prepare_fp8_state(self):
    """Create the FP8 kernel while vLLM's model-init config context is active."""
    import numpy as _np
    from safetensors import safe_open as _dv_safe_open
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    draft_head_path = _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD")
    if not draft_head_path:
        return
    with _dv_safe_open(draft_head_path, framework="pt", device="cpu") as _dv_file:
        if "weight_scale_inv" not in _dv_file.keys():
            return
        wk = _dv_file.get_tensor("weight")
        sk = _dv_file.get_tensor("weight_scale_inv")

    head = self.lm_head
    ids = torch.from_numpy(
        _np.load(_dv_os.environ["VLLM_MTP_DRAFT_VOCAB"]).astype(_np.int64)
    )
    vocab = int(getattr(self.logits_processor, "org_vocab_size", head.weight.shape[0]))
    ids = ids[(ids >= 0) & (ids < vocab)]
    if wk.shape[0] != ids.numel() or wk.shape[1] != head.weight.shape[1]:
        raise RuntimeError(
            f"MTP reduced draft head shape {tuple(wk.shape)} does not match "
            f"ids={ids.numel()} hidden={head.weight.shape[1]}"
        )
    if wk.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"FP8 reduced draft head has unexpected dtype {wk.dtype}")
    expected_scale = (
        (ids.numel() + 127) // 128,
        (head.weight.shape[1] + 127) // 128,
    )
    if tuple(sk.shape) != expected_scale:
        raise RuntimeError(
            f"FP8 reduced draft scale shape {tuple(sk.shape)} does not match "
            f"{expected_scale}"
        )
    source_method = getattr(head, "quant_method", None)
    quant_config = getattr(source_method, "quant_config", None)
    if quant_config is None:
        raise RuntimeError("target FP8 lm_head has no reusable quant_config")

    fp8_head = torch.nn.Module()
    fp8_head.quant_method = Fp8LinearMethod(quant_config)
    fp8_head.quant_method.create_weights(
        fp8_head,
        head.weight.shape[1],
        [ids.numel()],
        head.weight.shape[1],
        ids.numel(),
        params_dtype=self.vllm_config.model_config.dtype,
        weight_loader=None,
    )
    device = head.weight.device
    fp8_head = fp8_head.to(device)
    fp8_head.weight.data.copy_(wk.to(device))
    fp8_head.weight_scale_inv.data.copy_(sk.to(device))
    fp8_head.quant_method.process_weights_after_loading(fp8_head)
    ids = ids.to(device)
    self._dv_state = (ids, fp8_head, vocab, True)
    _dv_logger.info(
        "MTP reduced draft vocabulary: %d of %d rows (%.0f -> %.0f MiB per "
        "draft step, FP8; reduced FP8 draft head initialized during model construction)",
        ids.numel(), vocab,
        head.weight.shape[0] * head.weight.shape[1] * head.weight.element_size() / 2**20,
        wk.numel() * wk.element_size() / 2**20,
    )


_dv_original_init = Qwen3_8FlashNextMTP.__init__


def _dv_init(self, *args, **kwargs):
    _dv_original_init(self, *args, **kwargs)
    if _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB"):
        _dv_prepare_fp8_state(self)


def _dv_compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
    st = getattr(self, "_dv_state", None)
    if st is None:
        import numpy as _np

        head = self.lm_head  # the target's lm_head, shared in by the proposer
        ids = torch.from_numpy(_np.load(_dv_os.environ["VLLM_MTP_DRAFT_VOCAB"]).astype(_np.int64))
        vocab = int(getattr(self.logits_processor, "org_vocab_size", head.weight.shape[0]))
        ids = ids[(ids >= 0) & (ids < vocab)]
        draft_head_path = _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD")
        fp8_head = None
        if draft_head_path:
            from safetensors import safe_open as _dv_safe_open

            with _dv_safe_open(draft_head_path, framework="pt", device="cpu") as _dv_file:
                wk = _dv_file.get_tensor("weight")
                sk = (
                    _dv_file.get_tensor("weight_scale_inv")
                    if "weight_scale_inv" in _dv_file.keys()
                    else None
                )
            if wk.shape[0] != ids.numel() or wk.shape[1] != head.weight.shape[1]:
                raise RuntimeError(
                    f"MTP reduced draft head shape {tuple(wk.shape)} does not match "
                    f"ids={ids.numel()} hidden={head.weight.shape[1]}"
                )
            if sk is None:
                if wk.element_size() == 1:
                    raise RuntimeError(
                        "FP8 reduced draft head is missing weight_scale_inv"
                    )
                wk = wk.to(hidden_states.device)
            else:
                raise RuntimeError(
                    "FP8 reduced draft head was not initialized during model construction"
                )
        else:
            w = head.weight
            if w.element_size() == 1:
                raise RuntimeError(
                    "FP8 lm_head requires VLLM_MTP_DRAFT_HEAD with a matching reduced head"
                )
            ids = ids.to(w.device)
            wk = w.index_select(0, ids).contiguous()
        ids = ids.to(wk.device)
        st = self._dv_state = (ids, wk if fp8_head is None else fp8_head, vocab, fp8_head is not None)
        _dv_logger.info(
            "MTP reduced draft vocabulary: %d of %d rows (%.0f -> %.0f MiB per draft step, %s)",
            ids.numel(), vocab,
            head.weight.shape[0] * head.weight.shape[1] * head.weight.element_size() / 2**20,
            wk.numel() * wk.element_size() / 2**20,
            "FP8" if fp8_head is not None else str(wk.dtype),
        )
    ids, wk, vocab, is_fp8 = st
    red = (
        wk.quant_method.apply(wk, hidden_states)
        if is_fp8
        else torch.nn.functional.linear(hidden_states.to(wk.dtype), wk)
    )
    full = red.new_full((red.shape[0], vocab), float("-inf"))
    full.index_copy_(1, ids, red)
    return full


if _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB"):
    Qwen3_8FlashNextMTP.__init__ = _dv_init  # type: ignore[method-assign]
    Qwen3_8FlashNextMTP.compute_logits = _dv_compute_logits  # type: ignore[method-assign]
    _dv_logger.info("MTP reduced draft vocabulary enabled: %s", _dv_os.environ["VLLM_MTP_DRAFT_VOCAB"])
'''

src = open(TARGET).read()
if MARK in src:
    if FP8_REDUCED_MARK in src:
        print("  draft-vocab hook with reduced FP8-head support already installed")
        sys.exit(0)
    hook_start = src.index("# --- qwen38-flash-dgx: reduced draft vocabulary")
    src = src[:hook_start].rstrip("\n") + HOOK
    print("  existing draft-vocab hook upgraded for reduced FP8 head")
else:
    assert "class Qwen3_8FlashNextMTP(" in src, "MTP class not found"
    src = src.rstrip("\n") + HOOK
open(TARGET, "w").write(src)
import ast; ast.parse(open(TARGET).read())
print("  draft-vocab hook INSTALLED in", TARGET, "(inert unless VLLM_MTP_DRAFT_VOCAB is set)")
