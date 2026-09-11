#!/usr/bin/env python3
"""Build-time patch (qwen38-flash-dgx): reduced draft vocabulary for the MTP drafter.

vLLM shares the target model's lm_head with the MTP draft (llm_base_proposer._maybe_share_lm_head),
so every draft step scores all 248,320 vocabulary rows: a 1.27 GiB bf16 read per drafted token,
on a decode step that is memory-bandwidth bound. With VLLM_MTP_DRAFT_VOCAB=<ids.npy> the draft
scores only those rows (a private, sliced copy of the head; the target's lm_head is untouched)
and the logits of every other id are -inf, so the proposer's argmax/sampling code is unchanged.
The target still verifies every drafted token, so outputs are identical to full-vocabulary
drafting; only the acceptance rate can move (down, when the target wants an id outside the set).

usage: patch_mtp_draft_vocab.py <path to vllm/models/qwen3_8_flash_next/nvidia/mtp.py>
Inert unless VLLM_MTP_DRAFT_VOCAB is set at runtime.
"""
import sys

TARGET = sys.argv[1]
MARK = "qwen38-flash-dgx: reduced draft vocabulary"
HOOK = '''

# --- qwen38-flash-dgx: reduced draft vocabulary (VLLM_MTP_DRAFT_VOCAB=<ids.npy>) -------------
import os as _dv_os
from vllm.logger import init_logger as _dv_init_logger

_dv_logger = _dv_init_logger(__name__)


def _dv_compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
    st = getattr(self, "_dv_state", None)
    if st is None:
        import numpy as _np

        head = self.lm_head  # the target's lm_head, shared in by the proposer
        ids = torch.from_numpy(_np.load(_dv_os.environ["VLLM_MTP_DRAFT_VOCAB"]).astype(_np.int64))
        vocab = int(getattr(self.logits_processor, "org_vocab_size", head.weight.shape[0]))
        ids = ids[(ids >= 0) & (ids < vocab)]
        draft_head_path = _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD")
        if draft_head_path:
            from safetensors import safe_open as _dv_safe_open

            with _dv_safe_open(draft_head_path, framework="pt", device="cpu") as _dv_file:
                wk = _dv_file.get_tensor("weight")
            if wk.shape[0] != ids.numel() or wk.shape[1] != head.weight.shape[1]:
                raise RuntimeError(
                    f"MTP reduced draft head shape {tuple(wk.shape)} does not match "
                    f"ids={ids.numel()} hidden={head.weight.shape[1]}"
                )
            wk = wk.to(hidden_states.device)
        else:
            w = head.weight
            if w.element_size() == 1:
                raise RuntimeError(
                    "FP8 lm_head requires VLLM_MTP_DRAFT_HEAD with a BF16 reduced head"
                )
            ids = ids.to(w.device)
            wk = w.index_select(0, ids).contiguous()
        ids = ids.to(wk.device)
        st = self._dv_state = (ids, wk, vocab)
        _dv_logger.info(
            "MTP reduced draft vocabulary: %d of %d rows (%.0f -> %.0f MiB per draft step)",
            ids.numel(), vocab,
            head.weight.shape[0] * head.weight.shape[1] * head.weight.element_size() / 2**20,
            wk.numel() * wk.element_size() / 2**20,
        )
    ids, wk, vocab = st
    red = torch.nn.functional.linear(hidden_states.to(wk.dtype), wk)
    full = red.new_full((red.shape[0], vocab), float("-inf"))
    full.index_copy_(1, ids, red)
    return full


if _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB"):
    Qwen3_8FlashNextMTP.compute_logits = _dv_compute_logits  # type: ignore[method-assign]
    _dv_logger.info("MTP reduced draft vocabulary enabled: %s", _dv_os.environ["VLLM_MTP_DRAFT_VOCAB"])
'''

src = open(TARGET).read()
if MARK in src:
    if "VLLM_MTP_DRAFT_HEAD" in src:
        print("  draft-vocab hook with FP8-head support already installed")
        sys.exit(0)
    hook_start = src.index("# --- qwen38-flash-dgx: reduced draft vocabulary")
    src = src[:hook_start].rstrip("\n") + HOOK
    print("  existing draft-vocab hook upgraded for FP8 lm_head")
else:
    assert "class Qwen3_8FlashNextMTP(" in src, "MTP class not found"
    src = src.rstrip("\n") + HOOK
open(TARGET, "w").write(src)
import ast; ast.parse(open(TARGET).read())
print("  draft-vocab hook INSTALLED in", TARGET, "(inert unless VLLM_MTP_DRAFT_VOCAB is set)")
