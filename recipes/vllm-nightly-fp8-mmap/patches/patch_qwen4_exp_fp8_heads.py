#!/usr/bin/env python3
"""Enable the FP8 target and reduced MTP heads on vLLM 0.30 Qwen4Exp."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


site = Path(sys.argv[1])
model_path = site / "vllm/models/qwen4_exp/nvidia/model.py"
mtp_path = site / "vllm/models/qwen4_exp/nvidia/mtp.py"
proposer_path = site / "vllm/v1/spec_decode/llm_base_proposer.py"
eagle_utils_path = (
    site / "vllm/v1/worker/gpu/spec_decode/eagle/utils.py"
)


def patch_model() -> None:
    source = model_path.read_text()
    marker = "qwen38-v030: quantized target lm_head"
    if marker in source:
        return

    constructor = '''        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
'''
    replacement = '''        # qwen38-v030: quantized target lm_head
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
'''
    if source.count(constructor) != 1:
        raise RuntimeError("Qwen4Exp target lm_head constructor anchor mismatch")
    source = source.replace(constructor, replacement, 1)

    mapper = '''        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None}
        )
'''
    mapper_replacement = '''        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None}
        )
'''
    if source.count(mapper) != 1:
        raise RuntimeError("Qwen4Exp target weight mapper anchor mismatch")
    source = source.replace(mapper, mapper_replacement, 1)
    ast.parse(source)
    model_path.write_text(source)


def patch_mtp() -> None:
    source = mtp_path.read_text()
    marker = "qwen38-v030: reduced FP8 draft vocabulary"
    if marker in source:
        return

    remap_anchor = '''                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield remapped_name, weight
'''
    remap_replacement = '''                remapped_name = _remap_mtp_weight_name(name)
                # qwen38-v030: the proposer shares the already-loaded target
                # lm_head after MTP construction.  Do not load checkpoint head
                # tensors into this temporary unquantized placeholder.
                if remapped_name is not None and not remapped_name.startswith(
                    "lm_head."
                ):
                    yield remapped_name, weight
'''
    if source.count(remap_anchor) != 1:
        raise RuntimeError("Qwen4Exp MTP lm_head filter anchor mismatch")
    source = source.replace(remap_anchor, remap_replacement, 1)

    hook = r'''

# --- qwen38-v030: reduced FP8 draft vocabulary -------------------------------
import copy as _dv_copy
import os as _dv_os

from safetensors import safe_open as _dv_safe_open
from vllm.logger import init_logger as _dv_init_logger

_dv_logger = _dv_init_logger(__name__)


def _dv_prepare_reduced_head(self):
    vocab_path = _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB")
    head_path = _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD")
    if not vocab_path or not head_path:
        return

    import numpy as _np

    target_head = self.lm_head
    target_method = getattr(target_head, "quant_method", None)
    if target_method is None or not hasattr(target_method, "create_weights"):
        raise RuntimeError("target FP8 lm_head has no reusable quant method")

    ids = torch.from_numpy(_np.load(vocab_path).astype(_np.int64))
    vocab = int(getattr(self.logits_processor, "org_vocab_size", target_head.weight.shape[0]))
    ids = ids[(ids >= 0) & (ids < vocab)]
    with _dv_safe_open(head_path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("weight")
        scale = handle.get_tensor("weight_scale_inv")

    if tuple(weight.shape) != (ids.numel(), target_head.weight.shape[1]):
        raise RuntimeError(
            f"reduced draft head shape {tuple(weight.shape)} does not match "
            f"ids={ids.numel()} hidden={target_head.weight.shape[1]}"
        )
    if weight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"reduced draft head has unexpected dtype {weight.dtype}")

    method = _dv_copy.deepcopy(target_method)
    reduced_head = torch.nn.Module()
    reduced_head.quant_method = method
    with torch.device(target_head.weight.device):
        method.create_weights(
            reduced_head,
            target_head.weight.shape[1],
            [ids.numel()],
            target_head.weight.shape[1],
            ids.numel(),
            params_dtype=getattr(method, "out_dtype", torch.bfloat16),
            weight_loader=None,
        )
    reduced_head.weight.data.copy_(weight.to(reduced_head.weight.device))
    scale_param = getattr(reduced_head, "weight_scale", None)
    if scale_param is None:
        scale_param = getattr(reduced_head, "weight_scale_inv", None)
    if scale_param is None:
        raise RuntimeError("reduced FP8 head quant method created no scale parameter")
    if scale_param.numel() != scale.numel():
        raise RuntimeError(
            f"reduced draft scale shape {tuple(scale.shape)} does not match "
            f"runtime {tuple(scale_param.shape)}"
        )
    scale_param.data.copy_(scale.to(scale_param.device).reshape(scale_param.shape))
    method.process_weights_after_loading(reduced_head)

    self._dv_state = (ids.to(target_head.weight.device), reduced_head, vocab)
    _dv_logger.info(
        "MTP reduced draft vocabulary: %d of %d rows, FP8 head %.0f MiB",
        ids.numel(),
        vocab,
        weight.numel() * weight.element_size() / 2**20,
    )


_dv_original_compute_logits = Qwen4ExpMTP.compute_logits


def _dv_compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
    state = getattr(self, "_dv_state", None)
    if state is None:
        if _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB"):
            raise RuntimeError("reduced MTP draft head was not prepared before decode")
        return _dv_original_compute_logits(self, hidden_states, spec_step_idx)
    ids, reduced_head, vocab = state
    reduced = reduced_head.quant_method.apply(reduced_head, hidden_states)
    full = reduced.new_full((reduced.shape[0], vocab), float("-inf"))
    full.index_copy_(1, ids, reduced)
    return full


Qwen4ExpMTP._dv_prepare_reduced_head = _dv_prepare_reduced_head
Qwen4ExpMTP.compute_logits = _dv_compute_logits
'''
    source = source.rstrip() + hook
    ast.parse(source)
    mtp_path.write_text(source)


def patch_proposer() -> None:
    source = proposer_path.read_text()
    marker = "qwen38-v030: prepare reduced MTP head after target sharing"
    if marker in source:
        return

    anchor = '''            self.model.lm_head = target_language_model.lm_head

            # MTP models call compute_logits via shared_head.head (a
'''
    replacement = '''            self.model.lm_head = target_language_model.lm_head
            # qwen38-v030: prepare reduced MTP head after target sharing
            prepare_reduced_head = getattr(
                self.model, "_dv_prepare_reduced_head", None
            )
            if prepare_reduced_head is not None:
                prepare_reduced_head()

            # MTP models call compute_logits via shared_head.head (a
'''
    if source.count(anchor) != 1:
        raise RuntimeError("MTP lm_head sharing anchor mismatch")
    source = source.replace(anchor, replacement, 1)
    ast.parse(source)
    proposer_path.write_text(source)


def patch_gpu_eagle_loader() -> None:
    source = eagle_utils_path.read_text()
    marker = "qwen38-v030: prepare reduced GPU MTP head after target sharing"
    if marker in source:
        return

    anchor = '''    # MTP shares topk_indices_buffer with the target model. We update
'''
    replacement = '''    # qwen38-v030: prepare reduced GPU MTP head after target sharing
    prepare_reduced_head = getattr(eagle_model, "_dv_prepare_reduced_head", None)
    if prepare_reduced_head is not None:
        prepare_reduced_head()

    # MTP shares topk_indices_buffer with the target model. We update
'''
    if source.count(anchor) != 1:
        raise RuntimeError("GPU Eagle target-head sharing anchor mismatch")
    source = source.replace(anchor, replacement, 1)
    ast.parse(source)
    eagle_utils_path.write_text(source)


patch_model()
patch_mtp()
patch_proposer()
patch_gpu_eagle_loader()
print("Qwen4Exp FP8 target and reduced draft head patches installed")
