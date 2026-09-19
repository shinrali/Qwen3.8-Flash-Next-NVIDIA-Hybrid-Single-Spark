#!/usr/bin/env python3
"""Add block-FP8 ParallelLMHead support to the pinned Qwen3.8 vLLM image.

This is a narrow port of vllm-project/vllm#41000's companion-scale loader,
plus the missing ``quant_config`` plumbing in Qwen3Next's target and MTP model
classes.  The runtime hybrid shim decides whether a particular head is FP8.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys


site = Path(sys.argv[1])
fp8_path = site / "vllm/model_executor/layers/quantization/fp8.py"
model_path = site / "vllm/models/qwen3_8_flash_next/nvidia/model.py"
mtp_path = site / "vllm/models/qwen3_8_flash_next/nvidia/mtp.py"

source = fp8_path.read_text()
marker = "qwen38-fp8-lm-head: ParallelLMHead companion-scale loader"
if marker not in source:
    import_anchor = """from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
"""
    import_replacement = import_anchor + """from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
"""
    if source.count(import_anchor) != 1:
        raise RuntimeError("FP8 import anchor did not match exactly once")
    source = source.replace(import_anchor, import_replacement, 1)

    class_anchor = "class Fp8LinearMethod(LinearMethodBase):\n"
    helper = f'''# {marker}
def _make_lm_head_block_scale_loader(layer, block_size):
    block_out = block_size[0]

    def load(param, loaded_weight):
        start = layer.shard_indices.org_vocab_start_index
        if start % block_out:
            raise ValueError(
                f"FP8 lm_head shard start {{start}} is not divisible by {{block_out}}"
            )
        start_idx = start // block_out
        local_rows = param.shape[0]
        if loaded_weight.shape[0] < start_idx + local_rows:
            raise ValueError(
                f"FP8 lm_head scale has {{loaded_weight.shape[0]}} rows; "
                f"need {{start_idx + local_rows}}"
            )
        param.data.copy_(loaded_weight.narrow(0, start_idx, local_rows))

    return load


'''
    if source.count(class_anchor) != 1:
        raise RuntimeError("Fp8LinearMethod anchor did not match exactly once")
    source = source.replace(class_anchor, helper + class_anchor, 1)

    start = source.index(class_anchor)
    end = source.index("class Fp8MoEMethod", start)
    section = source[start:end]
    register_anchor = '''        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
'''
    register_replacement = '''        layer.register_parameter("weight", weight)

        scale_weight_loader = weight_loader
        if isinstance(layer, ParallelLMHead) and self.block_quant:
            scale_weight_loader = _make_lm_head_block_scale_loader(
                layer, self.weight_block_size
            )

        # WEIGHT SCALE
'''
    if section.count(register_anchor) != 1:
        raise RuntimeError("FP8 scale-loader insertion anchor did not match once")
    section = section.replace(register_anchor, register_replacement, 1)
    # Only the two weight-scale constructors in Fp8LinearMethod are changed.
    if section.count("                weight_loader,\n") < 2:
        raise RuntimeError("FP8 scale weight_loader anchors are absent")
    section = section.replace(
        "                weight_loader,\n",
        "                scale_weight_loader,\n",
        2,
    )
    source = source[:start] + section + source[end:]
    ast.parse(source)
    fp8_path.write_text(source)


def add_quant_config(path: Path) -> None:
    text = path.read_text()
    constructor = '''                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
'''
    direct = '''        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
'''
    if "quant_config=vllm_config.quant_config" not in text:
        if constructor in text:
            text = text.replace(
                constructor,
                constructor.replace(
                    '                    prefix=maybe_prefix(prefix, "lm_head"),',
                    '                    quant_config=vllm_config.quant_config,\n'
                    '                    prefix=maybe_prefix(prefix, "lm_head"),',
                ),
                1,
            )
        elif direct in text:
            text = text.replace(
                direct,
                direct.replace(
                    '            prefix=maybe_prefix(prefix, "lm_head"),',
                    '            quant_config=vllm_config.quant_config,\n'
                    '            prefix=maybe_prefix(prefix, "lm_head"),',
                ),
                1,
            )
        else:
            raise RuntimeError(f"ParallelLMHead constructor anchor absent in {path}")
    ast.parse(text)
    path.write_text(text)


add_quant_config(model_path)
add_quant_config(mtp_path)
print("FP8 lm_head loader and Qwen3Next quant_config plumbing installed")
