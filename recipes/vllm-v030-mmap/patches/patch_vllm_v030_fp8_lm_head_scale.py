#!/usr/bin/env python3
"""Teach vLLM 0.30 to shard block-FP8 ParallelLMHead scales.

The lm_head weight is sharded in vocabulary rows, while its blockwise scale
tensor has one row per output block.  ParallelLMHead.weight_loader therefore
cannot load the companion scale directly: it compares the scale row count to
the unblocked vocabulary size.  Install a scale-specific loader for this one
case without changing ordinary linear layers or the checkpoint.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys


fp8_path = Path(sys.argv[1])
source = fp8_path.read_text()
marker = "qwen38-v030: block-FP8 ParallelLMHead scale loader"

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
        raise RuntimeError("FP8 ParallelLMHead import anchor mismatch")
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
        raise RuntimeError("Fp8LinearMethod class anchor mismatch")
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
        raise RuntimeError("FP8 scale loader insertion anchor mismatch")
    section = section.replace(register_anchor, register_replacement, 1)

    # These are the per-tensor and blockwise weight scale constructors only.
    if section.count("                weight_loader,\n") < 2:
        raise RuntimeError("FP8 scale constructor anchors are absent")
    section = section.replace(
        "                weight_loader,\n",
        "                scale_weight_loader,\n",
        2,
    )
    source = source[:start] + section + source[end:]

    ast.parse(source)
    fp8_path.write_text(source)

print("vLLM 0.30 block-FP8 ParallelLMHead scale loader installed")
