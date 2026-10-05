#!/usr/bin/env python3
"""Normalize Transformers' indexed_attention alias for pinned vLLM nightly.

Only changes the runtime alias; neither checkpoint config nor weights change.
Normalize before QSA layer-ID discovery as well as decoder construction.
"""
import ast
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
marker = '# Pixion: Transformers indexed_attention alias for 0cbac6cd'
if marker not in source:
    anchor = '''class Qwen4ExpModel(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
'''
    if source.count(anchor) != 1:
        raise RuntimeError('Pinned nightly Qwen4ExpModel alias anchor changed')
    replacement = anchor + '''        # Pixion: Transformers indexed_attention alias for 0cbac6cd
        config.layer_types = [
            "qwen_sparse_attention" if t == "indexed_attention" else t
            for t in config.layer_types
        ]
'''
    source = source.replace(anchor, replacement, 1)
    ast.parse(source)
    path.write_text(source)
print('Qwen4Exp indexed_attention runtime alias installed')
