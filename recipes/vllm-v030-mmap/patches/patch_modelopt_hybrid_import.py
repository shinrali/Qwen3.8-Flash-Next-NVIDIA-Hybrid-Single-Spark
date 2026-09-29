#!/usr/bin/env python3
"""Install the opt-in FP8-side dispatcher after ModelOpt classes are defined."""

from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()
marker = "qwen38-v030: mixed ModelOpt FP8-side dispatcher"
if marker not in source:
    source = source.rstrip() + f'''\n\n# --- {marker} ---\nfrom vllm_fp8_hybrid_modelopt import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n'''
    path.write_text(source)
print("mixed ModelOpt FP8-side dispatcher installed in", path)
