#!/usr/bin/env python3
"""Standalone synthetic test for the isolated FP8 lm_head converter."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    source = root / "source"
    target = root / "target"
    source.mkdir()
    weight = torch.linspace(-2, 2, 256 * 128, dtype=torch.bfloat16).reshape(256, 128)
    other = torch.arange(32, dtype=torch.float32)
    shard = "model-00001-of-00001.safetensors"
    save_file({"lm_head.weight": weight, "other.weight": other}, source / shard)
    index = {
        "metadata": {"total_size": weight.numel() * 2 + other.numel() * 4},
        "weight_map": {"lm_head.weight": shard, "other.weight": shard},
    }
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    config = {
        "quantization_config": {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {},
            "ignore": ["lm_head", "other"],
        }
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "hf_quant_config.json").write_text(json.dumps(config))
    np.save(root / "ids.npy", np.array([0, 127, 128, 255], dtype=np.int64))
    shutil.copytree(source, target, copy_function=os.link)
    subprocess.run(
        [
            "python3",
            str(ROOT / "recipes/nvidia-hybrid/quantize_lm_head_fp8.py"),
            str(target),
            "--draft-vocab",
            str(root / "ids.npy"),
            "--chunk-rows",
            "128",
        ],
        check=True,
    )
    original = load_file(source / shard)
    converted = load_file(target / shard)
    draft = load_file(target / "mtp_draft_head_65536.safetensors")["weight"]
    assert original["lm_head.weight"].dtype == torch.bfloat16
    assert converted["lm_head.weight"].dtype == torch.float8_e4m3fn
    assert converted["lm_head.weight_scale_inv"].shape == (2, 1)
    assert torch.equal(draft, weight[[0, 127, 128, 255]])
    updated = json.loads((target / "config.json").read_text())["quantization_config"]
    assert updated["quantized_layers"]["lm_head"] == {
        "quant_algo": "FP8_PB_WO",
        "group_size": 128,
    }
    assert "lm_head" not in updated["ignore"]
    assert os.stat(source / shard).st_ino != os.stat(target / shard).st_ino
    print("synthetic FP8 lm_head conversion: PASS")
