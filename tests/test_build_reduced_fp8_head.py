#!/usr/bin/env python3
"""Standalone synthetic test for the block-FP8 reduced MTP head builder."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    weight = torch.linspace(-3, 3, 256 * 128, dtype=torch.bfloat16).reshape(256, 128)
    shard = "model-00001-of-00001.safetensors"
    save_file({"lm_head.weight": weight}, checkpoint / shard)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": shard}})
    )
    ids = np.arange(127, -1, -1, dtype=np.int32)
    np.save(root / "draft_vocab.npy", ids)
    output = root / "mtp_draft_head_fp8.safetensors"

    subprocess.run(
        [
            "python3",
            str(ROOT / "tools/build_reduced_fp8_head.py"),
            str(checkpoint),
            str(root / "draft_vocab.npy"),
            str(output),
            "--chunk-rows",
            "128",
        ],
        check=True,
    )

    with safe_open(output, framework="pt", device="cpu") as handle:
        quantized = handle.get_tensor("weight")
        scales = handle.get_tensor("weight_scale_inv")
        metadata = handle.metadata()
    reconstructed = (
        quantized.float().reshape(1, 128, 1, 128)
        * scales.reshape(1, 1, 1, 1)
    ).reshape(128, 128)
    expected = weight[torch.from_numpy(ids.astype(np.int64))].float()

    assert quantized.shape == (128, 128)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (1, 1)
    assert scales.dtype == torch.float32
    assert torch.isfinite(scales).all() and (scales > 0).all()
    assert torch.equal(torch.from_numpy(ids), torch.arange(127, -1, -1, dtype=torch.int32))
    relative_error = (reconstructed - expected).abs().max() / expected.abs().max()
    assert relative_error <= 0.04
    assert metadata["format"] == "blockwise-fp8-e4m3"
    assert metadata["block"] == "128x128"
    print("synthetic reduced FP8 MTP head conversion: PASS")
