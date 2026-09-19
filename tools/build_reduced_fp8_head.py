#!/usr/bin/env python3
"""Build a block-FP8 reduced MTP output head from a BF16 lm_head.

The selected rows are quantized with the same 128x128 E4M3 block layout used
by the full FP8 ``lm_head`` checkpoint.  The output contains only ``weight``
and ``weight_scale_inv`` and is consumed by the reduced-vocabulary runtime
hook; the target model checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


BLOCK = 128
FP8_MAX = 448.0


def quantize_blockwise(weight: torch.Tensor, chunk_rows: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    if weight.ndim != 2:
        raise ValueError(f"expected a matrix, got {tuple(weight.shape)}")
    rows, columns = weight.shape
    if rows % BLOCK or columns % BLOCK:
        raise ValueError(f"shape must be divisible by {BLOCK}: {tuple(weight.shape)}")
    chunk_rows = max(BLOCK, chunk_rows - chunk_rows % BLOCK)
    quantized = torch.empty(weight.shape, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows // BLOCK, columns // BLOCK), dtype=torch.float32)
    worst = 0.0
    denominator = weight.float().abs().amax().item()
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        floating = weight[start:end].float().reshape(
            (end - start) // BLOCK, BLOCK, columns // BLOCK, BLOCK
        )
        scale = floating.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / FP8_MAX
        chunk = (floating / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        quantized[start:end] = chunk.reshape(end - start, columns)
        scales[start // BLOCK : end // BLOCK] = scale.squeeze(1).squeeze(-1)
        worst = max(worst, (chunk.float() * scale - floating).abs().amax().item())
        print(f"quantized rows {end:,}/{rows:,}", flush=True)
    return quantized, scales, worst / max(denominator, 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="BF16-head checkpoint directory")
    parser.add_argument("draft_vocab", type=Path, help="selected token IDs (.npy)")
    parser.add_argument("output", type=Path, help="output safetensors path")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()

    root = args.checkpoint.resolve()
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_name = weight_map.get("lm_head.weight")
    if not shard_name:
        raise RuntimeError("lm_head.weight is absent from checkpoint index")
    if "lm_head.weight_scale_inv" in weight_map:
        raise RuntimeError("source lm_head is already FP8; a BF16 source is required")

    with safe_open(root / shard_name, framework="pt", device="cpu") as handle:
        full = handle.get_tensor("lm_head.weight")
    if full.dtype != torch.bfloat16:
        raise RuntimeError(f"expected BF16 lm_head, got {full.dtype}")

    ids_np = np.load(args.draft_vocab).astype(np.int64, copy=False)
    ids = torch.from_numpy(ids_np)
    if ids.ndim != 1 or ids.numel() == 0:
        raise RuntimeError("draft vocabulary must be a non-empty one-dimensional array")
    if int(ids.min()) < 0 or int(ids.max()) >= full.shape[0]:
        raise RuntimeError("draft vocabulary contains out-of-range token IDs")
    if ids.unique().numel() != ids.numel():
        raise RuntimeError("draft vocabulary contains duplicate token IDs")
    if ids.numel() % BLOCK:
        raise RuntimeError(f"draft vocabulary size must be divisible by {BLOCK}")

    selected = full.index_select(0, ids).contiguous()
    quantized, scales, relative_error = quantize_blockwise(selected, args.chunk_rows)
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    save_file(
        {"weight": quantized, "weight_scale_inv": scales},
        temporary,
        metadata={
            "format": "blockwise-fp8-e4m3",
            "block": f"{BLOCK}x{BLOCK}",
            "source": "BF16 lm_head selected rows",
            "vocab_rows": str(ids.numel()),
            "hidden_size": str(full.shape[1]),
            "max_relative_error": f"{relative_error:.8f}",
        },
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    print(
        f"wrote {destination}: weight={tuple(quantized.shape)} {quantized.dtype}, "
        f"scale={tuple(scales.shape)} {scales.dtype}, "
        f"max_relative_error={relative_error:.8f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
