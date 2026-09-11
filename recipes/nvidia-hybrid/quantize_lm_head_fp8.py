#!/usr/bin/env python3
"""Convert only ``lm_head.weight`` from BF16 to 128x128 block FP8.

The checkpoint directory must be an isolated copy/hard-link clone.  The source
shard is never edited in place: this script writes a replacement next to it and
atomically swaps the directory entry, which breaks any hard link to the source
checkpoint.  It also writes a BF16 reduced-vocabulary head for the existing MTP
draft-vocabulary optimization.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


BLOCK = 128
FP8_MAX = 448.0
HEAD = "lm_head.weight"
SCALE = "lm_head.weight_scale_inv"
DRAFT_TENSOR = "weight"


def atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_name(path.name + ".fp8head.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.chmod(temporary, path.stat().st_mode & 0o777)
    os.replace(temporary, path)


def quantize_blockwise(weight: torch.Tensor, chunk_rows: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    if weight.ndim != 2:
        raise ValueError(f"expected a matrix, got {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if out_features % BLOCK or in_features % BLOCK:
        raise ValueError(
            f"lm_head shape must be divisible by {BLOCK} in both dimensions: "
            f"{tuple(weight.shape)}"
        )
    chunk_rows = max(BLOCK, chunk_rows - chunk_rows % BLOCK)
    quantized = torch.empty(weight.shape, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (out_features // BLOCK, in_features // BLOCK), dtype=torch.float32
    )
    worst = 0.0
    denominator = weight.float().abs().amax().item()
    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        floating = weight[start:end].float().reshape(
            (end - start) // BLOCK, BLOCK, in_features // BLOCK, BLOCK
        )
        scale = floating.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / FP8_MAX
        chunk = (floating / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        quantized[start:end] = chunk.reshape(end - start, in_features)
        scales[start // BLOCK : end // BLOCK] = scale.squeeze(1).squeeze(-1)
        reconstructed = chunk.float() * scale
        worst = max(worst, (reconstructed - floating).abs().amax().item())
        print(f"quantized rows {end:,}/{out_features:,}", flush=True)
    relative_error = worst / max(denominator, 1e-12)
    return quantized, scales, relative_error


def update_quantization_metadata(path: Path) -> None:
    document = json.loads(path.read_text())
    quant = document.get("quantization_config", document.get("quantization"))
    if not isinstance(quant, dict) or quant.get("quant_algo") != "MIXED_PRECISION":
        raise RuntimeError(f"unexpected quantization metadata in {path}")
    layers = quant.get("quantized_layers")
    if not isinstance(layers, dict):
        raise RuntimeError(f"missing quantized_layers in {path}")
    layers["lm_head"] = {"quant_algo": "FP8_PB_WO", "group_size": BLOCK}
    for key in ("ignore", "exclude_modules"):
        ignored = quant.get(key)
        if isinstance(ignored, list):
            quant[key] = [entry for entry in ignored if entry != "lm_head"]
    atomic_json(path, document)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--draft-vocab", required=True, type=Path)
    parser.add_argument(
        "--draft-head-name", default="mtp_draft_head_65536.safetensors"
    )
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()

    root = args.checkpoint.resolve()
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    if SCALE in weight_map:
        raise RuntimeError("lm_head already has weight_scale_inv")
    shard_name = weight_map.get(HEAD)
    if not shard_name:
        raise RuntimeError("lm_head.weight is absent from the checkpoint index")
    shard_path = root / shard_name

    tensors = load_file(shard_path)
    weight = tensors.pop(HEAD)
    if weight.dtype != torch.bfloat16:
        raise RuntimeError(f"expected BF16 lm_head, got {weight.dtype}")
    print(
        f"lm_head {tuple(weight.shape)} {weight.dtype}, "
        f"{weight.numel() * weight.element_size() / 2**30:.3f} GiB",
        flush=True,
    )

    ids = torch.from_numpy(np.load(args.draft_vocab).astype(np.int64, copy=False))
    ids = ids[(ids >= 0) & (ids < weight.shape[0])]
    if ids.unique().numel() != ids.numel():
        raise RuntimeError("draft vocabulary contains duplicate ids")
    draft = weight.index_select(0, ids).contiguous()
    draft_path = root / args.draft_head_name
    draft_tmp = draft_path.with_name(draft_path.name + ".tmp")
    save_file({DRAFT_TENSOR: draft}, draft_tmp)
    os.replace(draft_tmp, draft_path)
    print(
        f"wrote BF16 draft head {tuple(draft.shape)} "
        f"({draft.numel() * draft.element_size() / 2**20:.1f} MiB)",
        flush=True,
    )
    del draft

    quantized, scales, relative_error = quantize_blockwise(weight, args.chunk_rows)
    old_bytes = weight.numel() * weight.element_size()
    new_bytes = quantized.numel() * quantized.element_size() + scales.numel() * scales.element_size()
    tensors[HEAD] = quantized
    tensors[SCALE] = scales

    replacement = shard_path.with_name(shard_path.name + ".fp8head.tmp")
    save_file(tensors, replacement)
    os.chmod(replacement, shard_path.stat().st_mode & 0o777)
    os.replace(replacement, shard_path)

    weight_map[SCALE] = shard_name
    metadata = index.setdefault("metadata", {})
    if "total_size" in metadata:
        metadata["total_size"] = int(metadata["total_size"]) - old_bytes + new_bytes
    atomic_json(index_path, index)
    for filename in ("config.json", "hf_quant_config.json"):
        path = root / filename
        if path.exists():
            update_quantization_metadata(path)

    marker = root / "FP8_LM_HEAD.txt"
    marker.write_text(
        f"lm_head=blockwise-fp8-e4m3\n"
        f"block={BLOCK}x{BLOCK}\n"
        f"shape={weight.shape[0]}x{weight.shape[1]}\n"
        f"scale_shape={scales.shape[0]}x{scales.shape[1]}\n"
        f"draft_head={args.draft_head_name}\n"
        f"draft_rows={ids.numel()}\n"
        f"max_relative_error={relative_error:.8f}\n"
    )
    print(
        f"done: saved {(old_bytes - new_bytes) / 2**20:.1f} MiB, "
        f"max relative error {relative_error:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
