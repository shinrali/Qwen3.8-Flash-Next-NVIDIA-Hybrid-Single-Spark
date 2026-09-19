#!/usr/bin/env python3
"""Extract a BF16 reduced MTP draft head from a sharded HF checkpoint.

The row order is exactly the order stored in ``draft_vocab.npy``.  The runtime
vocabulary patch sorts/scatters using the same array, so the two files must
always be deployed as a matched pair.
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


HEAD = "lm_head.weight"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("draft_vocab", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    root = args.checkpoint.resolve()
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shard_name = index.get("weight_map", {}).get(HEAD)
    if not shard_name:
        raise RuntimeError(f"{HEAD} is absent from checkpoint index")

    ids = np.load(args.draft_vocab).astype(np.int64, copy=False)
    if ids.ndim != 1 or ids.size == 0:
        raise RuntimeError("draft vocabulary must be a non-empty 1D array")
    if np.unique(ids).size != ids.size:
        raise RuntimeError("draft vocabulary contains duplicate IDs")

    with safe_open(root / shard_name, framework="pt", device="cpu") as source:
        weight = source.get_tensor(HEAD)
    if weight.dtype != torch.bfloat16:
        raise RuntimeError(f"expected BF16 {HEAD}, got {weight.dtype}")
    if ids.min() < 0 or ids.max() >= weight.shape[0]:
        raise RuntimeError(
            f"draft vocabulary range {ids.min()}..{ids.max()} exceeds head rows {weight.shape[0]}"
        )

    selected = weight.index_select(0, torch.from_numpy(ids)).contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    save_file({"weight": selected}, temporary)
    os.replace(temporary, args.output)
    os.chmod(args.output, 0o600)
    print(
        f"wrote {args.output}: rows={selected.shape[0]} hidden={selected.shape[1]} "
        f"dtype={selected.dtype} bytes={args.output.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
