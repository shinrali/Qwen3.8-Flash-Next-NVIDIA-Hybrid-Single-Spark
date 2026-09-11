#!/usr/bin/env python3
"""Restore FP8-side scale headers to the proven vLLM Fp8Config shim format.

Native ModelOpt PbWo fused loading is incompatible with this preview's legacy
MergedColumnParallelLinear loader.  The proven hybrid shim expects
`weight_scale_inv` with a 2-D block grid.  Data bytes remain unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import struct


ROOT = Path(os.environ["FINAL_MODEL_DIR"])
INDEX = ROOT / "model.safetensors.index.json"
NATIVE_SCALE = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj))"
    r"\.weight_scale$"
)


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(length)), 8 + length


def rewrite_shard(path: Path, names: set[str]) -> None:
    header, data_start = read_header(path)
    metadata = header.pop("__metadata__", None)
    if names.difference(header):
        raise RuntimeError(f"missing native scale in {path.name}")
    rewritten: dict[str, object] = {}
    if metadata is not None:
        rewritten["__metadata__"] = metadata
    for name, info in header.items():
        if name not in names:
            rewritten[name] = info
            continue
        shape = info.get("shape")
        if (
            info.get("dtype") != "F32"
            or not isinstance(shape, list)
            or len(shape) != 4
            or shape[1] != 1
            or shape[3] != 1
        ):
            raise RuntimeError(f"unexpected native scale metadata: {name} {info}")
        rewritten[name + "_inv"] = {
            **info,
            "shape": [shape[0], shape[2]],
        }

    raw = json.dumps(rewritten, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    temporary = path.with_name(path.name + ".shim-scale-rewrite")
    if temporary.exists():
        temporary.unlink()
    with path.open("rb") as incoming, temporary.open("xb") as outgoing:
        outgoing.write(struct.pack("<Q", len(raw)))
        outgoing.write(raw)
        incoming.seek(data_start)
        while True:
            chunk = incoming.read(8 * 1024**2)
            if not chunk:
                break
            outgoing.write(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    os.chmod(temporary, path.stat().st_mode & 0o777)
    os.replace(temporary, path)


def main() -> None:
    document = json.loads(INDEX.read_text())
    weight_map = document["weight_map"]
    native = sorted(name for name in weight_map if NATIVE_SCALE.match(name))
    shim = sorted(
        name for name in weight_map
        if name.endswith(".weight_scale_inv")
        and NATIVE_SCALE.match(name.removesuffix("_inv"))
    )
    if len(shim) == 300 and not native:
        print("FP8 side-scale shim contract already present")
        return
    if len(native) != 300 or shim:
        raise RuntimeError(f"expected native=300 shim=0, got {len(native)} / {len(shim)}")
    by_shard: dict[str, set[str]] = {}
    for name in native:
        by_shard.setdefault(weight_map[name], set()).add(name)
    for number, (shard, names) in enumerate(sorted(by_shard.items()), 1):
        rewrite_shard(ROOT / shard, names)
        print(f"[{number}/{len(by_shard)}] {shard}: {len(names)} shim scales ready", flush=True)
    for name in native:
        shard = weight_map.pop(name)
        weight_map[name + "_inv"] = shard
    temporary = INDEX.with_name(INDEX.name + ".shim-scale-rewrite")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.chmod(temporary, 0o644)
    os.replace(temporary, INDEX)
    verified = json.loads(INDEX.read_text())["weight_map"]
    shim = [
        name for name in verified
        if name.endswith(".weight_scale_inv")
        and NATIVE_SCALE.match(name.removesuffix("_inv"))
    ]
    if len(shim) != 300 or any(NATIVE_SCALE.match(name) for name in verified):
        raise RuntimeError("shim scale index verification failed")
    for name in shim:
        header, _ = read_header(ROOT / verified[name])
        info = header[name]
        if info["dtype"] != "F32" or len(info["shape"]) != 2:
            raise RuntimeError(f"shim scale verification failed: {name} {info}")
    (ROOT / ".fp8-side-scale-contract").write_text(
        "loader=vllm Fp8Config hybrid shim\n"
        "scales=300 weight_scale_inv two-dimensional block grids\n"
        "tensor_data_bytes_unchanged=true\n"
    )
    print("REWRITTEN 300 FP8 side scales for hybrid shim loading", flush=True)


if __name__ == "__main__":
    main()
