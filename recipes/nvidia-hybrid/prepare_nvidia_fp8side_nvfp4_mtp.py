#!/usr/bin/env python3
"""Prepare NVIDIA NVFP4 main + FP8 side layers + NVFP4 MTP.

The source is the already locally converted, isolated NVIDIA FP8-side
checkpoint.  No source checkpoint or base image is modified.  Referenced
files are hard-linked, NVIDIA's block-FP8 MTP expert tensors are removed from
the mixed PLE shard, and the pinned Inferact NVFP4 MTP expert shard is grafted
in.  ModelOpt MIXED_PRECISION metadata is made explicit for all 300 converted
side-layer linears so vLLM uses its native per-prefix dispatcher instead of
the older runtime hybrid shim.
"""

from __future__ import annotations

from fnmatch import fnmatch
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct


MODELS = Path(os.environ["MODEL_ROOT"])
SOURCE = Path(
    os.environ.get(
        "FP8_SIDE_MODEL_DIR",
        MODELS / "Qwen3.8-Flash-Next-NVIDIA-FP8-Hybrid",
    )
)
DESTINATION = Path(
    os.environ.get(
        "FINAL_MODEL_DIR",
        MODELS / "Qwen3.8-Flash-Next-NVIDIA-FP8-Hybrid-MTPNVFP4",
    )
)
TEMP = DESTINATION.with_name(f".{DESTINATION.name}.preparing")
MIXED_SHARD = "model-fp8-mtp-ple.safetensors"
DONOR = Path(
    os.environ.get(
        "NVFP4_MTP_DONOR",
        MODELS
        / "Qwen3.8-Flash-Next-Inferact-NVFP4-MTP"
        / "nvfp4_experts_mtp.safetensors",
    )
)
DONOR_NAME = "nvfp4_experts_mtp.safetensors"
DONOR_SHA256 = "0d44e6d705d2313c713e60114e56874adf358ed5f646dc8704bb5be15f5ddbf7"

OLD_EXPERT = re.compile(
    r"^mtp\.layers\.0\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.(?:weight|weight_scale_inv)$"
)
NEW_EXPERT = re.compile(r"^mtp\.layers\.0\.mlp\.experts\.\d+\.")
SIDE_WEIGHT = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj))\.weight$"
)


def link_or_copy(source: Path, destination: Path) -> str:
    """Hard-link when permitted; copy root-owned converted shards otherwise."""
    try:
        os.link(source, destination)
        return "linked"
    except OSError as error:
        if error.errno not in {errno.EPERM, errno.EXDEV}:
            raise
        shutil.copy2(source, destination)
        return "copied"


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(length)), 8 + length


def write_filtered_shard(source: Path, destination: Path) -> tuple[int, int, int]:
    header, source_data_start = read_header(source)
    metadata = header.pop("__metadata__", {"format": "pt"})
    removed = [name for name in header if OLD_EXPERT.match(name)]
    if len(removed) != 3072:
        raise RuntimeError(f"expected 3072 NVIDIA FP8 MTP tensors, got {len(removed)}")

    removed_set = set(removed)
    removed_bytes = sum(
        header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
        for name in removed
    )
    kept = [(name, value) for name, value in header.items() if name not in removed_set]
    new_header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for name, value in kept:
        start, end = value["data_offsets"]
        size = end - start
        new_header[name] = {**value, "data_offsets": [offset, offset + size]}
        offset += size

    raw = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        outgoing.write(struct.pack("<Q", len(raw)))
        outgoing.write(raw)
        copied = 0
        for number, (name, value) in enumerate(kept, 1):
            start, end = value["data_offsets"]
            incoming.seek(source_data_start + start)
            remaining = end - start
            while remaining:
                chunk = incoming.read(min(8 * 1024**2, remaining))
                if not chunk:
                    raise RuntimeError(f"unexpected EOF while copying {name}")
                outgoing.write(chunk)
                copied += len(chunk)
                remaining -= len(chunk)
            if number % 32 == 0 or number == len(kept):
                print(
                    f"mixed shard: {number}/{len(kept)} tensors, "
                    f"{copied / 2**30:.2f} GiB copied",
                    flush=True,
                )
        outgoing.flush()
        os.fsync(outgoing.fileno())

    if destination.stat().st_size != 8 + len(raw) + offset:
        raise RuntimeError("repacked mixed shard size mismatch")
    os.chmod(destination, 0o644)
    return len(removed), len(kept), removed_bytes


def side_prefixes(index: dict[str, str]) -> list[str]:
    prefixes = sorted(
        name.removesuffix(".weight") for name in index if SIDE_WEIGHT.match(name)
    )
    if len(prefixes) != 300:
        raise RuntimeError(f"expected 300 converted side-layer weights, got {len(prefixes)}")
    return prefixes


def update_quant_config(path: Path, prefixes: list[str]) -> tuple[int, int]:
    document = json.loads(path.read_text())
    quant = document.get("quantization_config", document.get("quantization"))
    if not isinstance(quant, dict) or quant.get("quant_algo") != "MIXED_PRECISION":
        raise RuntimeError(f"unexpected quantization metadata in {path.name}")
    layers = quant.get("quantized_layers")
    if not isinstance(layers, dict):
        raise RuntimeError(f"missing quantized_layers in {path.name}")

    old_mtp = layers.get("mtp.layers.0.mlp.experts")
    if not isinstance(old_mtp, dict) or old_mtp.get("quant_algo") not in {
        "FP8_PB_WO",
        "FP8_BLOCK_SCALES",
    }:
        raise RuntimeError(f"unexpected source MTP quant entry in {path.name}: {old_mtp!r}")
    layers["mtp.layers.0.mlp.experts"] = {
        "quant_algo": "NVFP4",
        "group_size": 16,
    }
    for prefix in prefixes:
        layers[prefix] = {"quant_algo": "FP8_PB_WO", "group_size": 128}

    ignore_key = "ignore" if "ignore" in quant else "exclude_modules"
    ignored = quant.get(ignore_key, [])
    if not isinstance(ignored, list):
        raise RuntimeError(f"{ignore_key} is not a list in {path.name}")
    removed_patterns = [
        pattern for pattern in ignored if any(fnmatch(prefix, pattern) for prefix in prefixes)
    ]
    quant[ignore_key] = [pattern for pattern in ignored if pattern not in removed_patterns]
    if len(removed_patterns) != 96:
        raise RuntimeError(
            f"expected 96 side-layer ignore patterns in {path.name}, got {len(removed_patterns)}"
        )
    if any(
        fnmatch(prefix, pattern)
        for prefix in prefixes
        for pattern in quant[ignore_key]
    ):
        raise RuntimeError(f"a converted side layer is still excluded in {path.name}")

    path.write_text(json.dumps(document, indent=2) + "\n")
    os.chmod(path, 0o644)
    return len(prefixes), len(removed_patterns)


def verify(root: Path) -> None:
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefixes = side_prefixes(index)
    old = [
        name for name, shard in index.items()
        if OLD_EXPERT.match(name) and shard != DONOR_NAME
    ]
    new = [
        name for name, shard in index.items()
        if shard == DONOR_NAME and NEW_EXPERT.match(name)
    ]
    if old:
        raise RuntimeError(f"old NVIDIA FP8 MTP keys remain in index: {old[:3]}")
    if len(new) != 6144:
        raise RuntimeError(f"expected 6144 donor NVFP4 keys in index, got {len(new)}")

    headers: dict[str, dict] = {}
    for shard in set(index.values()):
        header, _ = read_header(root / shard)
        header.pop("__metadata__", None)
        headers[shard] = header
    missing = [name for name, shard in index.items() if name not in headers[shard]]
    if missing:
        raise RuntimeError(f"index references absent tensors: {missing[:3]}")

    for prefix in prefixes:
        weight = prefix + ".weight"
        scale = prefix + ".weight_scale_inv"
        weight_info = headers[index[weight]][weight]
        if weight_info["dtype"] not in {"F8_E4M3", "F8_E4M3FN"}:
            raise RuntimeError(f"side weight is not FP8: {weight} {weight_info['dtype']}")
        if scale not in index or headers[index[scale]][scale]["dtype"] != "F32":
            raise RuntimeError(f"side scale is absent or not F32: {scale}")

    for filename in ("config.json", "hf_quant_config.json"):
        document = json.loads((root / filename).read_text())
        quant = document.get("quantization_config", document.get("quantization"))
        layers = quant["quantized_layers"]
        if layers["mtp.layers.0.mlp.experts"] != {
            "quant_algo": "NVFP4",
            "group_size": 16,
        }:
            raise RuntimeError(f"wrong NVFP4 MTP metadata in {filename}")
        wrong = [
            prefix for prefix in prefixes
            if layers.get(prefix) != {"quant_algo": "FP8_PB_WO", "group_size": 128}
        ]
        if wrong:
            raise RuntimeError(f"wrong FP8 side-layer metadata in {filename}: {wrong[:3]}")
        ignored = quant.get("ignore", quant.get("exclude_modules", []))
        if any(fnmatch(prefix, pattern) for prefix in prefixes for pattern in ignored):
            raise RuntimeError(f"converted side layer remains excluded in {filename}")


def main() -> None:
    if DESTINATION.exists():
        raise SystemExit(f"refusing to overwrite existing destination: {DESTINATION}")
    if TEMP.exists():
        required = {
            MIXED_SHARD,
            DONOR_NAME,
            "model.safetensors.index.json",
            "config.json",
            "hf_quant_config.json",
        }
        if not required.issubset({path.name for path in TEMP.iterdir()}):
            raise SystemExit(f"incomplete preparation directory requires inspection: {TEMP}")
        verify(TEMP)
        os.rename(TEMP, DESTINATION)
        print(f"PREPARED {DESTINATION}: resumed static verification passed", flush=True)
        return

    for required in (
        SOURCE / "model.safetensors.index.json",
        SOURCE / MIXED_SHARD,
        SOURCE / "config.json",
        SOURCE / "hf_quant_config.json",
        DONOR,
    ):
        if not required.is_file():
            raise SystemExit(f"required file missing: {required}")
    with DONOR.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != DONOR_SHA256:
        raise SystemExit(f"donor SHA-256 mismatch: {digest}")
    if shutil.disk_usage(MODELS).free < 64 * 2**30:
        raise SystemExit("at least 64 GiB free is required")

    TEMP.mkdir(mode=0o755)
    excluded = {
        MIXED_SHARD,
        "model.safetensors.index.json",
        "config.json",
        "hf_quant_config.json",
    }
    for source in SOURCE.iterdir():
        if not source.is_file() or source.name in excluded or source.name.endswith(".bf16.bak"):
            continue
        mode = link_or_copy(source, TEMP / source.name)
        if mode == "copied":
            print(f"copied protected source file: {source.name}", flush=True)
    for filename in ("model.safetensors.index.json", "config.json", "hf_quant_config.json"):
        shutil.copy2(SOURCE / filename, TEMP / filename)

    index_path = TEMP / "model.safetensors.index.json"
    index_document = json.loads(index_path.read_text())
    weight_map = index_document["weight_map"]
    prefixes = side_prefixes(weight_map)

    removed, kept, removed_bytes = write_filtered_shard(
        SOURCE / MIXED_SHARD, TEMP / MIXED_SHARD
    )
    link_or_copy(DONOR, TEMP / DONOR_NAME)
    removed_index = [name for name in weight_map if OLD_EXPERT.match(name)]
    if len(removed_index) != removed:
        raise RuntimeError(
            f"mixed shard removed {removed} tensors but index selected {len(removed_index)}"
        )
    for name in removed_index:
        del weight_map[name]

    donor_header, _ = read_header(DONOR)
    donor_header.pop("__metadata__", None)
    donor_experts = [name for name in donor_header if NEW_EXPERT.match(name)]
    if len(donor_experts) != 6144:
        raise RuntimeError(f"expected 6144 donor expert tensors, got {len(donor_experts)}")
    for name in donor_experts:
        weight_map[name] = DONOR_NAME
    donor_bytes = sum(
        donor_header[name]["data_offsets"][1] - donor_header[name]["data_offsets"][0]
        for name in donor_experts
    )
    metadata = index_document.setdefault("metadata", {})
    if isinstance(metadata.get("total_size"), int):
        metadata["total_size"] = metadata["total_size"] - removed_bytes + donor_bytes
    index_path.write_text(json.dumps(index_document, indent=2) + "\n")
    os.chmod(index_path, 0o644)

    results = [
        update_quant_config(TEMP / filename, prefixes)
        for filename in ("config.json", "hf_quant_config.json")
    ]
    verify(TEMP)
    (TEMP / ".prepared-nvfp4-mtp").write_text(
        "source=nvidia/Qwen3.8-Flash-Next-NVFP4@fc694b54fb0174e0913e6adf86691ef85a4ead47\n"
        "side_layers=300 locally converted FP8_PB_WO group128\n"
        f"mtp_donor_sha256={DONOR_SHA256}\n"
        "mtp=Inferact per-expert NVFP4 group16\n"
        "source_modified=false\n"
    )
    os.rename(TEMP, DESTINATION)
    print(
        f"PREPARED {DESTINATION}: side={results[0][0]} FP8, "
        f"removed_ignore_patterns={results[0][1]}, removed_mtp={removed}, "
        f"kept_mixed={kept}, added_mtp={len(donor_experts)} NVFP4",
        flush=True,
    )


if __name__ == "__main__":
    main()
