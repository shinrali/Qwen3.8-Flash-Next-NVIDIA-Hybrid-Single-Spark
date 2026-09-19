#!/usr/bin/env python3
"""Prepare and attest Pixion's persistent SGLang Qwen4 PLE raw table."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys


SCHEMA = 1
SAMPLE_BYTES = 1 << 20


def model_fingerprint(model_dir: pathlib.Path) -> str:
    """Cheap identity for the immutable local checkpoint used to build PLE."""
    digest = hashlib.sha256()
    for name in ("config.json", "hf_quant_config.json", "model.safetensors.index.json"):
        path = model_dir / name
        if not path.is_file():
            continue
        digest.update(name.encode())
        digest.update(path.read_bytes())
    ple_files = sorted(model_dir.glob("*ple*.safetensors"))
    if not ple_files:
        raise SystemExit(f"no PLE safetensors file under {model_dir}")
    for path in ple_files:
        stat = path.stat()
        digest.update(path.name.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        with path.open("rb", buffering=0) as handle:
            digest.update(handle.read(SAMPLE_BYTES))
            if stat.st_size > SAMPLE_BYTES:
                handle.seek(max(0, stat.st_size - SAMPLE_BYTES))
                digest.update(handle.read(SAMPLE_BYTES))
    return digest.hexdigest()


def table_sample(path: pathlib.Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for offset in sorted({0, max(0, size // 2), max(0, size - SAMPLE_BYTES)}):
            handle.seek(offset)
            digest.update(str(offset).encode())
            digest.update(handle.read(min(SAMPLE_BYTES, size - offset)))
    return digest.hexdigest()


def marker_path(table: pathlib.Path) -> pathlib.Path:
    return table.with_name(f"{table.name}.pixion-complete.json")


def valid_marker(table: pathlib.Path, fingerprint: str) -> bool:
    marker = marker_path(table)
    try:
        payload = json.loads(marker.read_text())
        return (
            payload.get("schema") == SCHEMA
            and payload.get("model_fingerprint") == fingerprint
            and payload.get("table_size") == table.stat().st_size
            and payload.get("sample_sha256") == table_sample(table)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def write_marker(table: pathlib.Path, fingerprint: str) -> pathlib.Path:
    if not table.is_file() or table.stat().st_size <= 0:
        raise SystemExit(f"PLE table is missing or empty: {table}")
    fd = os.open(table, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    marker = marker_path(table)
    payload = {
        "schema": SCHEMA,
        "model_fingerprint": fingerprint,
        "table_size": table.stat().st_size,
        "sample_sha256": table_sample(table),
    }
    temporary = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, marker)
    directory_fd = os.open(marker.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return marker


def tables(table_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(table_dir.glob("ple_table_*.bin")) if table_dir.is_dir() else []


def prepare(table_dir: pathlib.Path, fingerprint: str) -> int:
    found = tables(table_dir)
    if len(found) == 1 and valid_marker(found[0], fingerprint):
        print(f"[pixion] PLE reuse ready: {found[0]}")
        return 0
    # An unmarked, partial, stale, or ambiguous table must be rebuilt from a
    # fresh sparse inode. Rewriting an allocated old file is dramatically slower.
    for table in found:
        marker_path(table).unlink(missing_ok=True)
        table.unlink(missing_ok=True)
    if found:
        print("[pixion] stale/unverified PLE table removed; rebuilding once")
    else:
        print("[pixion] no reusable PLE table; building once")
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(
            f"usage: {sys.argv[0]} fingerprint MODEL_DIR | prepare MODEL_DIR TABLE_DIR | mark MODEL_DIR TABLE"
        )
    command = sys.argv[1]
    model_dir = pathlib.Path(sys.argv[2])
    fingerprint = model_fingerprint(model_dir)
    if command == "fingerprint" and len(sys.argv) == 3:
        print(fingerprint)
        return 0
    if command == "prepare" and len(sys.argv) == 4:
        return prepare(pathlib.Path(sys.argv[3]), fingerprint)
    if command == "mark" and len(sys.argv) == 4:
        marker = write_marker(pathlib.Path(sys.argv[3]), fingerprint)
        print(f"[pixion] PLE reuse marker committed: {marker}")
        return 0
    raise SystemExit("invalid arguments")


if __name__ == "__main__":
    raise SystemExit(main())
