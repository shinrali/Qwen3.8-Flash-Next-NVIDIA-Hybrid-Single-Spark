#!/usr/bin/env python3
"""Pixion patch: reuse a verified, complete SGLang Qwen4 raw PLE table.

Stock v0.5.20 keeps the deterministic mmap file but copies all 512 checkpoint
shards into it on every boot.  This patch skips those copies only when an
atomic completion marker matches the current checkpoint fingerprint and table
sample.  A first build (or any stale/partial marker) still takes the ordinary
loader path and commits the marker only after every expected shard was seen.
"""

import glob
import importlib.util
import pathlib
import sys
import sysconfig


MARKER = "pixion-verified-ple-reuse-v1"


def find_file(rel: str) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    try:
        spec = importlib.util.find_spec("sglang")
        if spec is not None and spec.submodule_search_locations:
            candidates.extend(pathlib.Path(p) for p in spec.submodule_search_locations)
    except Exception:
        pass
    for key in ("purelib", "platlib"):
        try:
            candidates.append(pathlib.Path(sysconfig.get_paths()[key]) / "sglang")
        except Exception:
            pass
    for pattern in (
        "/opt/*/lib/python3*/site-packages/sglang",
        "/usr/local/lib/python3*/site-packages/sglang",
        "/usr/lib/python3*/site-packages/sglang",
        "/sgl-workspace/*/python/sglang",
    ):
        candidates.extend(pathlib.Path(p) for p in glob.glob(pattern))
    seen: set[pathlib.Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = candidate / rel
        if path.is_file():
            return path
    raise FileNotFoundError(f"cannot locate sglang/{rel} in {sorted(map(str, seen))}")


def main() -> int:
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) == 2 else find_file("srt/models/qwen4_exp.py")
    source = path.read_text()
    if MARKER in source:
        print(f"[pixion] already patched: {path}")
        return 0

    imports_old = '''import math
from contextlib import nullcontext
'''
    imports_new = '''import hashlib
import json
import math
import os
import pathlib
from contextlib import nullcontext
'''
    if source.count(imports_old) != 1:
        raise SystemExit("[pixion] import anchor count != 1; aborting")
    source = source.replace(imports_old, imports_new)

    helpers_anchor = '''_QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD = 1024


'''
    helpers = '''_QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD = 1024

# pixion-verified-ple-reuse-v1
_PIXION_PLE_MARKER_SCHEMA = 1
_PIXION_PLE_SAMPLE_BYTES = 1 << 20


def _pixion_ple_sample(path: pathlib.Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for offset in sorted({0, max(0, size // 2), max(0, size - _PIXION_PLE_SAMPLE_BYTES)}):
            handle.seek(offset)
            digest.update(str(offset).encode())
            digest.update(handle.read(min(_PIXION_PLE_SAMPLE_BYTES, size - offset)))
    return digest.hexdigest()


def _pixion_ple_marker(path: pathlib.Path) -> pathlib.Path:
    return path.with_name(f"{path.name}.pixion-complete.json")


def _pixion_ple_reusable(path_value) -> bool:
    fingerprint = os.environ.get("SGLANG_QWEN4_PLE_REUSE_FINGERPRINT", "")
    if not fingerprint or not path_value:
        return False
    path = pathlib.Path(path_value)
    try:
        payload = json.loads(_pixion_ple_marker(path).read_text())
        valid = (
            payload.get("schema") == _PIXION_PLE_MARKER_SCHEMA
            and payload.get("model_fingerprint") == fingerprint
            and payload.get("table_size") == path.stat().st_size
            and payload.get("sample_sha256") == _pixion_ple_sample(path)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        valid = False
    if valid:
        logger.info("PLE table: verified persistent table, shard copies will be skipped: %s", path)
    return valid


def _pixion_mark_ple_complete(path_value) -> None:
    fingerprint = os.environ.get("SGLANG_QWEN4_PLE_REUSE_FINGERPRINT", "")
    if not fingerprint or not path_value:
        return
    path = pathlib.Path(path_value)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    marker = _pixion_ple_marker(path)
    payload = {
        "schema": _PIXION_PLE_MARKER_SCHEMA,
        "model_fingerprint": fingerprint,
        "table_size": path.stat().st_size,
        "sample_sha256": _pixion_ple_sample(path),
    }
    temporary = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, marker)
    directory_fd = os.open(marker.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    logger.info("PLE table: persistent completion marker committed: %s", marker)


'''
    if source.count(helpers_anchor) != 1:
        raise SystemExit("[pixion] helper anchor count != 1; aborting")
    source = source.replace(helpers_anchor, helpers)

    parameter_old = '''        cpu_weight = nn.Parameter(host_table, requires_grad=False)
        for name, value in vars(source_weight).items():
'''
    parameter_new = '''        cpu_weight = nn.Parameter(host_table, requires_grad=False)
        # Preserve mmap identity across the Tensor -> Parameter wrapper.
        cpu_weight._pixion_ple_file_path = getattr(
            host_table, "_sglang_ple_file_path", None
        )
        for name, value in vars(source_weight).items():
'''
    if source.count(parameter_old) != 1:
        raise SystemExit("[pixion] host parameter anchor count != 1; aborting")
    source = source.replace(parameter_old, parameter_new)

    lookup_old = '''            emb = ple_mod.ngram_embedding
            if (
'''
    lookup_new = '''            emb = ple_mod.ngram_embedding
            ple_seen_shards.add((mod_prefix, shard_idx))
            if mod_prefix in ple_reuse_modules:
                loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
                return True
            if (
'''
    if source.count(lookup_old) != 1:
        raise SystemExit("[pixion] shard lookup anchor count != 1; aborting")
    source = source.replace(lookup_old, lookup_new)

    modules_old = '''        ple_modules = {
            mod_name: mod
            for mod_name, mod in self.named_modules()
            if isinstance(mod, Qwen4ExpNGramEmbedding)
        }
        text_config = getattr(self.config, "text_config", self.config)
'''
    modules_new = '''        ple_modules = {
            mod_name: mod
            for mod_name, mod in self.named_modules()
            if isinstance(mod, Qwen4ExpNGramEmbedding)
        }
        ple_reuse_modules = {
            mod_name
            for mod_name, mod in ple_modules.items()
            if _pixion_ple_reusable(
                getattr(mod.ngram_embedding.weight, "_pixion_ple_file_path", None)
            )
        }
        ple_seen_shards: Set[Tuple[str, int]] = set()
        text_config = getattr(self.config, "text_config", self.config)
'''
    if source.count(modules_old) != 1:
        raise SystemExit("[pixion] PLE module anchor count != 1; aborting")
    source = source.replace(modules_old, modules_new)

    finish_old = '''        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)

        if skipped_visual_count > 0:
'''
    finish_new = '''        file_ple_modules = {
            name: mod for name, mod in ple_modules.items()
            if getattr(mod.ngram_embedding.weight, "_pixion_ple_file_path", None)
        }
        expected_ple_shards = {
            (name, shard_idx)
            for name in file_ple_modules
            for shard_idx in range(ple_num_sync_shards)
        }
        if file_ple_modules and ple_seen_shards != expected_ple_shards:
            missing = len(expected_ple_shards - ple_seen_shards)
            extra = len(ple_seen_shards - expected_ple_shards)
            raise RuntimeError(
                f"PLE persistent-table verification failed: missing={missing}, extra={extra}"
            )
        marked_paths = set()
        for name, mod in file_ple_modules.items():
            if name in ple_reuse_modules:
                continue
            table_path = getattr(
                mod.ngram_embedding.weight, "_pixion_ple_file_path", None
            )
            if table_path and table_path not in marked_paths:
                _pixion_mark_ple_complete(table_path)
                marked_paths.add(table_path)

        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)

        if skipped_visual_count > 0:
'''
    if source.count(finish_old) != 1:
        raise SystemExit("[pixion] load completion anchor count != 1; aborting")
    source = source.replace(finish_old, finish_new)

    compile(source, str(path), "exec")
    path.write_text(source)
    print(f"[pixion] enabled verified persistent SGLang PLE reuse: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
