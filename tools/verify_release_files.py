#!/usr/bin/env python3
"""Fail when a published Dockerfile or runtime manifest omits a local file."""

from __future__ import annotations

import hashlib
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED = {
    "src/patch_mtp_fp8.py",
    "src/patch_fp8_lm_head.py",
    "src/patch_mtp_draft_vocab.py",
    "src/sglang/patch_sglang_block_fp8_lmhead.py",
    "src/sglang/patch_sglang_draft65k.py",
    "src/sglang/patch_sglang_ple_reuse.py",
    "src/sglang/patch_sglang_prefill_progress.py",
    "src/sglang/patch_sglang_redact_secrets.py",
    "src/sglang/sglang_ple_reuse_helper.py",
}
V030_MMAP_PATCHES = {
    "recipes/vllm-v030-mmap/patches/apply-and-serve.sh",
    "recipes/vllm-v030-mmap/patches/patch_hybrid_mixed_config.py",
    "recipes/vllm-v030-mmap/patches/patch_modelopt_hybrid_import.py",
    "recipes/vllm-v030-mmap/patches/patch_ple_mmap_v030.py",
    "recipes/vllm-v030-mmap/patches/patch_prefill_metrics.py",
    "recipes/vllm-v030-mmap/patches/patch_qwen4_exp_fp8_heads.py",
    "recipes/vllm-v030-mmap/patches/patch_safetensors_index_filter.py",
    "recipes/vllm-v030-mmap/patches/patch_vllm_v030_fp8_lm_head_scale.py",
    "recipes/vllm-v030-mmap/patches/vllm_fp8_hybrid_modelopt.py",
}
MIAAI_PLE_SHA256 = "b63bb3d86ececfe5d0dac7f4c072407c9745fab7b6ed856175bc8f668c107309"


def main() -> int:
    required = REQUIRED | V030_MMAP_PATCHES
    missing = {path for path in required if not (ROOT / path).is_file()}
    local_sources: set[str] = set()
    for dockerfile in ROOT.glob("Dockerfile*"):
        for line in dockerfile.read_text().splitlines():
            match = re.match(r"\s*(?:COPY|ADD)\s+([^\s]+)", line)
            if not match:
                continue
            source = match.group(1)
            if source.startswith(("http://", "https://", "--")) or "$" in source:
                continue
            local_sources.add(source)
            if not (ROOT / source).exists():
                missing.add(f"{dockerfile.name}: {source}")
    if missing:
        raise SystemExit("missing release files:\n" + "\n".join(sorted(missing)))
    miaai_patch = ROOT / "recipes/vllm-v030-mmap/patches/patch_ple_mmap_v030.py"
    actual_sha256 = hashlib.sha256(miaai_patch.read_bytes()).hexdigest()
    if actual_sha256 != MIAAI_PLE_SHA256:
        raise SystemExit(
            "MiaAI PLE patch checksum mismatch: "
            f"expected {MIAAI_PLE_SHA256}, got {actual_sha256}"
        )
    print(
        "release file audit passed: "
        f"{len(required)} required patches, {len(local_sources)} Docker sources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
