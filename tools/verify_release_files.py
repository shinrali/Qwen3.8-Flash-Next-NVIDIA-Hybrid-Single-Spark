#!/usr/bin/env python3
"""Fail when a published Dockerfile or runtime manifest omits a local file."""

from __future__ import annotations

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


def main() -> int:
    missing = {path for path in REQUIRED if not (ROOT / path).is_file()}
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
    print(f"release file audit passed: {len(REQUIRED)} required patches, {len(local_sources)} Docker sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
