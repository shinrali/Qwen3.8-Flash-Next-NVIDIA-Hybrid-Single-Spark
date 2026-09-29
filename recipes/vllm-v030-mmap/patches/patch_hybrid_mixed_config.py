#!/usr/bin/env python3
"""Retarget the proven FP8-side shim to ModelOptMixedPrecisionConfig."""

from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()
anchor = "    cfg_cls = m.ModelOptNvFp4Config\n"
replacement = "    cfg_cls = m.ModelOptMixedPrecisionConfig\n"
if source.count(anchor) != 1:
    raise SystemExit("hybrid ModelOpt config-class anchor not found exactly once")
source = source.replace(anchor, replacement)

log_anchor = '    logger.info("fp8 hybrid patch applied to ModelOptNvFp4Config")\n'
log_replacement = (
    '    logger.info("fp8 hybrid patch applied to '
    'ModelOptMixedPrecisionConfig")\n'
)
if source.count(log_anchor) != 1:
    raise SystemExit("hybrid ModelOpt log-message anchor not found exactly once")
path.write_text(source.replace(log_anchor, log_replacement))
print("Hybrid FP8-side shim retargeted to ModelOptMixedPrecisionConfig")
