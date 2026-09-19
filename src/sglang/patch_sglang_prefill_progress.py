#!/usr/bin/env python3
"""Pixion patch: export SGLang's live pending-prefill token gauge.

SGLang v0.5.20 already exports cumulative compute and cache-hit counters after
every chunked-prefill batch, but it does not export the remaining-token count
that its scheduler logs as ``#pending-token``.  Without that denominator a
client can measure prefill speed and cache hits, but cannot estimate progress.

This patch adds ``sglang:prefill_pending_tokens`` and updates it alongside the
existing per-prefill-batch counters.  It is engine-scoped, like the existing
realtime counters, and is intentionally an approximate aggregate when several
requests prefill concurrently.
"""

import glob
import importlib.util
import pathlib
import sys
import sysconfig


MARKER = "pixion-prefill-pending-tokens-v1"


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
    if len(sys.argv) == 3:
        collector_path = pathlib.Path(sys.argv[1])
        reporter_path = pathlib.Path(sys.argv[2])
    elif len(sys.argv) == 1:
        collector_path = find_file("srt/observability/metrics_collector.py")
        reporter_path = find_file("srt/managers/scheduler_components/metrics_reporter.py")
    else:
        raise SystemExit(f"usage: {sys.argv[0]} [metrics_collector.py metrics_reporter.py]")

    collector = collector_path.read_text()
    reporter = reporter_path.read_text()
    if MARKER in collector and MARKER in reporter:
        print(f"[pixion] already patched: {collector_path}, {reporter_path}")
        return 0
    if MARKER in collector or MARKER in reporter:
        raise SystemExit("[pixion] partial prefill-progress patch detected; aborting")

    collector_anchor = '''        for mode in ("input", "device_hit", "host_hit", "storage_hit"):
            self.prefill_effective_tokens_total.labels(**labels, mode=mode)
'''
    collector_replacement = collector_anchor + '''        # pixion-prefill-pending-tokens-v1: denominator for live prefill progress.
        self.prefill_pending_tokens = Gauge(
            name="sglang:prefill_pending_tokens",
            documentation=(
                "Current engine-wide tokens still pending chunked prefill. "
                "Combine with deltas of prefill_effective_tokens_total."
            ),
            labelnames=labels.keys(),
        )
'''
    if collector.count(collector_anchor) != 1:
        raise SystemExit("[pixion] collector gauge anchor count != 1; aborting")
    collector = collector.replace(collector_anchor, collector_replacement)

    method_anchor = '''    def increment_forward_execution_seconds(
'''
    method_replacement = '''    def set_prefill_pending_tokens(self, value: int) -> None:
        self.prefill_pending_tokens.labels(**self.labels).set(max(0, value))

''' + method_anchor
    if collector.count(method_anchor) != 1:
        raise SystemExit("[pixion] collector method anchor count != 1; aborting")
    collector = collector.replace(method_anchor, method_replacement)

    reporter_anchor = '''            self.metrics_collector.increment_effective_prefill_tokens(
                input_tokens=effective_input_tokens,
                device_hit_tokens=prefill_stats.log_device_hit_tokens,
                host_hit_tokens=prefill_stats.log_host_hit_tokens,
                storage_hit_tokens=prefill_stats.log_storage_hit_tokens,
            )
'''
    reporter_replacement = reporter_anchor + '''            # pixion-prefill-pending-tokens-v1: updated with the same chunk.
            self.metrics_collector.set_prefill_pending_tokens(
                prefill_stats.num_pending_tokens
            )
'''
    if reporter.count(reporter_anchor) != 1:
        raise SystemExit("[pixion] reporter update anchor count != 1; aborting")
    reporter = reporter.replace(reporter_anchor, reporter_replacement)

    compile(collector, str(collector_path), "exec")
    compile(reporter, str(reporter_path), "exec")
    collector_path.write_text(collector)
    reporter_path.write_text(reporter)
    print(f"[pixion] exported live SGLang prefill progress: {collector_path}, {reporter_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
