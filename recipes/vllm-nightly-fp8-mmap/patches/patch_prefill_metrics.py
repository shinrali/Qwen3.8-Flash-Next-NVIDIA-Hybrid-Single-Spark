#!/usr/bin/env python3
"""Add live per-step prefill counters to the pinned vLLM image.

The stock ``vllm:prompt_tokens_total`` counter is updated only after a request's
prefill completes.  With ``--enable-logging-iteration-details`` vLLM already
attaches ``SchedulerIterationDetails`` to each engine step; this patch exports
that data as Prometheus counters and suppresses the otherwise noisy per-step
INFO log line.
"""

SP = "/usr/local/lib/python3.12/dist-packages"
F = f"{SP}/vllm/v1/metrics/loggers.py"

src = open(F).read()
assert "class PrometheusStatLogger" in src and "def _log_iteration_details" in src
assert "_pixion_prefill_metrics" not in src, "already patched"

src += '''

# --- PixionFilm: live prefill metrics (--enable-logging-iteration-details) ---
def _pixion_prefill_metrics():
    _orig_init = PrometheusStatLogger.__init__
    _orig_record = PrometheusStatLogger.record

    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        names = ["model_name", "engine"]
        ctx = self._counter_cls(
            name="vllm:scheduled_ctx_tokens",
            documentation="Prefill context tokens scheduled per engine step; "
                          "rate() is live prefill tok/s. Requires "
                          "--enable-logging-iteration-details.",
            labelnames=names)
        iterations = self._counter_cls(
            name="vllm:scheduled_iterations",
            documentation="Engine steps observed via iteration details.",
            labelnames=names)
        self._pixion_ctx = {
            i: ctx.labels(*v) for i, v in self.per_engine_labelvalues.items()
        }
        self._pixion_iterations = {
            i: iterations.labels(*v)
            for i, v in self.per_engine_labelvalues.items()
        }

    def _record_details(self, scheduler_stats, engine_idx=0):
        details = (
            getattr(scheduler_stats, "iteration_details", None)
            if scheduler_stats else None
        )
        if (
            details is not None
            and not details.is_dummy
            and engine_idx in self._pixion_ctx
        ):
            self._pixion_ctx[engine_idx].inc(details.num_ctx_tokens)
            self._pixion_iterations[engine_idx].inc()

    def record(
        self,
        scheduler_stats,
        iteration_stats,
        mm_cache_stats=None,
        engine_idx=0,
    ):
        _orig_record(
            self,
            scheduler_stats,
            iteration_stats,
            mm_cache_stats,
            engine_idx,
        )
        self._pixion_record_details(scheduler_stats, engine_idx)

    PrometheusStatLogger.__init__ = __init__
    PrometheusStatLogger._pixion_record_details = _record_details
    PrometheusStatLogger.record = record
    LoggingStatLogger._log_iteration_details = (
        lambda self, scheduler_stats, engine_idx: None
    )


_pixion_prefill_metrics()
'''

open(F, "w").write(src)

import ast

ast.parse(src)
print("loggers.py: PixionFilm live prefill metrics added OK")
