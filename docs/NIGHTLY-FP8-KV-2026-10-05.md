# Single DGX Spark: pinned vLLM nightly + FP8 KV

Validated on 2026-10-05 (Australia/Melbourne). This is an independent community
integration, not an official NVIDIA or Qwen performance claim. Model weights
were not changed for this runtime update.

## Runtime and memory

- Official image, no custom image build:
  `vllm/vllm-openai:nightly-0cbac6cd1305f710e12193596b27488397bcb205@sha256:8e6e3752946ff2dde451b313cbaec2ea8e7b52221b21bfbcf53fb68dd4f4c42e`.
- Runtime version: `0.30.1rc1.dev558+g0cbac6cd1`.
- One GB10 DGX Spark, 128 GB unified memory. NVIDIA NVFP4 main experts,
  FP8 side layers and target head, NVFP4 MTP experts, matched personal 65K FP8
  draft head, file-backed 47.68 GiB FP8 PLE through MiaAI-derived mmap.
- Main KV: `fp8_e4m3`, explicit **9 GiB** (`9663676416` bytes).
  Recurrent/Mamba state remains BF16; FP8 KV does not mean all state is FP8.
- Native context **262,144**, no YaRN; admission limit **4 sequences**.
  Profiled pool: **542,103 tokens**, or **2.07 full-length slots**. Four
  admitted sequences share this pool; this is not four simultaneous full 256K
  contexts, nor a two-full-context pressure-test result.
- MTP 3, prefix caching, chunked prefill (8192), adaptive long-prefill
  threshold 2048, FULL_DECODE_ONLY CUDA Graphs. No `--enforce-eager`.
- Latest startup reported 75.33 GiB model loading allocation. This is a loader
  measurement, not whole-system RAM accounting.

Use the complete [mounted-patch recipe](../recipes/vllm-nightly-fp8-mmap/README.md).
The image already contains upstream QSA FP8-KV support and the QSA indexer
workspace fix; the recipe does **not** apply the old QSA FP8 backport or a
legacy deterministic QSA binary. Compatibility patches remain necessary for
the mixed-precision heads and SSD PLE layout.

## 8K speed benchmark

Source: user-exported speed test, `2026-10-05T01:52:32Z` (12:52:32 AEDT).
Single stream, target 8192 input tokens, 512 output limit, temperature 0,
three warmups followed by three measured requests. All outputs reached the
512-token cap (`finish_reason=length`); this is throughput, not answer quality.
No private prompts, responses, API keys or production endpoints are published.

| Run | Input tokens | Output tokens | TTFT | Effective input tok/s | Decode tok/s | MTP acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8191 | 512 | 2.766 s | 2960.93 | 42.17 | 50.65% |
| 2 | 8190 | 512 | 2.738 s | 2991.20 | 42.38 | 49.35% |
| 3 | 8190 | 512 | 2.755 s | 2972.70 | 43.43 | 51.32% |
| **Mean** | **8190.33** | **512** | **2.753 s** | **2974.94** | **42.66** | **50.44%** |

Mean request end-to-end time: **14.733 s**. Batch-counter aggregate decode:
42.657 tok/s. MTP comes from vLLM `/metrics`. Effective input is prompt tokens
divided by TTFT, not isolated GPU prefill; the export does not provide cached
token counts or thinking settings, so neither cache-free prefill nor thinking
mode is inferred. Server default enables thinking, but client overrides may
change it. There is no strict same-runtime BF16/FP8 KV A/B here.

The [sanitized machine-readable export](../benchmarks/nightly-fp8-kv-speed-2026-10-05.json)
retains numerical measurements and removes endpoint and local identifiers.

## Bounded long-context regression

A separate synthetic `creative-v1` export on 2026-10-03 AEST tested character
relationships, fixed creative constraints, parallel timing arithmetic and
successive approved revisions. Thinking off, temperature 0:

| Round | Input tokens | Checks | TTFT | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| R1 | 59,290 | 12/12 | 2.513 s | 55.46 |
| R2 | 132,375 | 12/12 | 1.830 s | 61.15 |
| R3 | 193,199 | 12/12 | 2.059 s | 63.48 |
| R4 | 254,140 | 12/12 | 2.171 s | 63.11 |

Total **48/48**, mean decode **60.80 tok/s**. Short, repetitive JSON output
had 91.56% speculative acceptance; this is not a general prose/coding speed.
Very short TTFTs at these lengths are consistent with prefix reuse; the export
does not expose per-round cached counts. This bounded success does not establish
long-context equivalence or absence of FP8 errors. Follow-up UI results at
server-default temperature 1 scored 47/48 with thinking off and 48/48 with
thinking low; these single runs are not a statistically controlled comparison.
The historical 162-question scores elsewhere in this repository were measured
on older runtimes, not rerun on this nightly.

## Patch inventory

The startup script calls the following files, all included in the recipe:

1. `patch_qwen4_layer_type_nightly.py`: Transformers layer-type alias compatibility.
2. `patch_ple_mmap_nightly.py`, using `patch_ple_mmap_v030.py`: shared n-gram
   storage and NVIDIA loader adaptation for file-backed PLE.
3. `patch_qwen4_exp_fp8_heads.py`, `patch_vllm_v030_fp8_lm_head_scale.py`:
   quantized target/reduced draft head and companion-scale loading.
4. `vllm_fp8_hybrid_modelopt.py`, `patch_hybrid_mixed_config.py`,
   `patch_modelopt_hybrid_import.py`: mixed-precision dispatch.
5. `patch_safetensors_index_filter.py`: indexed tensor filtering.
6. `patch_prefill_metrics.py`: scheduled context/prefill counters.

See the recipe's third-party notices for MiaAI attribution and license.
Only the pinned image is validated; retest anchors, startup, tools, cache,
accuracy and performance before changing the nightly tag.
