# Performance evidence

## Latest: official pinned nightly + FP8 KV (2026-10-05)

One Spark, native 262144 context, 9 GiB FP8 E4M3 KV, BF16 recurrent state,
file-backed FP8 PLE and MTP3. Three warmups, three measured 8K/512 requests,
temperature 0, concurrency 1: **2974.94 effective input tok/s, 42.66 decode
tok/s, 2.753 s TTFT, 14.733 s total, 50.44% MTP acceptance**.

See [the complete nightly report](NIGHTLY-FP8-KV-2026-10-05.md) for the fixed
image digest, per-run data, long-context 48/48 regression and limitations.
The export does not establish cache-free prefill or a controlled KV-dtype A/B.

This document keeps the detailed measurements out of the project entrance page.
Results use different prompt lengths, runtimes and hosts unless explicitly
described as paired. Do not treat the tables as one cross-platform leaderboard.

## DGX Spark — vLLM v0.30.0 with MiaAI file-backed PLE

Measurement date: 2026-09-29 AEST. Hardware: one GB10 DGX Spark with 128 GB
unified memory. Runtime: official `vllm/vllm-openai:v0.30.0`, this project's
FP8-side/FP8-`lm_head`/NVFP4-MTP checkpoint, a personally optimized matched 65K
FP8 draft pair, BF16 KV/recurrent state, MTP 3, native 262,144 context and
prefix caching. MiaAI Lab's mmap patch kept the 47.68 GiB FP8 PLE table
file-backed and read it over ATS.

The explicit 16 GiB KV cache profiled 554,044 tokens, or 2.11 full-length
262,144-token slots. Container health was green with zero restarts and no OOM.
Model loading reported 72.98 GiB and 558.13 seconds; the existing PLE mmap file
was reused.

| Stage | Prompt tokens | TTFT | Effective input | Decode | Checks |
| ---: | ---: | ---: | ---: | ---: | ---: |
| R11 | 116,392 | 42.86 s | 2715.46 tok/s | 62.50 tok/s | 8/8 |
| R12 | 137,352 | 8.86 s | 15500.26 tok/s | 68.18 tok/s | 8/8 |
| R13 | 201,341 | 24.79 s | 8122.53 tok/s | 66.57 tok/s | 8/8 |
| R14 | 258,310 | 23.20 s | 11132.82 tok/s | 68.13 tok/s | 8/8 |
| **Mean** | — | **24.93 s** | **9367.77 tok/s** | **66.35 tok/s** | **32/32** |

Each stage appended a newer eight-field target state among similar neighboring
records, then forced one structured tool call. Effective input is
`prompt_tokens / TTFT`; after R11 it includes prefix-cache reuse and is not a
pure prefill-kernel rate. Cumulative speculative counters for this workload
reported 429 accepted of 434 drafted tokens (98.85%, about 3.88 positions out
of four). The machine-readable result is
[`benchmarks/vllm030-mmap-longkv-2026-09-29.json`](../benchmarks/vllm030-mmap-longkv-2026-09-29.json).

The identical four-stage test on the SGLang v0.5.20 profile scored 32/32 with
2,288.21 mean effective input tok/s and 63.39 mean decode tok/s, but that run did
not receive equivalent prefix-cache reuse. This is an end-to-end profile
comparison, not an engine-only or PLE-only A/B.

## DGX Spark — patched vLLM

Hardware: one GB10 DGX Spark, 128 GB unified memory. Runtime: the pinned vLLM
preview, MTP 3, personal 65K reduced draft vocabulary, deterministic QSA top-k,
BF16 KV/recurrent state, a 500,000-token YaRN profile and prefix caching.

| Profile | 8K prefill | 8K decode | 100K prefill | 100K decode |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA BF16-side + NVFP4 MTP | 2253.28 | 20.36 | 2177.88 | 21.21 |
| FP8-side + BF16 `lm_head` + NVFP4 MTP | 2303.64 | 24.77 | 2129.24 | 25.92 |
| FP8-side + FP8 `lm_head` + NVFP4 MTP | 2233.42 | 27.34 | 2141.79 | 28.11 |
| RadixArk FP8-side + NVFP4 MTP | 2341.93 | 26.05 | 2154.00 | 26.86 |

Throughput units are tokens/s. The BF16-target-head NVIDIA hybrid's three 100K
runs averaged 66.88 seconds end-to-end with 33.73% weighted MTP acceptance. It
loaded 73.89 GiB of weights and profiled a 19.73 GiB / 721,556-token KV pool.

The FP8-target-head profile's three 100K runs measured
2140.69/28.46, 2143.12/26.62 and 2141.57/29.25 prefill/decode tok/s. Weighted
MTP acceptance was 32.77% and mean end-to-end time was 65.08 seconds. Its first
formal 8K run still included runtime JIT work, so interpret the 8K mean
conservatively.

## Acceptance-scaled decode estimate

MTP 3 verifies four output positions per engine step: one target token plus up
to three accepted draft tokens. The measured 100K rates imply mean accepted
lengths of `1 + 3 x 0.3373 = 2.012` for the BF16 target head and
`1 + 3 x 0.3277 = 1.983` for the FP8 target head. Dividing measured decode by
those lengths gives approximately 12.88 and 14.17 engine steps/s.

| Mean accepted length (max 4) | Implied draft acceptance | BF16 `lm_head` estimate | FP8 `lm_head` estimate |
| ---: | ---: | ---: | ---: |
| 2.0 | 33.3% | 25.77 tok/s | 28.35 tok/s |
| 2.5 | 50.0% | 32.21 tok/s | 35.44 tok/s |
| 3.0 | 66.7% | 38.65 tok/s | 42.52 tok/s |
| 3.5 | 83.3% | 45.09 tok/s | 49.61 tok/s |
| 3.71 | 90.3% | 47.80 tok/s | 52.59 tok/s |
| 4.0 | 100% | 51.53 tok/s | 56.70 tok/s |

These are linear estimates anchored to the measured 100K runs, not a measured
coding benchmark. PLE locality, context, concurrency, output distribution and
warm-up can change engine-step cost.

## Reduced draft-head A/B on DGX Spark

Both lanes used the same complete FP8 target head, 65,536 token IDs, main
checkpoint, MTP, PLE, BF16 KV and serving arguments. Only the reduced draft
head format changed. Hot runs generated 4,916 tokens with `temperature=0`,
`seed=0` and a 16,384-token output limit.

| Reduced draft head | Hot decode runs | Mean | MTP acceptance |
| --- | --- | ---: | ---: |
| BF16, about 320 MiB | 41.36 / 41.18 tok/s | 41.27 tok/s | 48.57% |
| FP8, about 160 MiB | 43.70 / 40.66 / 43.92 / 43.95 tok/s | 43.05 tok/s | 48.41% |

The all-run FP8 mean improved 4.32%; its median was 43.81 tok/s, 6.15% above
the BF16 mean. One FP8 run was a low outlier, so describe this as an observed
4–6% benefit with runtime variance rather than a guaranteed fixed gain.

## DGX Spark — SGLang v0.5.20

| Draft configuration | Prefill tok/s | Decode tok/s | MTP acceptance |
| --- | ---: | ---: | ---: |
| Complete vocabulary | 2604.22 | 37.31 | 46.94% |
| Personal 65K, first run | 2604.29 | 37.47 | 48.61% |
| Personal 65K, hot repeat | 2663.39 | 42.53 | 56.11% |

The hot repeat includes runtime warm-up and output-dependent acceptance. It is
a bounded sample, not a universal 14% speed guarantee. The engine profiled
18.85 GiB of BF16 KV for 688,622 tokens in this run; host memory state affects
automatic KV sizing.

## RTX PRO 6000 Blackwell — Windows/WSL2 vLLM

Measurement date: 2026-09-19 AEST. The workstation used the FP8 target head,
personal 65K FP8 draft pair, 16 GiB explicit BF16 KV, `max_model_len=262144`,
`max_num_seqs=2`, prefix caching, chunked prefill and PIECEWISE CUDA graphs.
The speed test itself was single-stream.

The OpenAI-compatible harness used one warm-up followed by three formal runs,
`temperature=0`, a nominal 4,096-token input and 512-token output limit. Every
formal request reached the output limit.

| Run | Prompt tokens | TTFT | Effective prefill | Decode | MTP acceptance | End-to-end |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4,069 | 0.400 s | 10,171.26 tok/s | 92.96 tok/s | 48.95% | 5.897 s |
| 2 | 4,095 | 0.407 s | 10,056.50 tok/s | 86.62 tok/s | 41.85% | 6.307 s |
| 3 | 4,100 | 0.393 s | 10,441.78 tok/s | 95.38 tok/s | 52.17% | 5.750 s |
| **Mean** | **4,088** | **0.400 s** | **10,223.18 tok/s** | **91.66 tok/s** | **47.66%** | **5.985 s** |

`Effective prefill` is `prompt_tokens / TTFT`; it includes transport,
tokenization, scheduling and time to first streamed token rather than claiming
a pure GPU-kernel rate. The harness inserts a fresh UUID nonce near the start
of every prompt. Prefix identity therefore breaks before the large test corpus,
preventing the warm-up and prior runs from turning this into a cached 4K result.
A small fixed header before the nonce may still be reusable.

This test does not measure 65K/256K prefill, concurrent aggregate throughput,
decode latency during a concurrent prefill, or operation near the KV-capacity
limit. The private endpoint and credentials are intentionally omitted.
