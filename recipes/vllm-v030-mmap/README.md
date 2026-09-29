# DGX Spark: vLLM 0.30 with file-backed FP8 PLE

This profile serves the recommended FP8-`lm_head` checkpoint on one 128 GB DGX
Spark using the official `vllm/vllm-openai:v0.30.0` image. The 47.68 GiB FP8
PLE table remains file-backed and is accessed over ATS through MiaAI Lab's mmap
patch instead of being pinned in unified memory.

The checked-in patch directory is complete. The container applies the patches
at startup, so this recipe does not create another local image.

## Validated boundary

- one GB10 DGX Spark with 128 GB unified memory;
- native 262,144-token context, no YaRN;
- two sequences;
- explicit 16 GiB BF16 KV cache;
- 554,044 profiled KV tokens, or 2.11 full-context slots;
- MTP 3 with the public matched 65K FP8 draft vocabulary/head;
- prefix caching and chunked prefill enabled;
- no `--enforce-eager`.

The validation checkpoint used the same tensor layout as the public recommended
checkpoint. A personally optimized 65K pair was used for the measurement; the
public pair has the same shape and kernel path, but acceptance depends on the
output distribution and should be remeasured.

## Start

```bash
cp .env.example .env
# Edit MODEL_DIR, CACHE_DIR and VLLM_API_KEY.
docker compose -f compose.example.yaml config
docker compose -f compose.example.yaml up -d
docker compose -f compose.example.yaml logs -f
```

The model directory must contain the checkpoint plus the matched public pair:

```text
draft_vocab_65536.npy
mtp_draft_head_65536_fp8.safetensors
```

Do not mix a vocabulary array with a head generated from a different token-ID
order. Keep `CACHE_DIR` persistent: the first launch creates the PLE mmap file
and its fingerprint sidecar; subsequent launches reuse them when the checkpoint
identity, shape and dtype still match.

## Measured result

On 2026-09-29 AEST, the service reached health with zero restarts and no OOM.
The model loader reported 72.98 GiB and 558.13 seconds; API health arrived after
roughly 10 minutes 46 seconds on the cold checkpoint load. The existing 47.68
GiB PLE mmap file was reused rather than rebuilt.

The four-stage long-context test incrementally updated one target among similar
key/value records and forced an eight-field tool call after every stage:

| Stage | Prompt tokens | TTFT | Effective input | Decode | Checks |
| ---: | ---: | ---: | ---: | ---: | ---: |
| R11 | 116,392 | 42.86 s | 2,715.46 tok/s | 62.50 tok/s | 8/8 |
| R12 | 137,352 | 8.86 s | 15,500.26 tok/s | 68.18 tok/s | 8/8 |
| R13 | 201,341 | 24.79 s | 8,122.53 tok/s | 66.57 tok/s | 8/8 |
| R14 | 258,310 | 23.20 s | 11,132.82 tok/s | 68.13 tok/s | 8/8 |
| **Mean** | — | **24.93 s** | **9,367.77 tok/s** | **66.35 tok/s** | **32/32** |

The large effective-input values after R11 include prefix-cache reuse and are
not pure prefill kernel throughput. R11 is the useful cold-prefill reference.
For this tool-call workload the cumulative vLLM counters reported 429 accepted
of 434 drafted speculative tokens (98.85%, about 3.88 accepted positions out of
four). This acceptance rate is workload-specific, not a general model claim.

The same four-stage test on the published SGLang v0.5.20 profile also scored
32/32, averaging 2,288.21 effective input tok/s and 63.39 decode tok/s, but did
not obtain equivalent prefix-cache reuse in that run. This demonstrates that
the complete vLLM 0.30 mmap profile worked; it does not isolate one engine or
one patch as the sole cause of the difference.

## Attribution and limits

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the exact MiaAI mmap
source and license. The file-backed table avoids permanent PLE residency but
can still consume Linux page cache and generate NVMe traffic. Cold startup,
long-prefill latency and decode performance depend on cache warmth, SSD, memory
pressure and request distribution.
