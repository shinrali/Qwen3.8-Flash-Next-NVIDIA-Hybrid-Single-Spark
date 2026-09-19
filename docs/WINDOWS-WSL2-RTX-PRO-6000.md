# Windows/WSL2 vLLM on RTX PRO 6000 Blackwell

## Validated result

The same NVIDIA-main hybrid checkpoint, FP8 target `lm_head`, NVFP4 MTP and
personally optimized 65K FP8 draft pair run under Windows Docker Desktop/WSL2
on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition.

The validated boundary is:

- patched vLLM `v0.1.dev20073+g8e685d198`;
- `max_model_len=262144`;
- `max_num_seqs=2`;
- explicit 16 GiB BF16 KV cache;
- two concurrent 256K-class sequence slots;
- PLE FP8 table served through mmap/SSD;
- CUDA graphs enabled in `PIECEWISE` mode;
- no `--enforce-eager`.

The workstation has now completed a reproducible short-context throughput run.
This replaces the earlier note that no reliable RTX PRO 6000 tok/s record was
available. It does not replace the separate two-slot functional validation.

## Measured 4K single-stream throughput

Measurement date: 2026-09-19 AEST. The OpenAI-compatible speed harness used one
warm-up followed by three formal runs, `temperature=0`, a nominal 4,096-token
input and a 512-token output limit. Each formal request reached the output limit.

| Run | Prompt tokens | TTFT | Effective prefill | Decode | MTP acceptance | End-to-end |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4,069 | 0.400 s | 10,171.26 tok/s | 92.96 tok/s | 48.95% | 5.897 s |
| 2 | 4,095 | 0.407 s | 10,056.50 tok/s | 86.62 tok/s | 41.85% | 6.307 s |
| 3 | 4,100 | 0.393 s | 10,441.78 tok/s | 95.38 tok/s | 52.17% | 5.750 s |
| **Mean** | **4,088** | **0.400 s** | **10,223.18 tok/s** | **91.66 tok/s** | **47.66%** | **5.985 s** |

`Effective prefill` is `prompt_tokens / TTFT`, so it includes request transport,
tokenization, scheduling and time to first streamed token rather than claiming a
pure GPU-kernel rate. The harness places a fresh UUID nonce near the beginning of
every generated prompt. Prefix identity therefore breaks before the large test
corpus, preventing the warm-up and earlier runs from turning this into a cached
4K-prefill result. A small fixed header before the nonce may still be reusable.

This is a single-stream 4K result. It does not measure 65K/256K prefill,
two-stream aggregate throughput, latency under concurrent prefill and decode, or
performance with the KV pool near capacity. Those remain separate follow-up
profiles. The public result intentionally omits the private test endpoint and
credentials.

## Required files

Use the complete public patch set documented in `PATCH-MANIFEST.md`. In
particular, `Dockerfile.nvidia-nvfp4mtp` requires `src/patch_mtp_fp8.py`; this
file is now included in the repository.

Set the matched personal pair together:

```text
VLLM_MTP_DRAFT_VOCAB=/draft/studio_draft_vocab_65536_all.npy
VLLM_MTP_DRAFT_HEAD=/draft/mtp_draft_head_studio_65536_all_fp8.safetensors
```

The filenames are historical compatibility names. The pair is a personal
optimization and has no application dependency.

## Platform difference: QSADET

The optional deterministic QSA extension `_C_det.so` built for GB10 is not a
portable CUDA binary. On the RTX PRO 6000 it failed with:

```text
cudaFuncGetAttributes failed: no kernel image is available for execution on the device
```

Do not set `VLLM_QSA_DET_TOPK` or mount the Spark-built `_C_det.so` on the
workstation. Rebuild that extension for the workstation architecture before
enabling it. The standard QSA fallback is functional.

## Working runtime profile

`recipes/windows-wsl2/compose.example.yaml` mirrors the validated settings while
keeping host paths, GPU selection and API key external. The important runtime
arguments are:

```text
--max-model-len 262144
--max-num-seqs 2
--kv-cache-memory-bytes 16g
--enable-prefix-caching
--enable-prompt-tokens-details
--enable-chunked-prefill
--max-num-batched-tokens 8192
-cc.cudagraph_mode=PIECEWISE
--speculative-config {"method":"mtp","num_speculative_tokens":3,"max_model_len":262144}
```

The current vLLM build reports that fused multi-step draft decode is not
supported by `QWEN38_FLASH_NEXT_EXP_QSA_STATE`; it rebuilds attention metadata
between draft steps. Together with PLE mmap scheduling, this is a likely source
of the remaining utilization gap. Useful follow-up A/Bs are MTP off, MTP 1 and
MTP 3 with identical prompts, output lengths, PLE and KV settings, plus 65K and
256K single-stream and two-stream profiles.
