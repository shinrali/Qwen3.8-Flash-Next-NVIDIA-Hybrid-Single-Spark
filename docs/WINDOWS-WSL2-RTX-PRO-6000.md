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

The supplied workstation notes confirm functional loading and concurrency but
do not preserve a reliable prefill/decode benchmark. Do not reuse the DGX Spark
tok/s figures as RTX PRO 6000 measurements. Observed utilization was about 50%,
so workstation-specific optimization remains open.

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
of the utilization gap. Useful follow-up A/Bs are MTP off, MTP 1 and MTP 3 with
identical prompts, output lengths, PLE and KV settings.
