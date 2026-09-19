# Runtime patch manifest

This manifest exists to prevent a documented Docker build from referring to an
unpublished patch. All paths below are tracked by Git.

## vLLM checkpoint/runtime path

- `src/vllm_ple_mmap.py`
- `src/vllm_fp8_hybrid_modelopt.py`
- `src/patch_mtp_fp8.py`
- `src/patch_hybrid_mixed_config.py`
- `src/patch_fp8_lm_head.py`
- `src/patch_mtp_draft_vocab.py`
- `src/patch_mamba_block_size.py`
- `src/patch_qsa_exact_topk.py`
- `src/patch_qsa_fp8_kv.py`
- `src/mamba_utils_guarded.py`

`src/patch_mtp_fp8.py` is required by `Dockerfile.nvidia-nvfp4mtp`. An earlier
publication omitted it; it is now part of the tracked release and must not be
removed while that Dockerfile exists.

## SGLang v0.5.20 path

- `src/sglang/patch_sglang_block_fp8_lmhead.py`
- `src/sglang/patch_sglang_draft65k.py`
- `src/sglang/patch_sglang_ple_reuse.py`
- `src/sglang/sglang_ple_reuse_helper.py`
- `src/sglang/patch_sglang_prefill_progress.py`
- `src/sglang/patch_sglang_redact_secrets.py`
- `Dockerfile.sglang`
- `recipes/sglang/compose.example.yaml`

The patch scripts are anchored, idempotent where practical, and syntax-gated.
An anchor mismatch aborts the image build instead of silently producing a
partially patched runtime.

## Release audit

Run before publishing:

```bash
python3 tools/verify_release_files.py
python3 -m py_compile src/*.py src/sglang/*.py tools/*.py
```

This verifies every local `COPY`/`ADD` source in the Dockerfiles and the
required vLLM/SGLang manifest entries. Remote URL `ADD` sources remain protected
by their Dockerfile checksums.
