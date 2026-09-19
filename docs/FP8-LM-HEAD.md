# FP8 target and reduced draft heads

The recommended checkpoint converts the complete `248320 x 2560` target
`lm_head.weight` from BF16 to 128x128 blockwise FP8 E4M3. It keeps the NVIDIA
NVFP4 routed experts, locally converted FP8 side layers, NVFP4 MTP experts and
FP8 PLE layout unchanged.

Stock vLLM does not yet load this target head. The pinned runtime needs the
opt-in `ParallelLMHead` companion-scale loader in `src/patch_fp8_lm_head.py`.

## Convert an isolated checkpoint

```bash
cp -al \
  /data/models/Qwen3.8-Flash-Next-NVIDIA-FP8-Hybrid-MTPNVFP4 \
  /data/models/Qwen3.8-Flash-Next-NVIDIA-FP8-Hybrid-MTPNVFP4-FP8Head

python3 recipes/nvidia-hybrid/quantize_lm_head_fp8.py \
  /data/models/Qwen3.8-Flash-Next-NVIDIA-FP8-Hybrid-MTPNVFP4-FP8Head \
  --draft-vocab src/draft_vocab_65536.npy

docker build -f Dockerfile.fp8-lm-head \
  -t nvidia-hybrid-single-spark:fp8-lm-head .
```

Never point the destination at the source checkpoint. The converter updates the
ModelOpt mixed metadata and safetensors index in the isolated destination.

## Build a matched block-FP8 reduced draft head

```bash
python3 tools/build_reduced_fp8_head.py \
  /data/models/Qwen3.8-Flash-Next-NVIDIA-NVFP4 \
  src/draft_vocab_65536.npy \
  /data/models/Qwen3.8-Flash-Next-NVIDIA-FP8-Hybrid-MTPNVFP4-FP8Head/mtp_draft_head_65536_fp8.safetensors

docker build -f Dockerfile.reduced-fp8-draft \
  -t nvidia-hybrid-single-spark:fp8-draft .
```

The public matched pair is:

| File | SHA-256 |
| --- | --- |
| `draft_vocab_65536.npy` | `6459e0fdc8df30e0c1d1f45be7c1b6bef0d68b0e73c073a82c52ea2a7f4b26d4` |
| `mtp_draft_head_65536_fp8.safetensors` | `25c4d394b283d3dd6f117a24d8aeb050f16f26d04a9decc81918a4d325912ab6` |

The optional personally optimized matched pair is:

| File | SHA-256 |
| --- | --- |
| `studio_draft_vocab_65536_all.npy` | `a2f067314f15fc5725896af951ddd8184702b46ba5e8ea56c57b09e493d16bb2` |
| `mtp_draft_head_studio_65536_all_fp8.safetensors` | `f02cb253f71c934041c0377572a63cae8687513790198e4434a5fc1460b78df1` |

The `studio_` prefix is historical only. These files contain selected token IDs
and matched quantized head rows, not prompts or application data.

The reduced FP8 head has shape `(65536, 2560)` with `float8_e4m3fn` weights and
a `(512, 20)` FP32 `weight_scale_inv`. It is 167,813,472 bytes, versus roughly
320 MiB for the BF16 reduced head.

Reduced-head row `i` must correspond to the complete source-head row selected by
vocabulary ID `i`. Renaming is safe; reordering or replacing only one file is
not. Every proposed token is still verified by the complete target model.

Measured performance and variance are in [`BENCHMARKS.md`](BENCHMARKS.md).
