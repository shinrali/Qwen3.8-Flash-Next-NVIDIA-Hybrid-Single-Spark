# Build the hybrid checkpoint from source

Use this path only when reproducing the conversion. The ready checkpoints are
already published on Hugging Face. Never point a destination at the original
NVIDIA model or an actively served directory.

## Inputs

Download:

- `nvidia/Qwen3.8-Flash-Next-NVFP4`;
- the pinned Inferact NVFP4 MTP donor.

The donor file must be named `nvfp4_experts_mtp.safetensors`. The preparation
script verifies SHA-256:

```text
0d44e6d705d2313c713e60114e56874adf358ed5f646dc8704bb5be15f5ddbf7
```

## Prepare the BF16-target-head reference

```bash
export MODEL_ROOT=/data/models

# 1. Hard-link the NVIDIA checkpoint into an isolated working copy and convert
#    the 300 supported side linears to blockwise FP8.
recipes/nvidia-hybrid/prepare-nvidia-hybrid.sh

# 2. Remove NVIDIA's FP8 MTP expert tensors and graft the NVFP4 donor.
python3 recipes/nvidia-hybrid/prepare_nvidia_fp8side_nvfp4_mtp.py

# 3. Restore the scale metadata contract used by the proven Fp8Config shim.
export FINAL_MODEL_DIR="$MODEL_ROOT/Qwen3.8-Flash-Next-NVIDIA-FP8-Hybrid-MTPNVFP4"
python3 recipes/nvidia-hybrid/rewrite_fp8_side_scales_shim.py
```

The first step requires hard-link support and about 13 GiB for rewritten side
shards. The second requires at least 64 GiB free while repacking the mixed
PLE/MTP shard.

## Build the vLLM runtime

```bash
docker build -t nvidia-hybrid-single-spark:base .
docker build -f Dockerfile.nvidia-nvfp4mtp \
  -t nvidia-hybrid-single-spark:mtp .
docker build -f Dockerfile.nvidia-hybrid \
  -t nvidia-hybrid-single-spark:latest .
```

The quant metadata resolver identifies the 300 side layers, but native
`FP8_PB_WO` dispatch in the pinned preview combines a legacy
`MergedColumnParallelLinear.load_weights` path with a newer parameter contract
and fails while loading a fused layer:

```text
AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'
```

The validated lane therefore reuses the block-FP8 `Fp8Config` loader and
retargets its owner class to `ModelOptMixedPrecisionConfig`.

## Serve

```bash
export FINAL_MODEL_DIR=/data/models/Qwen3.8-Flash-Next-NVIDIA-FP8-Hybrid-MTPNVFP4
export CACHE_DIR=/data/cache/nvidia-hybrid-single-spark
export API_KEY_FILE=/run/secrets/qwen-api-key
docker compose -f recipes/nvidia-hybrid/compose.example.yaml up -d
```

The example publishes port `30000`, serves alias `qwen38-flash-next`, uses
`restart: "no"` and enables the validated YaRN profile. Keep API keys outside
the repository and monitor memory headroom. See [`FP8-LM-HEAD.md`](FP8-LM-HEAD.md)
for the recommended target-head derivative.
