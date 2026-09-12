---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model:
  - nvidia/Qwen3.8-Flash-Next-NVFP4
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - multimodal
  - vision-language
  - video
  - dgx-spark
  - nvfp4
  - fp8
  - modelopt
  - vllm
  - quantization
---

# Qwen3.8-Flash-Next NVIDIA Hybrid Single Spark

Run Qwen3.8-Flash-Next on one NVIDIA DGX Spark using the official NVIDIA
NVFP4 checkpoint, locally converted FP8 E4M3 dense side layers, and an NVFP4
MTP expert graft. This repository extends
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
at pinned commit `bd60fcb1b492ca920f74df7462f05da7b6d98f73` and preserves its MIT
license and contributor credits.

The checkpoint retains the base model's multimodal architecture and processor:
text, image, and video-frame inputs are supported by a compatible vLLM build.
The Hugging Face pipeline category is therefore `image-text-to-text`, matching
the upstream Qwen model. Video is carried by the same multimodal chat API; it
is not a separate Hugging Face pipeline category. Runtime media limits and
frame sampling still depend on the serving configuration.

> **Want the fastest tested NVIDIA-main lane?** Use the separately published
> [FP8 `lm_head` checkpoint](https://huggingface.co/Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-FP8-LMHead-Single-Spark).
> It keeps this NVIDIA NVFP4 main checkpoint, FP8 side layers and NVFP4 MTP,
> while converting only the target model's final projection to blockwise FP8.
> On the same 100K profile it measured 28.11 instead of 25.92 decode tok/s
> (+8.4%), with no directional regression observed in our three-run 162-case
> local suite. That is bounded local evidence, not proof of model equivalence.

## What this lane adds

- NVIDIA's official `nvidia/Qwen3.8-Flash-Next-NVFP4` is the main checkpoint.
- 300 dense side-layer linears are converted locally to blockwise FP8 E4M3:
  GDN `in_proj_qkv`, `in_proj_z`, `out_proj`; QSA q/k/v/o; and shared-expert
  gate/up/down projections.
- `in_proj_ba`, norms, gates, hyperconnection parameters, and `lm_head` remain
  BF16. The current blockwise-FP8 loader does not support `in_proj_ba`.
- NVIDIA's FP8 MTP expert block is replaced with the pinned Inferact per-expert
  NVFP4 donor. The target model still verifies speculative tokens.
- The FP8 hybrid shim is retargeted from `ModelOptNvFp4Config` to the official
  checkpoint's real `ModelOptMixedPrecisionConfig`.
- PLE stays FP8 and is served from NVMe with mmap; KV cache and recurrent state
  remain BF16 in the measured profile.

The complete converted checkpoint is published in the
[Hugging Face model repository](https://huggingface.co/Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark).
The GitHub source repository contains only code and documentation. The source
NVIDIA checkpoint is never overwritten during local preparation: the scripts
use an isolated destination and refuse to replace an existing one.

## Measured on one DGX Spark

Hardware: GB10, 128 GB unified memory. Runtime: vLLM preview pinned by the base
repository, MTP 3, 65,536-token reduced draft vocabulary, deterministic QSA
top-k, BF16 KV/recurrent state, 500,000-token YaRN profile, prefix caching on.

| Profile | 8K prefill | 8K decode | 100K prefill | 100K decode |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA BF16-side + NVFP4 MTP | 2253.28 | 20.36 | 2177.88 | 21.21 |
| **NVIDIA FP8-side + NVFP4 MTP** | **2303.64** | **24.77** | **2129.24** | **25.92** |
| NVIDIA FP8-side + FP8 `lm_head` + NVFP4 MTP | 2233.42 | **27.34** | **2141.79** | **28.11** |
| RadixArk FP8-side + NVFP4 MTP | 2341.93 | 26.05 | 2154.00 | 26.86 |

Throughput units are tokens/s. The NVIDIA hybrid's three 100K runs averaged
66.88 seconds end-to-end and 33.73% weighted MTP acceptance. It loaded 73.89
GiB of weights and profiled a 19.73 GiB / 721,556-token KV pool.

### Acceptance-scaled decode estimate

MTP 3 verifies a fixed width of four output positions per engine step: one
target token plus up to three accepted draft tokens. The measured 100K counter
rates imply mean accepted lengths of `1 + 3 x 0.3373 = 2.012` for the BF16
`lm_head` parent and `1 + 3 x 0.3277 = 1.983` for the FP8 `lm_head` lane. Dividing
measured decode by those lengths gives approximately 12.88 and 14.17 engine
steps/s respectively.

If the verification width and per-step cost stay comparable, decode scales
approximately with mean accepted length:

| Mean accepted length (max 4) | Implied draft acceptance | BF16 `lm_head` estimate | FP8 `lm_head` estimate |
| ---: | ---: | ---: | ---: |
| 2.0 | 33.3% | 25.77 tok/s | 28.35 tok/s |
| 2.5 | 50.0% | 32.21 tok/s | 35.44 tok/s |
| 3.0 | 66.7% | 38.65 tok/s | 42.52 tok/s |
| 3.5 | 83.3% | 45.09 tok/s | 49.61 tok/s |
| **3.71** | **90.3%** | **47.80 tok/s** | **52.59 tok/s** |
| 4.0 | 100% | 51.53 tok/s | 56.70 tok/s |

The `3.71/4` row is useful for predictable coding output only when the live
request actually reports that mean accepted length. These are linear estimates
anchored to the measured 100K runs, not a measured coding benchmark. PLE page
locality, output distribution, context length, concurrency and runtime warmup
can change engine-step cost, so report the live decode rate whenever available.

### Local quality regression

This is a private 162-case deployment regression suite, not an official Qwen,
NVIDIA, or community benchmark. Three repeated runs of the uploaded NVIDIA
FP8-side checkpoint scored 144/162, 146/162, and 148/162 (mean 146/162); the
paired NVIDIA BF16-side and RadixArk hybrid comparison profiles each scored
147/162. Pass/fail status was identical for 156/162 cases across all three
runs: 143 always passed, 13 always failed, and only six MMLU boundary cases
varied. In the latest 148-point run, each 147-point comparison had five
discordant cases: three uploaded-checkpoint-only passes and two baseline-only
passes (exact paired two-sided p=1.0). Only 107/162 raw outputs were
byte-identical across all three runs, showing that this concurrent MTP runtime
is not strictly byte-deterministic even at temperature zero. This sample does
not establish a reliable quality difference; report the observed 144-148 range
rather than selecting a single run. Tool calling was 4/4 and long-context
retrieval was 6/6 in all three uploaded-checkpoint runs. Validate
application-specific prompts before replacing a quality-first setup.

### Experimental block-FP8 `lm_head`

The optional `Dockerfile.fp8-lm-head` and
`recipes/nvidia-hybrid/quantize_lm_head_fp8.py` extend the same checkpoint with
a 128x128 blockwise FP8 E4M3 final projection. The converter writes an isolated
destination, keeps the source checkpoint unchanged, updates the ModelOpt mixed
metadata and safetensors index, and emits a separate 65,536-row BF16 draft head.
The MTP transformer and experts remain NVFP4; only its small reduced-vocabulary
draft output head remains BF16.

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

The pinned vLLM preview needs the companion-scale loader in
`src/patch_fp8_lm_head.py`; this is a narrow port of the unmerged upstream
`ParallelLMHead` block-FP8 work. Three local 162-case runs scored 147, 146 and
148 (mean 147), including 4/4 tool calls and 6/6 long-context retrieval in every
run. Compared with three runs of the BF16 `lm_head` profile, 141 cases always
passed in both profiles and 11 always failed in both. The remaining differences
were boundary cases rather than a directional regression.

The measured 100K profile improved decode from 25.92 to 28.11 tok/s while
prefill remained effectively flat (2129.24 to 2141.79 tok/s). The first 8K run
still included runtime JIT work; its three-run mean was 2233.42/27.34 tok/s.
Treat this as an experimental lane until the loader support lands upstream and
validate application-specific long conversations before publishing derivative
weights.

The ready FP8-head checkpoint is published separately at
[Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-FP8-LMHead-Single-Spark](https://huggingface.co/Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-FP8-LMHead-Single-Spark),
so the earlier BF16-head checkpoint remains available and unchanged.

## Download the ready checkpoint

The Hugging Face repository contains all 10 main shards, the 51.2 GB FP8 PLE
shard, the 1.6 GB NVFP4 MTP donor shard, configs, processors, tokenizer, and
the 33 MB safetensors index. Reported repository storage is approximately
128.90 GB decimal (about 120.05 GiB).

```bash
hf download \
  Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark \
  --local-dir /data/models/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark
```

The uploaded weight revision was committed as
`15edf04a1b38dce19dffef9c7de77c8a7522561b`. Continue below only if you want
to reproduce the conversion yourself from the parent checkpoints.

## Build the runtime image

Build the pinned upstream base, add the NVFP4 MTP dispatch patch, and then
retarget the FP8-side shim:

```bash
docker build -t nvidia-hybrid-single-spark:base .
docker build -f Dockerfile.nvidia-nvfp4mtp \
  -t nvidia-hybrid-single-spark:mtp .
docker build -f Dockerfile.nvidia-hybrid \
  -t nvidia-hybrid-single-spark:latest .
```

## Prepare the checkpoint

Download the official NVIDIA checkpoint and the pinned Inferact donor first.
The donor file must be named `nvfp4_experts_mtp.safetensors`; the preparation
script verifies SHA-256
`0d44e6d705d2313c713e60114e56874adf358ed5f646dc8704bb5be15f5ddbf7`.

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
shards. The second step requires at least 64 GiB free while repacking the mixed
PLE/MTP shard. Do not point any destination variable at your original model.

## Run

```bash
export FINAL_MODEL_DIR=/data/models/Qwen3.8-Flash-Next-NVIDIA-FP8-Hybrid-MTPNVFP4
export CACHE_DIR=/data/cache/nvidia-hybrid-single-spark
export API_KEY_FILE=/run/secrets/qwen-api-key
docker compose -f recipes/nvidia-hybrid/compose.example.yaml up -d
```

The example publishes port `30000`, serves model alias `qwen38-flash-next`,
uses `restart: "no"`, and enables a 500K YaRN context profile. Watch memory
headroom closely: the measured host had about 12 GiB available after startup.

## Why not native `FP8_PB_WO` dispatch?

The quant metadata resolver correctly identifies the 300 side layers, but the
pinned preview combines a legacy `MergedColumnParallelLinear.load_weights`
path with a newer parameter contract and fails while loading a fused layer:

```text
AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'
```

This lane therefore reuses the proven block-FP8 `Fp8Config` loader and only
retargets its owner class to `ModelOptMixedPrecisionConfig`.

## Related work and credits

- [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX):
  complete single-Spark baseline, PLE mmap, prefix-cache/QSA fixes, FP8-side
  hybrid conversion, reduced draft vocabulary, and NVFP4 MTP graft design.
- [tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark):
  official NVIDIA checkpoint on one Spark without requantizing its side layers.
- [dolf3131/qwen3.8-flash-next-dgx-spark](https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark):
  official NVIDIA mixed-precision and single-Spark loader research.
- [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound):
  FP8 side-layer conversion and GB10 kernel work credited by the base project.
- [Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4):
  pinned NVFP4 MTP donor.

This is an independent integration and validation recipe, not an NVIDIA
official repository. NVIDIA, DGX, and related names are trademarks of their
respective owners.

The code and patches in this repository retain the upstream MIT license. The
Hugging Face repository redistributes the converted checkpoint under the model
licenses applicable to the NVIDIA parent and Inferact donor, including the
NVIDIA Open Model License and the terms stated on each model card.

Redistributed derivative weights include the required attribution in
[`NOTICE`](NOTICE). A copy of the NVIDIA Open Model License and the applicable
Qwen Community License accompany the repository as
`NVIDIA-OPEN-MODEL-LICENSE.pdf` and `QWEN-COMMUNITY-LICENSE.txt`.
