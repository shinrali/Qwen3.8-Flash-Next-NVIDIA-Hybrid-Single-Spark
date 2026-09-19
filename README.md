# Qwen3.8-Flash-Next NVIDIA Hybrid for Blackwell

> **Recommended checkpoint:**
> [FP8 `lm_head`](https://huggingface.co/Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-FP8-LMHead-Single-Spark)
> · [BF16 `lm_head` reference](https://huggingface.co/Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark)
> · [all Hugging Face releases](https://huggingface.co/collections/Shinrali/qwen38-flash-next-nvidia-hybrid-for-dgx-spark-6aa603b1e29d415b0d8d8ebd)

Run Qwen3.8-Flash-Next on one NVIDIA DGX Spark or a sufficiently large
Blackwell workstation. The project combines NVIDIA's official NVFP4 target
checkpoint with locally converted FP8 side layers, an NVFP4 MTP expert graft,
FP8 PLE mmap/offload and optional matched 65K FP8 draft heads. It publishes two
checkpoint lanes and three validated runtime profiles across vLLM and SGLang.

This repository extends
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
at pinned commit `bd60fcb1b492ca920f74df7462f05da7b6d98f73` and retains
its upstream credits and MIT-licensed code notices.

## Choose a checkpoint

| Lane | Use it when | Main difference |
| --- | --- | --- |
| **[FP8 `lm_head`](https://huggingface.co/Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-FP8-LMHead-Single-Spark)** | You want the fastest quality-preserving NVIDIA-main profile validated here | Complete target `lm_head` is block-FP8 E4M3; includes public and personal matched 65K FP8 draft pairs |
| [BF16 `lm_head`](https://huggingface.co/Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark) | You want the conservative reference or rollback checkpoint | Target `lm_head` remains BF16; main experts, side layers, MTP graft and PLE layout otherwise match |

The FP8 target head measured 28.11 instead of 25.92 decode tok/s on the same
100K DGX Spark vLLM profile (+8.4%). A bounded local regression suite found no
directional quality loss, but it is not proof of model equivalence. Validate
application-specific prompts before replacing a quality-first setup.

## Validated runtime matrix

The rows below use different workloads and are representative validations, not
a single cross-platform leaderboard.

| Host | Backend | Checkpoint and draft | Validated boundary | Representative measurement |
| --- | --- | --- | --- | --- |
| DGX Spark, GB10 128 GB | patched vLLM preview | FP8 `lm_head` + personal 65K FP8 draft | 500K YaRN profile; BF16 KV/recurrent state | 100K: 2141.79 prefill, 28.11 decode tok/s |
| DGX Spark, GB10 128 GB | SGLang v0.5.20 | FP8 `lm_head` + personal 65K FP8 draft | 262,144 context; persistent FP8 PLE | 4K hot run: 2663.39 prefill, 42.53 decode tok/s, 56.11% MTP |
| RTX PRO 6000 Blackwell, Windows Docker Desktop/WSL2 | same patched vLLM preview | FP8 `lm_head` + personal 65K FP8 draft | BF16 KV; 16 GiB explicit KV; two 262,144-token slots | 4K three-run mean: 10,223.18 prefill, 91.66 decode tok/s, 47.66% MTP |

The WSL2 harness used a fresh UUID nonce near the beginning of every prompt,
so its 10.2K tok/s result is not a repeated 4K prefix-cache hit. It remains a
single-stream short-context result and does not establish 256K prefill speed or
two-stream aggregate throughput. See [the full benchmark record](docs/BENCHMARKS.md).

## Quick start

### Download the recommended checkpoint

```bash
hf download \
  Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-FP8-LMHead-Single-Spark \
  --local-dir /data/models/Qwen3.8-Flash-Next-NVIDIA-Hybrid-FP8-LMHead-Single-Spark
```

Use the BF16 reference repository instead only when you deliberately want the
rollback lane. The source NVIDIA checkpoint is never modified by the conversion
scripts; local preparation always targets an isolated destination.

### Build the patched vLLM runtime

```bash
git clone https://github.com/shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark.git
cd Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark

docker build -t nvidia-hybrid-single-spark:base .
docker build -f Dockerfile.nvidia-nvfp4mtp \
  -t nvidia-hybrid-single-spark:mtp .
docker build -f Dockerfile.nvidia-hybrid \
  -t nvidia-hybrid-single-spark:latest .
docker build -f Dockerfile.fp8-lm-head \
  -t nvidia-hybrid-single-spark:fp8-lm-head .
docker build -f Dockerfile.reduced-fp8-draft \
  -t nvidia-hybrid-single-spark:fp8-draft .
```

Set the matched reduced-vocabulary files together. Never mix a vocabulary array
with a head generated from a different token-ID order.

```text
VLLM_MTP_DRAFT_VOCAB=/draft/studio_draft_vocab_65536_all.npy
VLLM_MTP_DRAFT_HEAD=/draft/mtp_draft_head_studio_65536_all_fp8.safetensors
```

The `studio_` prefix is a historical filename retained for compatibility. The
pair contains token IDs and matched head rows only, with no source prompts or
application dependency.

Use [`recipes/nvidia-hybrid/compose.example.yaml`](recipes/nvidia-hybrid/compose.example.yaml)
for vLLM. The WSL2-specific boundary and non-portable QSA extension warning are
documented in [`docs/WINDOWS-WSL2-RTX-PRO-6000.md`](docs/WINDOWS-WSL2-RTX-PRO-6000.md).
Do not enable `--enforce-eager` in the validated performance profile.

### Build the SGLang runtime

`Dockerfile.sglang` and every startup patch it invokes are tracked here. Build
and runtime auditing are documented in [`docs/PATCH-MANIFEST.md`](docs/PATCH-MANIFEST.md),
while the matched 65K draft setup and measured Spark profile are in
[`docs/PERSONAL-65K-SGLANG-2026-09-19.md`](docs/PERSONAL-65K-SGLANG-2026-09-19.md).

## Precision layout

| Component | BF16 reference | FP8 recommended |
| --- | --- | --- |
| Main routed experts | NVIDIA W4A4 NVFP4 | NVIDIA W4A4 NVFP4 |
| 300 GDN/QSA/shared-expert side linears | block-FP8 E4M3 | block-FP8 E4M3 |
| Main target `lm_head` | BF16 | block-FP8 E4M3, 128x128 |
| MTP routed experts | Inferact per-expert NVFP4 donor | Inferact per-expert NVFP4 donor |
| Reduced MTP head | optional BF16 or FP8 matched head | FP8 matched head recommended |
| PLE/n-gram table | FP8 E4M3, NVMe mmap | FP8 E4M3, NVMe mmap |
| Validated KV/recurrent state | BF16 | BF16 |

The 300 converted side linears cover GDN `in_proj_qkv`, `in_proj_z`, `out_proj`;
QSA q/k/v/o; and shared-expert gate/up/down projections. `in_proj_ba`, norms,
gates and hyperconnection parameters remain BF16. The hybrid shim is retargeted
from `ModelOptNvFp4Config` to NVIDIA's real `ModelOptMixedPrecisionConfig`.

The target model always verifies speculative tokens. Reducing and quantizing
the draft output head changes draft cost and proposal distribution; it does not
remove full-target verification.

## Benchmark summary

### DGX Spark — patched vLLM

Hardware: GB10, 128 GB unified memory. Profile: MTP 3, personal 65K draft,
deterministic QSA top-k, BF16 KV/recurrent state, 500K YaRN and prefix caching.

| Profile | 8K prefill | 8K decode | 100K prefill | 100K decode |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA BF16-side + NVFP4 MTP | 2253.28 | 20.36 | 2177.88 | 21.21 |
| FP8-side + BF16 `lm_head` + NVFP4 MTP | 2303.64 | 24.77 | 2129.24 | 25.92 |
| **FP8-side + FP8 `lm_head` + NVFP4 MTP** | 2233.42 | **27.34** | **2141.79** | **28.11** |
| RadixArk FP8-side + NVFP4 MTP | 2341.93 | 26.05 | 2154.00 | 26.86 |

### DGX Spark — SGLang v0.5.20

| Draft configuration | Prefill tok/s | Decode tok/s | MTP acceptance |
| --- | ---: | ---: | ---: |
| Complete vocabulary | 2604.22 | 37.31 | 46.94% |
| Personal 65K, first run | 2604.29 | 37.47 | 48.61% |
| Personal 65K, hot repeat | 2663.39 | 42.53 | 56.11% |

### RTX PRO 6000 — WSL2 patched vLLM

| Prompt / output | Runs | Mean TTFT | Mean prefill | Mean decode | Mean MTP | Mean end-to-end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,088 / 512 tokens | 3 | 0.400 s | 10,223.18 tok/s | 91.66 tok/s | 47.66% | 5.985 s |

All detailed methodology, per-run WSL2 values, acceptance-scaled estimates and
scope limits are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Quality validation

The local 162-case suite is a regression check, not an official Qwen, NVIDIA or
community benchmark. The FP8-side/BF16-`lm_head` checkpoint scored 144, 146 and
148; the FP8-`lm_head` checkpoint scored 147, 146 and 148. Tool calling was 4/4
and long-context retrieval was 6/6 in every published-checkpoint run. Boundary
MMLU items varied between runs even at temperature zero, so report ranges rather
than selecting one favorable score.

See [`docs/QUALITY-VALIDATION.md`](docs/QUALITY-VALIDATION.md) for paired counts,
non-determinism evidence and interpretation limits.

## Capabilities and limitations

- The checkpoint retains the upstream multimodal processor and architecture.
  Text, image and sampled video-frame inputs work with a compatible runtime;
  `image-text-to-text` is therefore the correct Hugging Face pipeline category.
- Stock vLLM and stock SGLang do not load every mixed-precision component used
  here. Use the pinned images and the complete patch manifest.
- The optional deterministic QSA binary built for GB10 is not portable to RTX
  PRO 6000. Rebuild it for the workstation architecture or use the validated
  standard fallback.
- Personal 65K draft vocabulary improves the measured workload but is not
  universally optimal. Measure acceptance on your own output distribution.
- Context capacity, concurrent capacity and speed depend on KV dtype, recurrent
  state, host memory, PLE residency and runtime arguments. Do not infer a 256K
  or two-stream speed from the published 4K WSL2 result.

## Documentation

- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — complete performance evidence.
- [`docs/QUALITY-VALIDATION.md`](docs/QUALITY-VALIDATION.md) — local regression evidence.
- [`docs/FP8-LM-HEAD.md`](docs/FP8-LM-HEAD.md) — FP8 target/draft-head conversion.
- [`docs/BUILD-FROM-SOURCE.md`](docs/BUILD-FROM-SOURCE.md) — reproduce the checkpoint.
- [`docs/WINDOWS-WSL2-RTX-PRO-6000.md`](docs/WINDOWS-WSL2-RTX-PRO-6000.md) — workstation runtime.
- [`docs/PERSONAL-65K-SGLANG-2026-09-19.md`](docs/PERSONAL-65K-SGLANG-2026-09-19.md) — SGLang 65K path.
- [`docs/PATCH-MANIFEST.md`](docs/PATCH-MANIFEST.md) — required runtime patches.
- [`docs/HOW-IT-WORKS.md`](docs/HOW-IT-WORKS.md) — architecture and implementation notes.

## Related work and credits

- [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) —
  single-Spark baseline, PLE mmap, prefix-cache/QSA fixes, hybrid conversion and
  reduced-draft work.
- [tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark) —
  NVIDIA checkpoint on one Spark without requantizing its side layers.
- [dolf3131/qwen3.8-flash-next-dgx-spark](https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark) —
  official NVIDIA mixed-precision and single-Spark loader research.
- [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound) —
  FP8 side-layer conversion and GB10 kernel work credited upstream.
- [Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4) —
  pinned per-expert NVFP4 MTP donor.

This is an independent integration and validation project, not an NVIDIA
official repository. Code and patches retain applicable upstream MIT notices.
Redistributed derivative weights remain subject to the NVIDIA Open Model
License, Qwen Community License and donor terms. Required attribution is in
[`NOTICE`](NOTICE); license copies are included in the repository.
