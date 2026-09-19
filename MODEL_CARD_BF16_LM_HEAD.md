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

# Qwen3.8-Flash-Next NVIDIA Hybrid — BF16 LM Head Reference

> [!NOTE]
> This is the conservative BF16-`lm_head` reference and rollback checkpoint.
> The separately published
> [FP8 `lm_head` checkpoint](https://huggingface.co/Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-FP8-LMHead-Single-Spark)
> is the faster NVIDIA-main lane validated by this project.
>
> [GitHub source and runtime](https://github.com/shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark)
> · [all Hugging Face releases](https://huggingface.co/collections/Shinrali/qwen38-flash-next-nvidia-hybrid-for-dgx-spark-6aa603b1e29d415b0d8d8ebd)

This checkpoint starts from NVIDIA's official
[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4),
converts 300 supported GDN/QSA/shared-expert side linears to blockwise FP8 E4M3,
and replaces NVIDIA's FP8 MTP expert block with the pinned Inferact per-expert
NVFP4 donor. The complete target `lm_head` remains BF16.

## Precision layout

| Component | Format |
| --- | --- |
| Main routed experts | NVIDIA W4A4 NVFP4 |
| 300 GDN/QSA/shared-expert side linears | blockwise FP8 E4M3 |
| Main target `lm_head` | BF16 |
| `in_proj_ba`, norms, gates and hyperconnection parameters | BF16 |
| MTP routed experts | Inferact per-expert NVFP4 donor |
| PLE/n-gram table | FP8 E4M3, NVMe mmap at runtime |
| Validated KV and recurrent state | BF16 |

The target model still verifies every speculative token. The checkpoint retains
the upstream multimodal processor and architecture, including compatible text,
image and sampled video-frame inputs. Hugging Face therefore categorizes it as
`image-text-to-text`.

## Runtime requirement

Stock vLLM does not load this mixed checkpoint. Use the pinned source and patch
manifest in
[`shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark`](https://github.com/shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark).
The repository also carries the SGLang v0.5.20 path, but this BF16 target-head
model card records the vLLM reference checkpoint rather than claiming that all
backend schedules are byte-identical.

```bash
hf download \
  Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark \
  --local-dir /data/models/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark
```

The uploaded weight revision is
`15edf04a1b38dce19dffef9c7de77c8a7522561b`. See the source repository's
[`BUILD-FROM-SOURCE.md`](https://github.com/shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark/blob/main/docs/BUILD-FROM-SOURCE.md)
to reproduce the conversion without modifying the NVIDIA parent checkpoint.

## DGX Spark measurements

Hardware: one GB10 DGX Spark with 128 GB unified memory. Runtime: patched vLLM,
MTP 3, 65,536-token reduced draft vocabulary, deterministic QSA top-k, BF16
KV/recurrent state, 500,000-token YaRN profile and prefix caching.

| Profile | 8K prefill | 8K decode | 100K prefill | 100K decode |
| --- | ---: | ---: | ---: | ---: |
| This BF16-`lm_head` checkpoint | 2303.64 | 24.77 | 2129.24 | 25.92 |
| FP8-`lm_head` derivative | 2233.42 | 27.34 | 2141.79 | 28.11 |

Throughput units are tokens/s. The BF16 target-head checkpoint's three 100K
runs averaged 66.88 seconds end-to-end with 33.73% weighted MTP acceptance. It
loaded 73.89 GiB of weights and profiled a 19.73 GiB / 721,556-token KV pool.

These measurements are checkpoint- and workload-specific. The RTX PRO 6000
WSL2 result published by this project used the recommended FP8-`lm_head` lane;
it must not be attributed to this BF16 checkpoint.

## Local quality regression

The local 162-case regression suite is not an official Qwen, NVIDIA or community
benchmark. This FP8-side/BF16-`lm_head` checkpoint scored 144, 146 and 148 across
three temperature-zero runs. Tool calling was 4/4 and long-context retrieval
was 6/6 in every run. Boundary cases varied despite temperature zero, so the
observed range is more honest than selecting one favorable score.

Full paired methodology and interpretation limits are in
[`docs/QUALITY-VALIDATION.md`](https://github.com/shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark/blob/main/docs/QUALITY-VALIDATION.md).

## License and attribution

This is an independent derivative, not an NVIDIA official repository.
Redistributed weights remain subject to the NVIDIA Open Model License, Qwen
Community License and Inferact donor terms. The source repository includes
`NOTICE`, the applicable license copies, conversion scripts and contributor
credits.
