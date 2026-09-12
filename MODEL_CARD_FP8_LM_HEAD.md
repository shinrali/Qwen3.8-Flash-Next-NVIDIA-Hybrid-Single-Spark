---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model:
  - Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark
  - nvidia/Qwen3.8-Flash-Next-NVFP4
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - dgx-spark
  - nvfp4
  - fp8
  - modelopt
  - vllm
  - quantization
---

# Qwen3.8-Flash-Next NVIDIA Hybrid FP8 LMHead Single Spark

This is an experimental single-DGX-Spark derivative of
[`Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark`](https://huggingface.co/Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark).
It keeps the NVIDIA-based hybrid checkpoint and converts only the complete
`248320 x 2560` final `lm_head.weight` from BF16 to 128x128 blockwise FP8 E4M3.
It is intended to reduce the final-projection bandwidth cost without moving to
an INT3 target model.

For users arriving from the BF16-`lm_head` parent: this is the faster tested
lane. On the same 100K profile it measured 28.11 instead of 25.92 decode tok/s
(+8.4%). Three local 162-case runs found no directional quality regression, but
that bounded regression suite does not prove model equivalence.

The reproducible converter, pinned runtime patches, Dockerfiles and validation
notes are in
[`shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark`](https://github.com/shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark).

## Precision layout

| Component | Format |
| --- | --- |
| Main routed experts | NVIDIA W4A4 NVFP4 |
| 300 GDN/QSA/shared-expert side linears | blockwise FP8 E4M3 |
| Main `lm_head` | blockwise FP8 E4M3, 128x128 |
| MTP routed experts | NVFP4 donor from Inferact |
| MTP reduced draft output head | BF16, 65,536 rows |
| PLE/n-gram table | FP8 E4M3, NVMe mmap at runtime |
| Validated KV and recurrent state | BF16 |

The MTP model itself was not changed to BF16. Only the small reduced-vocabulary
draft output head remains BF16 because the custom proposer currently performs a
plain `F.linear`; every proposed token is still verified by the target model.

## Runtime requirement

Stock vLLM does not yet load this checkpoint. It needs the opt-in
`ParallelLMHead` FP8 companion-scale loader and Qwen3.8 model plumbing included
in the linked GitHub repository. Build the experimental image:

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
```

Set both variables when serving:

```text
VLLM_MTP_DRAFT_VOCAB=/opt/llm/draft_vocab_65536.npy
VLLM_MTP_DRAFT_HEAD=/model/mtp_draft_head_65536.safetensors
```

Use the repository's `recipes/nvidia-hybrid/compose.example.yaml` as the serving
reference. Do not use `VLLM_MTP_DRAFT_HEAD` without the corresponding fixed
65,536-token ID list.

## One DGX Spark measurements

Hardware: GB10 with 128 GB unified memory. Profile: vLLM pinned by the base
repository, MTP 3, deterministic QSA top-k, BF16 KV/recurrent state, 500,000
token YaRN context, prefix caching enabled, 512 generated tokens per speed run.

| Profile | 8K prefill | 8K decode | 100K prefill | 100K decode |
| --- | ---: | ---: | ---: | ---: |
| BF16 `lm_head` parent | 2303.64 | 24.77 | 2129.24 | 25.92 |
| This FP8 `lm_head` checkpoint | 2233.42 | 27.34 | 2141.79 | 28.11 |

The three 100K runs measured 2140.69/28.46, 2143.12/26.62 and
2141.57/29.25 prefill/decode tok/s. Weighted MTP acceptance was 32.77% and mean
end-to-end time was 65.08 seconds. The first formal 8K run still included
runtime JIT work, so interpret the 8K mean conservatively.

The engine profiled 18.85 GiB of BF16 KV for 688,622 tokens. Host memory state
affects automatic KV sizing; this release does not claim that the theoretical
606 MiB disk/weight saving always becomes additional KV capacity.

### Acceptance-scaled decode estimate

With MTP 3, mean accepted length ranges from one to four output tokens per
engine step. The measured weighted draft-acceptance rates imply mean lengths of
2.012 for the BF16 parent and 1.983 for this FP8 lane. Their measured 100K
decode rates therefore correspond to approximately 12.88 and 14.17 engine
steps/s.

| Mean accepted length (max 4) | Implied draft acceptance | BF16 `lm_head` estimate | FP8 `lm_head` estimate |
| ---: | ---: | ---: | ---: |
| 2.0 | 33.3% | 25.77 tok/s | 28.35 tok/s |
| 2.5 | 50.0% | 32.21 tok/s | 35.44 tok/s |
| 3.0 | 66.7% | 38.65 tok/s | 42.52 tok/s |
| 3.5 | 83.3% | 45.09 tok/s | 49.61 tok/s |
| **3.71** | **90.3%** | **47.80 tok/s** | **52.59 tok/s** |
| 4.0 | 100% | 51.53 tok/s | 56.70 tok/s |

The `3.71/4` row is a linear estimate for a request that actually sustains that
mean accepted length, not a measured coding benchmark. PLE locality, context,
concurrency, output distribution and warmup can change per-step time; use live
decode measurements when available.

## Local quality regression

This is a private deployment regression suite, not an official Qwen or NVIDIA
benchmark. Three temperature-zero runs scored:

| Run | Total | GSM8K | MMLU | Instruction | Tools | Retrieval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 147/162 | 47/48 | 83/96 | 7/8 | 4/4 | 6/6 |
| 2 | 146/162 | 47/48 | 82/96 | 7/8 | 4/4 | 6/6 |
| 3 | 148/162 | 46/48 | 85/96 | 7/8 | 4/4 | 6/6 |

The BF16-head parent scored 144, 146 and 148 in the same suite. Across the two
three-run sets, 141 cases always passed in both and 11 always failed in both.
No directional quality regression was observed, but these measurements do not
prove equivalence. Validate production prompts, long conversations, tool calls
and multimodal inputs before replacing a quality-first deployment.

## Provenance

- Base model: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- NVIDIA checkpoint: [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
- Parent hybrid: [`Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark`](https://huggingface.co/Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-Single-Spark)
- NVFP4 MTP donor: [`Inferact/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)
- Single-Spark runtime base: [`blazux/qwen3.8-Flash-DGX`](https://github.com/blazux/qwen3.8-Flash-DGX), pinned in the source repository

The FP8 `lm_head` was converted locally from the parent checkpoint without
fine-tuning. The source checkpoint was not modified in place. The conversion
reported a maximum absolute reconstruction error equal to 0.01929586 of the
original tensor's maximum absolute value.

## License and limitations

This derivative is distributed under the NVIDIA Open Model License and the
applicable Qwen Community License terms. The accompanying `NOTICE`,
`NVIDIA-OPEN-MODEL-LICENSE.pdf` and `QWEN-COMMUNITY-LICENSE.txt` files are part
of this repository. Review the licenses of the NVIDIA parent and Inferact donor
before redistribution or deployment.

This is an independent community integration, not an NVIDIA, Qwen, Inferact or
vLLM official release. Model outputs may be inaccurate, biased, unsafe or
unsuitable for a particular application. The runtime patch is experimental and
must be retested when vLLM, CUDA, drivers or the model implementation changes.
