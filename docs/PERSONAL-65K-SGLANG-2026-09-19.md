# Personally Optimized 65K FP8 MTP Pair on SGLang — 2026-09-19

## Scope

This repository publishes the generic SGLang v0.5.20 runtime patch and an
optional personally optimized 65,536-token FP8 MTP pair. The pair contains only
a token-ID array and the corresponding block-FP8 draft-head rows; it does not
contain source prompts, responses or conversation text.

The published files keep their original names for compatibility:

- `studio_draft_vocab_65536_all.npy`
- `mtp_draft_head_studio_65536_all_fp8.safetensors`

The `studio_` prefix is historical naming only. It does not indicate a required
application integration.

## Runtime boundary

`src/sglang/patch_sglang_draft65k.py` adds a separate 65,536-row block-FP8 head
to the NVIDIA Qwen4/Flash-Next MTP worker in SGLang v0.5.20. It keeps the target
`lm_head` at all 248,320 rows, clones the PyTorch module registries before
replacing draft parameters, and aborts startup if the draft copy aliases or
mutates the complete target module.

The vocabulary and reduced head are a matched pair. Row `i` of the reduced head
must correspond to the complete-head row identified by vocabulary entry `i`.

## One DGX Spark measurement

One fixed speed-test profile produced:

| Draft configuration | Prefill tok/s | Decode tok/s | MTP acceptance |
| --- | ---: | ---: | ---: |
| Complete vocabulary | 2604.22 | 37.31 | 46.94% |
| Personal 65K, first run | 2604.29 | 37.47 | 48.61% |
| Personal 65K, hot repeat | 2663.39 | 42.53 | 56.11% |

The hot repeat includes runtime warm-up effects and output-dependent draft
acceptance. It is evidence for this one profile, not a universal speed claim.
Users should compare multiple hot runs on their own output distribution.
