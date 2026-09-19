# Local quality validation

The 162-case suite used here is a local regression harness, not an official
Qwen, NVIDIA, MMLU, GPQA, coding or community leaderboard. It is useful for
catching gross conversion regressions, tool-call breakage and long-context
retrieval failures; it cannot establish general model quality.

## FP8-side checkpoint with BF16 target head

Three repeated temperature-zero runs scored 144/162, 146/162 and 148/162
(mean 146). The paired NVIDIA BF16-side and RadixArk hybrid comparison profiles
each scored 147/162.

Pass/fail status was identical for 156/162 cases across the three uploaded
checkpoint runs: 143 always passed, 13 always failed and six MMLU boundary cases
varied. In the 148-point run, each 147-point comparison had five discordant
cases: three uploaded-checkpoint-only passes and two baseline-only passes. The
exact paired two-sided result was `p=1.0`.

Only 107/162 raw outputs were byte-identical across all three runs. Concurrent
MTP scheduling is therefore not strictly byte-deterministic at temperature zero.

## FP8 target-head derivative

Three local 162-case runs scored 147/162, 146/162 and 148/162 (mean 147). In the
paired comparison with three BF16-target-head runs, 141 cases always passed in
both profiles and 11 always failed in both. Remaining differences were boundary
cases rather than a directional regression.

## Tool and long-context checks

Both published checkpoint lanes achieved:

- tool calling: 4/4;
- long-context retrieval: 6/6 in each run.

These checks show that the conversions did not obviously break the tested tool
schema or retrieval prompts. They do not prove universal tool compatibility,
instruction following, safety, coding ability or equivalence to the parent.

## Reporting rule

Report the observed score ranges rather than selecting one favorable run. A
small local suite with variable boundary items does not support a claim that
either checkpoint is globally more accurate. Validate real application prompts
and long conversations before adopting a derivative as a quality-critical
default.
