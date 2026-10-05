#!/usr/bin/env bash
set -euo pipefail

SP=$(python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent.parent)')
QWEN_DIR="$SP/vllm/models/qwen4_exp/nvidia"

# Pinned nightly contains upstream QSA FP8 KV and indexer workspace fixes.
python3 /patches/patch_qwen4_layer_type_nightly.py "$QWEN_DIR/model.py"

# MiaAI v0.30 file-backed FP8 PLE. A restarted container already contains the
# patched file, while a recreated container starts from the official image.
python3 /patches/patch_ple_mmap_nightly.py "$SP/vllm/models/qwen4_exp"

# Quantized target lm_head plus the matched 65K reduced MTP draft head.
python3 /patches/patch_qwen4_exp_fp8_heads.py "$SP"
python3 /patches/patch_vllm_v030_fp8_lm_head_scale.py \
  "$SP/vllm/model_executor/layers/quantization/fp8.py"

# Always restore the pristine helper from the read-only host mount before
# retargeting it, making docker restart and container recreation equivalent.
cp /patches/vllm_fp8_hybrid_modelopt.py "$SP/vllm_fp8_hybrid_modelopt.py"
python3 /patches/patch_hybrid_mixed_config.py \
  "$SP/vllm_fp8_hybrid_modelopt.py"
python3 /patches/patch_modelopt_hybrid_import.py \
  "$SP/vllm/model_executor/layers/quantization/modelopt.py"

if ! grep -q 'indexed_names_by_shard' \
  "$SP/vllm/model_executor/model_loader/weight_utils.py"; then
  python3 /patches/patch_safetensors_index_filter.py \
    "$SP/vllm/model_executor/model_loader/weight_utils.py"
fi

if ! grep -q '_pixion_prefill_metrics' "$SP/vllm/v1/metrics/loggers.py"; then
  python3 /patches/patch_prefill_metrics.py
fi

python3 -m py_compile \
  "$SP/vllm/models/qwen4_exp/common/ngram_embedding.py" \
  "$QWEN_DIR/qsa.py" \
  "$QWEN_DIR/ops/qsa.py" \
  "$QWEN_DIR/ngram_embedding.py" \
  "$QWEN_DIR/model.py" \
  "$QWEN_DIR/mtp.py" \
  "$SP/vllm/v1/worker/gpu/spec_decode/eagle/utils.py" \
  "$SP/vllm/v1/spec_decode/llm_base_proposer.py" \
  "$SP/vllm/model_executor/layers/quantization/fp8.py" \
  "$SP/vllm/model_executor/layers/quantization/modelopt.py" \
  "$SP/vllm/model_executor/model_loader/weight_utils.py" \
  "$SP/vllm/v1/metrics/loggers.py" \
  "$SP/vllm_fp8_hybrid_modelopt.py"

exec vllm serve "$@"
