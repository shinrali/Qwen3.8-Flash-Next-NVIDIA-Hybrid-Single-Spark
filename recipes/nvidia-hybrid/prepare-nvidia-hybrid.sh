#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS="${MODEL_ROOT:?set MODEL_ROOT to the directory containing model folders}"
SRC="${SOURCE_MODEL_DIR:-$MODELS/Qwen3.8-Flash-Next-NVIDIA-NVFP4}"
DST="${FP8_SIDE_MODEL_DIR:-$MODELS/Qwen3.8-Flash-Next-NVIDIA-FP8-Hybrid}"
IMAGE="${CONVERTER_IMAGE:-nvidia-hybrid-single-spark:base}"

test -f "$SRC/model.safetensors.index.json"
if test -e "$DST"; then
  echo "Destination already exists: $DST" >&2
  exit 1
fi

tmp="$(mktemp -d "$MODELS/.nvidia-fp8-hybrid.preparing.XXXXXX")"
tmp_name="$(basename "$tmp")"
echo "Preparing isolated hard-linked checkpoint at $tmp"
cp -al "$SRC/." "$tmp/"
cp --remove-destination "$SRC/model.safetensors.index.json" \
  "$tmp/model.safetensors.index.json"

docker run --rm \
  --name qwen38-nvidia-fp8-convert \
  -v "$MODELS:/models" \
  -v "$REPO_ROOT/tools:/tools:ro" \
  --entrypoint python3 \
  "$IMAGE" \
  /tools/fp8_convert.py "/models/$tmp_name"

python3 -c 'import json,re,sys; w=json.load(open(sys.argv[1]))["weight_map"]; r=re.compile(r"model\.language_model\.layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.shared_expert\.(gate_proj|up_proj|down_proj))\.weight_scale_inv$"); n=sum(bool(r.search(k)) for k in w); print("converted_side_layer_scales",n); assert n == 300' \
  "$tmp/model.safetensors.index.json"

printf '%s\n' \
  'source=Qwen3.8-Flash-Next-NVIDIA-NVFP4' \
  'side_layers=blockwise-fp8-e4m3' \
  'mtp=NVIDIA-block-fp8' \
  > "$tmp/.prepared"
mv "$tmp" "$DST"
echo "Prepared: $DST"
