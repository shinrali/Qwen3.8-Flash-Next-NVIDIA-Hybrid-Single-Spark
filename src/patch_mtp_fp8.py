"""Enable NVIDIA's blockwise-FP8 MTP experts in the pinned vLLM preview.

NVIDIA numbers the loaded MTP layer from the target model's layer count while
ModelOpt metadata uses a draft-local index.  Its routed experts use 128x128
block FP8 (``FP8_PB_WO``), for which this preview has a linear implementation
but no RoutedExperts dispatch.  Add only those two missing compatibility paths.
"""
from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()

imports_old = "from fnmatch import fnmatch\nfrom typing import TYPE_CHECKING, Any, cast"
imports_new = "from fnmatch import fnmatch\nimport re\nfrom typing import TYPE_CHECKING, Any, cast"
assert source.count(imports_old) == 1, "Unexpected modelopt import block"
source = source.replace(imports_old, imports_new)

group_anchor = '''    @staticmethod
    def _quantized_layer_prefix_candidates(prefix: str) -> tuple[str, ...]:
'''
group_method = '''    def _quantized_layer_group_size(self, prefix: str) -> int | None:
        for candidate in self._quantized_layer_prefix_candidates(prefix):
            info = self.quantized_layers.get(candidate)
            if info is None:
                candidate_dot = candidate + "."
                for key, value in self.quantized_layers.items():
                    if key.startswith(candidate_dot):
                        info = value
                        break
            if info:
                try:
                    return int(info.get("group_size") or 0) or None
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _quantized_layer_prefix_candidates(prefix: str) -> tuple[str, ...]:
'''
assert source.count(group_anchor) == 1, "Unexpected prefix-candidate method"
source = source.replace(group_anchor, group_method)

candidate_old = '''        if prefix.endswith(".lm_head"):
            candidates.append("lm_head")

        if prefix.startswith("language_model.model."):
'''
candidate_new = '''        if prefix.endswith(".lm_head"):
            candidates.append("lm_head")

        match = re.match(r"^(.*?mtp\\.layers\\.)(\\d+)(\\..*)$", prefix)
        if match:
            layer_index = int(match.group(2))
            for local_index in range(0, min(layer_index, 8) + 1):
                if local_index != layer_index:
                    candidates.append(
                        f"{match.group(1)}{local_index}{match.group(3)}"
                    )

        if prefix.startswith("language_model.model."):
'''
assert source.count(candidate_old) == 1, "Unexpected candidate mapping block"
source = source.replace(candidate_old, candidate_new)

moe_old = '''            if quant_algo == "W4A16_NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.w4a16_nvfp4_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "MXFP8":
'''
moe_new = '''            if quant_algo == "W4A16_NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.w4a16_nvfp4_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo in ("FP8_BLOCK_SCALES", "FP8_BLOCK", "FP8_PB_WO"):
                from vllm.model_executor.layers.quantization.fp8 import (
                    Fp8Config as _Fp8Config,
                    Fp8MoEMethod as _Fp8MoEMethod,
                )

                group_size = self._quantized_layer_group_size(prefix) or 128
                return _Fp8MoEMethod(
                    _Fp8Config(
                        is_checkpoint_fp8_serialized=True,
                        activation_scheme="dynamic",
                        weight_block_size=[group_size, group_size],
                    ),
                    layer,
                )
            if quant_algo == "MXFP8":
'''
assert source.count(moe_old) == 1, "Unexpected RoutedExperts dispatch block"
source = source.replace(moe_old, moe_new)

path.write_text(source)
compile(path.read_text(), str(path), "exec")
print("Enabled NVIDIA FP8_PB_WO MTP RoutedExperts dispatch and layer mapping")
