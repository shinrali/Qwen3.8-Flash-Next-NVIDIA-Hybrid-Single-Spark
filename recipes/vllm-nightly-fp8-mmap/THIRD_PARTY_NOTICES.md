# Third-party notice

`patches/patch_ple_mmap_v030.py` is copied without functional changes from
[MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
at commit `7d0712dcb83cc58b467235e24d3a4429ed14518b`.

- Upstream file: `files/patch_ple_mmap_v030.py`
- SHA-256: `b63bb3d86ececfe5d0dac7f4c072407c9745fab7b6ed856175bc8f668c107309`
- Copyright: MiaAI Lab
- License: AGPL-3.0-or-later, as declared in the file's SPDX header

The recipe uses the official pinned nightly image at upstream commit
`0cbac6cd1305f710e12193596b27488397bcb205` and remains subject to vLLM's
license. Other project patches retain their existing source notices and
attribution.

`patch_ple_mmap_nightly.py` adapts the above AGPL-licensed helpers and edit
definitions to the new common/NVIDIA module split. It retains that attribution
and is distributed under AGPL-3.0-or-later for this derivative mmap component.
