#!/usr/bin/env python3
"""Make the default safetensors iterator honor weight_map per shard.

vLLM normally uses the index to choose shard files, then yields every tensor
physically present in each selected shard. Selective overlay checkpoints can
legitimately leave superseded tensors in hard-linked shards, so the iterator
must also enforce the index's name-to-shard mapping.
"""

from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()

setup_anchor = "    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)\n"
setup_replacement = setup_anchor + """

    indexed_names_by_shard: dict[str, set[str]] | None = None
    if sorted_files:
        index_path = Path(sorted_files[0]).parent / SAFE_WEIGHTS_INDEX_NAME
        if index_path.is_file():
            try:
                weight_map = json.loads(index_path.read_text()).get("weight_map")
                if isinstance(weight_map, dict):
                    indexed_names_by_shard = defaultdict(set)
                    for tensor_name, shard_name in weight_map.items():
                        indexed_names_by_shard[Path(shard_name).name].add(tensor_name)
                    logger.info_once(
                        "Safetensors index tensor filter enabled for %d shards",
                        len(indexed_names_by_shard),
                    )
            except Exception as exc:
                logger.warning_once("Cannot read safetensors tensor index: %s", exc)
"""
if source.count(setup_anchor) != 1:
    raise SystemExit("safetensors sorted-files anchor not found exactly once")
source = source.replace(setup_anchor, setup_replacement)

loop_anchor = """    for st_file in tqdm(
        sorted_files,
        desc=loading_desc,
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
"""
loop_replacement = loop_anchor + """        allowed_names = (
            indexed_names_by_shard.get(Path(st_file).name, set())
            if indexed_names_by_shard is not None
            else None
        )
"""
if source.count(loop_anchor) != 1:
    raise SystemExit("safetensors shard-loop anchor not found exactly once")
source = source.replace(loop_anchor, loop_replacement)

skip_anchor = """                    if should_skip_weight(name, local_expert_ids):
                        continue
"""
skip_replacement = """                    if (
                        (allowed_names is not None and name not in allowed_names)
                        or should_skip_weight(name, local_expert_ids)
                    ):
                        continue
"""
if source.count(skip_anchor) != 2:
    raise SystemExit("safetensors per-name skip anchor not found exactly twice")
source = source.replace(skip_anchor, skip_replacement)

eager_anchor = """                if not should_skip_weight(name, local_expert_ids):
                    yield name, param
"""
eager_replacement = """                if (
                    (allowed_names is None or name in allowed_names)
                    and not should_skip_weight(name, local_expert_ids)
                ):
                    yield name, param
"""
if source.count(eager_anchor) != 1:
    raise SystemExit("safetensors eager skip anchor not found exactly once")
source = source.replace(eager_anchor, eager_replacement)

path.write_text(source)
