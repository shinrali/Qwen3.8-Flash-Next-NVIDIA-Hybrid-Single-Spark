#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Derived from MiaAI Lab's mmap PLE patch; see ../THIRD_PARTY_NOTICES.md.
"""Retarget the licensed MiaAI mmap patch to pinned 0cbac6cd common PLE.

Keep upstream prefetch, FP8 scale, hashing and ETP semantics unchanged.
Both modules are validated before either is written.
"""
import ast
from pathlib import Path
import runpy
import sys

legacy = runpy.run_path(str(Path(__file__).with_name('patch_ple_mmap_v030.py')))
edits = legacy['EDITS']
root = Path(sys.argv[1])
common = root / 'common/ngram_embedding.py'
nvidia = root / 'nvidia/ngram_embedding.py'
marker = '# Pixion mmap PLE for pinned nightly 0cbac6cd'

def replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Nightly PLE anchor expected once: {old[:140]!r}')
    return source.replace(old, new, 1)

cs, ns = common.read_text(), nvidia.read_text()
if marker in cs and marker in ns:
    print('Nightly common/NVIDIA mmap PLE already installed')
    sys.exit(0)
if marker in cs or marker in ns:
    raise RuntimeError('Incomplete two-module PLE patch')
for i in (0, 1, 2, 3, 4, 7, 8, 9):
    cs = replace(cs, *edits[i])
cs = replace(cs, 'from vllm.logger import init_logger\n',
    'from vllm.config import get_current_vllm_config\nfrom vllm.logger import init_logger\n')
cs = replace(cs,
    '        self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)\n'
    '        self._block_d = triton.next_power_of_2(self.embedding_dim)\n',
    '        if self._mmap_path is None:\n'
    '            self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)\n'
    '            self._lookup_device = self._uva_weight.device\n'
    '        else:\n'
    '            self._uva_weight = None\n'
    '            self._lookup_device = torch.device("cuda", torch.cuda.current_device())\n'
    '            logger.info("PLE %s: %s file-backed table %s (%.2f GiB), read over ATS",\n'
    '                        prefix, "reused" if self._mmap_ready else "building",\n'
    '                        self._mmap_path, self.weight.numel() * self.weight.element_size() / 2**30)\n'
    '        self._block_d = triton.next_power_of_2(self.embedding_dim)\n'
    '        self._row_bytes = self.embedding_dim * self.weight.element_size()\n'
    '        self._block_b = triton.next_power_of_2(self._row_bytes)\n')
if cs.count('device=self._uva_weight.device') != 2:
    raise RuntimeError('Nightly prefetch stream/buffer device anchors changed')
cs = cs.replace('device=self._uva_weight.device', 'device=self._lookup_device')
ns = replace(ns, *edits[10])
prepared = [(common, cs + '\n' + marker + '\n'), (nvidia, ns + '\n' + marker + '\n')]
for path, source in prepared:
    ast.parse(source)
for path, source in prepared:
    path.write_text(source)
    print(f'Nightly mmap PLE installed: {path}')
