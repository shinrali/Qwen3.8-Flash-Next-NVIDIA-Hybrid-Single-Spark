# Pinned nightly: FP8 KV + SSD FP8 PLE on one DGX Spark

This profile uses an official image plus read-only mounted startup patches;
no custom image is built. The older `../vllm-v030-mmap` BF16-KV profile remains
unchanged for rollback. Do not use these nightly patches with v0.30.0.

```bash
cp .env.example .env
# Set MODEL_DIR, CACHE_DIR and VLLM_API_KEY; never commit .env.
docker compose -f compose.example.yaml config -q
docker compose -f compose.example.yaml up -d
docker compose -f compose.example.yaml logs -f
```

The example uses the public matched `draft_vocab_65536.npy` and
`mtp_draft_head_65536_fp8.safetensors`. Reported measurements used a personally
optimized matched pair; identical shapes do not imply identical acceptance.
PLE remains file-backed but may consume reclaimable Linux page cache. Keep
CACHE_DIR persistent; it is separate from MODEL_DIR and this patch directory.

Main KV is FP8 E4M3 with 9 GiB, native max length 262144, MTP3, four admitted
sequences sharing a measured 542103-token pool (2.07 full-length slots).
Recurrent state remains BF16. No eager mode, no old QSA FP8 backport.

Full image pin, patch inventory, per-run measurements, accuracy boundaries and
methodology: [nightly benchmark report](../../docs/NIGHTLY-FP8-KV-2026-10-05.md).
Third-party source and license: [notices](THIRD_PARTY_NOTICES.md).

Stop only this service: `docker compose -f compose.example.yaml stop`.
The example publishes port 8000 and requires VLLM_API_KEY; restrict access to
trusted networks. Do not start it alongside an existing large-model service
without first checking memory and active requests.
