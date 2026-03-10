# MinerReview

Training-free, conference-aware multi-agent peer review system with selective memory and leakage controls.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart

1) Build index from OpenReview:

```bash
peerreviewer build_index --venue_id ICLR.cc/2024/Conference --embedding_backend vllm --embedding_model bge-embedding --embedding_base_url http://10.20.49.150:8001/v1
```

Or build from local parquet files (ICLR 2017-2023 dumps):

```bash
peerreviewer build_index --venue_id ICLR --embedding_backend vllm --embedding_model bge-embedding \
  --embedding_base_url http://10.20.49.150:8001/v1 \
  --parquet_paths /Users/peelsannaw/Downloads/ICLR_2017.parquet /Users/peelsannaw/Downloads/ICLR_2018.parquet \
  --max_embed_chars 6000 \
  --vector_store_backend milvus --milvus_host 10.20.49.150 --milvus_port 19530 \
  --milvus_papers_collection papers_iclr --milvus_reviews_collection reviews_iclr
```

If you do not want review embeddings (only paper index), pass `--skip_review_index`.
If your embedding model has a strict context limit, use `--max_embed_chars` to truncate inputs.

2) Run a review (abstract-only by default):

```bash
peerreviewer review_paper --config configs/iclr.yaml --paper_id <OPENREVIEW_PAPER_ID>
```

To review a local parquet row without indexing it (e.g. 2024 test paper):

```bash
peerreviewer review_paper --config configs/iclr.yaml --parquet_path /mnt/data/zzh/datasets/crosseval/crosseval_std/ICLR_2024.parquet --parquet_row 0 --target_year 2024
```

3) Evaluate on a time split:

```bash
peerreviewer evaluate --config configs/iclr.yaml --target_year 2025
```

## Notes

- DashScope usage: set `DASHSCOPE_API_KEY`, and ensure `llm.backend` is `dashscope`.
- vLLM embeddings: set `embedding.backend` to `vllm` and `embedding.vllm_base_url` to your server, e.g. `http://10.20.49.150:8001/v1`. For CLI indexing, pass `--embedding_base_url`.
- Calibration artifacts and memory cards are saved to `data/processed/`.
- Output JSON includes trace metadata for auditability.

## Tests

```bash
pytest
```
# minerreviewer
