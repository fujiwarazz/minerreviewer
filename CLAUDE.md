# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MinerReview is a training-free, conference-aware multi-agent peer review system with selective memory and leakage controls. It generates paper reviews by mining criteria from historical papers/reviews and using multi-agent deliberation.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate && pip install -e .

# Build index from OpenReview or local parquet files
peerreviewer build_index --venue_id ICLR --embedding_backend vllm --embedding_model bge-embedding --embedding_base_url <URL>

# Review a paper
peerreviewer review_paper --config configs/iclr.yaml --paper_id <ID>
peerreviewer review_paper --config configs/iclr.yaml --parquet_path <PATH> --parquet_row 0 --target_year 2024

# Run evaluation
peerreviewer evaluate --config configs/iclr.yaml --target_year 2025

# Run tests
pytest
```

## Architecture

The system follows a pipeline architecture with multi-agent deliberation:

```
src/
├── main.py              # CLI entry point (build_index, review_paper, evaluate)
├── pipeline/            # Core review pipeline stages
│   ├── review_pipeline.py   # Orchestrates the full review flow
│   ├── retrieve.py          # Vector similarity search for papers/reviews
│   ├── mine_criteria.py     # Extract review criteria from historical data
│   ├── distill_criteria.py  # Dedup and select criteria
│   ├── rewrite_criteria.py  # Rewrite criteria for target paper context
│   ├── aggregate.py         # Aggregate theme agent outputs
│   ├── calibrate.py         # Rating calibration (isotonic regression)
│   └── update_memory.py     # Update experience cards
├── agents/              # Multi-agent system
│   ├── base.py              # AgentConfig dataclass
│   ├── theme_agent.py       # Per-theme review generation
│   └── arbiter_agent.py     # Final rating/decision synthesis
├── clients/             # External service clients
│   ├── llm_client.py        # Supports openai, dashscope backends
│   ├── embedding_client.py  # Supports sentence-transformers, vllm
│   └── openreview_client.py # OpenReview API
├── storage/             # Data persistence
│   ├── doc_store.py         # JSON-based paper/review storage
│   ├── faiss_index.py       # Local FAISS vector index
│   ├── milvus_store.py      # Remote Milvus vector store
│   ├── memory_store.py      # Experience cards (selective memory)
│   └── parquet_loader.py    # Load parquet datasets
├── eval/                # Evaluation metrics
│   ├── metrics.py           # Rating/decision accuracy
│   └── coverage.py          # Coverage evaluation
└── common/
    └── types.py             # Pydantic models (Paper, Review, Criterion, etc.)
```

## Review Pipeline Flow

1. **Retrieve**: Find similar papers/reviews from vector index (FAISS or Milvus)
2. **Mine Criteria**: Extract content criteria (from similar papers) and policy criteria (from accept/reject reviews)
3. **Distill**: Deduplicate and select top criteria per theme
4. **Rewrite**: Adapt criteria to target paper context
5. **Theme Agents**: Each agent reviews one theme, outputs strengths/weaknesses
6. **Arbiter**: Aggregates theme outputs, produces final rating and decision
7. **Calibrate**: Apply isotonic regression to predict acceptance likelihood
8. **Memory Update**: Update experience cards based on review outcome

## Configuration

Config files are YAML (e.g., `configs/iclr.yaml`). Key sections:
- `retrieval`: top_k for papers/reviews, similarity threshold
- `vector_store`: backend (faiss/milvus), connection settings
- `embedding`: backend (vllm/sentence-transformers), model, base_url
- `llm`: backend (openai/dashscope), model, temperature
- `distill`: criteria selection params (max_total, dedup_threshold)
- `memory`: experience card management thresholds

## Environment Variables

- `DASHSCOPE_API_KEY`: Required when `llm.backend=dashscope`
- `OPENAI_API_KEY`: Required when `llm.backend=openai`