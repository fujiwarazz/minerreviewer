# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MinerReview is a training-free, conference-aware multi-agent peer review system with selective memory and leakage controls. It generates paper reviews by mining criteria from historical papers/reviews and using multi-agent deliberation.

**Current Architecture**: Memory-driven reviewer system using case-based reasoning. Papers are reviewed by retrieving similar paper cases from memory, learning from their strengths/weaknesses patterns, and calibrating ratings based on historical decisions.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate && pip install -e .

# Build index from OpenReview or local parquet files
peerreviewer build_index --venue_id ICLR --embedding_backend vllm --embedding_model bge-embedding --embedding_base_url <URL>

# Build paper cases memory (from reviewer2 dataset)
python scripts/build_reviewer2_memory.py --config configs/iclr.yaml

# Build full memory (cases + policy/critique/failure cards)
python scripts/build_full_memory.py --config configs/iclr.yaml

# Review a paper
peerreviewer review_paper --config configs/iclr.yaml --paper_id <ID>
peerreviewer review_paper --config configs/iclr.yaml --parquet_path <PATH> --parquet_row 0 --target_year 2024

# Run evaluation
peerreviewer evaluate --config configs/iclr.yaml --target_year 2025

# Cross-venue evaluation (test generalization)
python scripts/cross_venue_test.py --config configs/iclr.yaml --test_venue NeurIPS --test_year 2024

# Run tests
PYTHONPATH=src pytest tests/ -v
```

## Architecture

```
src/
├── main.py              # CLI entry point
├── pipeline/            # Core review pipeline stages
│   ├── review_pipeline.py   # Main orchestrator
│   ├── retrieve.py          # Multi-channel retrieval (cases + papers + policies)
│   ├── mine_criteria.py     # Extract criteria from historical data
│   ├── plan_criteria.py     # Activate criteria from memory
│   ├── aggregate.py         # Arbiter agent
│   └── calibrate.py         # Rating calibration
├── agents/              # Multi-agent system
│   ├── theme_agent.py       # Per-theme review generation
│   └── arbiter_agent.py     # Final rating/decision synthesis with case learning
├── clients/             # External service clients
│   ├── llm_client.py        # Supports openai, dashscope backends
│   └── embedding_client.py  # Supports sentence-transformers, vllm
├── storage/             # Data persistence
│   ├── case_store.py        # PaperCase storage with hybrid retrieval
│   ├── memory_store.py      # ExperienceCards (policy/critique/failure)
│   ├── doc_store.py         # JSON-based paper/review storage
│   └── milvus_store.py      # Remote Milvus vector store
└── common/
    └── types.py             # Pydantic models
```

## Key Data Types

- **PaperCase**: Historical paper review record with `top_strengths`, `top_weaknesses`, `decisive_issues`, `rating`, `decision`. Used for rating calibration.
- **ExperienceCard**: Reusable review experience with kinds: `policy` (standards), `critique` (common criticisms), `failure` (rejection patterns)
- **PaperSignature**: Structured paper features for similarity matching (paper_type, tasks, domain, method_family)
- **RetrievalBundle**: Multi-channel retrieval result containing similar cases, policy cards, and supporting papers

## Review Pipeline Flow

1. **Retrieve**: Multi-channel retrieval - find similar PaperCases (for rating calibration) + policy cards + supporting papers
2. **Mine Criteria**: Extract content criteria from similar papers, policy criteria from accept/reject reviews
3. **Theme Agents**: Each agent reviews one theme, outputs strengths/weaknesses
4. **Arbiter**: Aggregates outputs, learns from similar cases' patterns, produces calibrated rating
5. **Verify**: Check score-text alignment and decision consistency
6. **Calibrate**: Apply isotonic regression for acceptance likelihood

## Memory System

Two memory stores work together:
- **CaseStore** (`data/processed/cases.jsonl`): PaperCase records with S/W for learning review patterns
- **MemoryStore** (`data/processed/memory_store.json`): ExperienceCards (policy/critique/failure)

Key principle: Arbiter learns from similar cases' strengths/weaknesses patterns to calibrate ratings. Cases with similar embeddings and signatures provide rating anchors.

## Configuration

Config files use YAML with environment variable substitution (`configs/iclr.yaml`). Key sections:
- `retrieval.use_case_memory`: Enable case-based retrieval
- `retrieval.case_embedding_weight/signature_weight`: Hybrid retrieval weights
- `score_consistency`: Consistency check parameters
- `calibration.mode`: ordinal/three_way/binary
- `memory`: Memory store paths and thresholds

## Environment Variables

Required for LLM/embedding services:
- `OPENAI_API_KEY` or `DASHSCOPE_API_KEY`
- `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`
- `MILVUS_HOST` (for vector store)