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

# Run a single test file
PYTHONPATH=src pytest tests/test_memory_store.py -v

# Run tests with coverage
PYTHONPATH=src pytest tests/ -v --cov=src --cov-report=term-missing
```

## Architecture

```
src/
├── main.py              # CLI entry point (peerreviewer command)
├── pipeline/            # Core review pipeline stages
│   ├── review_pipeline.py   # Main orchestrator
│   ├── parse_paper.py       # Extract PaperSignature from papers
│   ├── retrieve.py          # Multi-channel retrieval (cases + papers + policies)
│   ├── mine_criteria.py     # Extract criteria from historical data
│   ├── plan_criteria.py     # Activate criteria from memory
│   ├── distill_criteria.py  # Deduplicate and refine criteria
│   ├── rewrite_criteria.py  # Format criteria for agents
│   ├── aggregate.py         # Arbiter agent
│   ├── verify_decision.py   # Check score-text alignment
│   ├── check_score_consistency.py  # Consistency warnings (never modifies scores)
│   ├── calibrate.py         # Rating calibration (isotonic regression)
│   ├── distill_experience.py    # Extract reusable experience from traces
│   └── memory_editor.py     # Manage short/long-term memory admission
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

1. **ParsePaper**: Extract structured `PaperSignature` from paper (paper_type, tasks, domain, method_family)
2. **Retrieve**: Multi-channel retrieval - find similar PaperCases + policy/critique/failure cards + supporting papers
3. **Mine Criteria**: Extract content criteria from similar papers, policy criteria from accept/reject reviews
4. **Plan Criteria**: Activate relevant criteria from memory based on paper signature
5. **Distill/Rewrite Criteria**: Deduplicate and format criteria for theme agents
6. **Theme Agents**: Each agent reviews one theme, outputs strengths/weaknesses with severity tags
7. **Arbiter**: Aggregates outputs, learns from similar cases' patterns, produces rating with rationale
8. **Verify**: Check score-text alignment, evidence support, venue alignment
9. **ScoreConsistencyChecker**: Compare rating against similar cases, warn if deviating (never modifies scores)
10. **Calibrate**: Apply isotonic regression for acceptance likelihood based on historical data
11. **DistillExperience**: Extract reusable experience cards from review trace
12. **MemoryEditor**: Decide admission to short-term vs long-term memory

## Review Output Fields

Each review produces an `ArbiterOutput` with:
- `raw_rating`: Initial arbiter rating (1-10 scale)
- `decision_recommendation`: Accept/Reject/Borderline
- `acceptance_likelihood`: Calibrated probability from historical data
- `key_decisive_issues`: Issues that determined the decision
- `decision_rationale` / `score_rationale`: Explainability fields
- `verification`: `DecisionVerificationReport` (score-text alignment, evidence support)
- `consistency`: `ScoreConsistencyReport` (comparison with similar cases)
- `calibration`: `CalibrationResult` (multi-way likelihood breakdown)
- `trace`: Full audit trail for debugging

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
- `OPENAI_API_KEY` or `DASHSCOPE_API_KEY` (choose based on LLM backend)
- `LLM_BACKEND`: "openai" or "dashscope"
- `LLM_MODEL`: e.g., "qwen-plus"
- `LLM_BASE_URL`: LLM endpoint URL
- `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`
- `MILVUS_HOST` (for vector store)

Config files use `${VAR:default}` syntax for environment variable substitution. See `configs/iclr.example.yaml` for full template.