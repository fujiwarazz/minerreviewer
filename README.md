# MinerReview

Training-free, conference-aware multi-agent peer review system with selective memory and leakage controls.

## Architecture

MinerReview is a **memory-driven reviewer system** that evolves from criteria-driven pipeline to a case-based reasoning system:

```
Paper -> PaperParser -> MultiChannelRetriever -> CriteriaPlanner
    -> ThemeAgents -> Arbiter -> Verifier -> ScoreConsistencyChecker
    -> Calibrator -> ExperienceDistiller -> MemoryEditor -> Final Review
```

### Key Components

- **PaperParser**: Extracts structured `PaperSignature` from papers
- **MultiChannelRetriever**: Retrieves similar paper cases, policy cards, critique cases, and failure cards
- **CriteriaPlanner**: Activates criteria from memory and mined sources
- **Verifier**: Checks score-text alignment, evidence support, and venue alignment
- **ScoreConsistencyChecker**: Provides consistency warnings (never modifies scores)
- **Calibrator**: Multi-way calibration (ordinal/three_way/binary)
- **ExperienceDistiller**: Extracts reusable experience from review traces
- **MemoryEditor**: Manages short-term and long-term memory admission

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Environment Variables

```bash
# LLM Configuration
export LLM_BACKEND="openai"
export LLM_MODEL="qwen-plus"
export LLM_BASE_URL="https://your-llm-endpoint/v1"
export OPENAI_API_KEY="your-api-key"

# Embedding Configuration
export EMBEDDING_MODEL="bge-embedding"
export EMBEDDING_BASE_URL="http://your-embedding-server:8001/v1"

# Vector Store (Milvus)
export MILVUS_HOST="your-milvus-host"
```

## Quickstart

### 1. Build Index

From OpenReview:
```bash
peerreviewer build_index --venue_id ICLR.cc/2024/Conference \
  --embedding_backend vllm --embedding_model bge-embedding \
  --embedding_base_url $EMBEDDING_BASE_URL
```

From local parquet files:
```bash
peerreviewer build_index --venue_id ICLR \
  --embedding_backend vllm --embedding_model bge-embedding \
  --embedding_base_url $EMBEDDING_BASE_URL \
  --parquet_paths ICLR_2017.parquet ICLR_2018.parquet \
  --vector_store_backend milvus --milvus_host $MILVUS_HOST
```

### 2. Build Paper Cases (Optional but Recommended)

```bash
peerreviewer build_cases --config configs/iclr.yaml
```

### 3. Run a Review

```bash
peerreviewer review_paper --config configs/iclr.yaml --paper_id <PAPER_ID>
```

Review a local parquet row:
```bash
peerreviewer review_paper --config configs/iclr.yaml \
  --parquet_path ICLR_2024.parquet --parquet_row 0 --target_year 2024
```

### 4. Evaluate

```bash
peerreviewer evaluate --config configs/iclr.yaml --target_year 2025
```

## Configuration

See `configs/iclr.example.yaml` for a complete configuration template.

Key configuration sections:
- `retrieval.use_case_memory`: Enable case-based retrieval
- `score_consistency`: Consistency check parameters (replaces old `decision_scoring`)
- `calibration.mode`: Calibration mode (ordinal/three_way/binary)
- `memory`: Memory management thresholds

## Output

The review output includes:
- `raw_rating`: Initial arbiter rating
- `decision_recommendation`: Initial decision
- `acceptance_likelihood`: Calibrated acceptance probability
- `verification`: Decision verification report
- `consistency`: Score consistency report
- `calibration`: Multi-way calibration results
- `trace`: Full audit trail

## Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

## License

MIT
