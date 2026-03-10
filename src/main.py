from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path

import numpy as np

from clients.openreview_client import OpenReviewClient
from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from common.logging import setup_logging
from common.types import Paper
from common.utils import write_json
from pipeline.review_pipeline import ReviewPipeline
from storage.doc_store import DocStore
from storage.faiss_index import FaissIndex
from storage.milvus_store import MilvusConfig, MilvusStore
from storage.parquet_loader import load_parquet_files, load_parquet_paper, load_parquet_ground_truth
from eval.run_eval import run_evaluation
from eval.coverage import evaluate_coverage

logger = logging.getLogger(__name__)


def build_index(
    venue_id: str,
    embedding_backend: str,
    embedding_model: str,
    embedding_base_url: str | None,
    parquet_paths: list[str] | None,
    skip_review_index: bool,
    max_embed_chars: int,
    vector_store_backend: str,
    milvus_host: str | None,
    milvus_port: int,
    milvus_papers_collection: str | None,
    milvus_reviews_collection: str | None,
) -> None:
    doc_store = DocStore()

    if parquet_paths:
        papers, reviews = load_parquet_files(parquet_paths, venue_id=venue_id)
        policy = None
    else:
        client = OpenReviewClient()
        papers = client.fetch_submissions(venue_id)
        reviews = client.fetch_reviews(venue_id)
        policy = client.fetch_policy(venue_id)

    doc_store.save_papers(venue_id, papers)
    doc_store.save_reviews(venue_id, reviews)
    doc_store.save_policy(venue_id, policy)

    paper_texts = [_truncate_text(f"{paper.title}\n{paper.abstract}", max_embed_chars) for paper in papers]
    embedding_kwargs = {"backend": embedding_backend, "model": embedding_model}
    if embedding_base_url:
        if embedding_backend == "vllm":
            embedding_kwargs["vllm_base_url"] = embedding_base_url
        else:
            embedding_kwargs["base_url"] = embedding_base_url
    embedding_client = EmbeddingClient(EmbeddingConfig(**embedding_kwargs))
    paper_embeddings = embedding_client.embed(paper_texts)
    if vector_store_backend == "milvus":
        if not milvus_host:
            raise ValueError("milvus_host is required when vector_store_backend=milvus")
        milvus_cfg = MilvusConfig(
            host=milvus_host,
            port=milvus_port,
            papers_collection=milvus_papers_collection or f"papers_{venue_id.replace('/', '_')}",
            reviews_collection=milvus_reviews_collection or f"reviews_{venue_id.replace('/', '_')}",
        )
        milvus = MilvusStore(milvus_cfg)
        milvus.upsert_embeddings(milvus_cfg.papers_collection, [paper.paper_id for paper in papers], paper_embeddings.tolist())
    else:
        paper_index = FaissIndex(
            f"data/index/papers__{venue_id.replace('/', '_')}.faiss",
            f"data/index/papers__{venue_id.replace('/', '_')}.meta.json",
        )
        paper_index.build(np.array(paper_embeddings), [paper.paper_id for paper in papers])
        paper_index.save()

    if not skip_review_index and reviews:
        review_texts = [_truncate_text(review.text, max_embed_chars) for review in reviews]
        review_embeddings = embedding_client.embed(review_texts)
        if vector_store_backend == "milvus":
            if not milvus_host:
                raise ValueError("milvus_host is required when vector_store_backend=milvus")
            milvus_cfg = MilvusConfig(
                host=milvus_host,
                port=milvus_port,
                papers_collection=milvus_papers_collection or f"papers_{venue_id.replace('/', '_')}",
                reviews_collection=milvus_reviews_collection or f"reviews_{venue_id.replace('/', '_')}",
            )
            milvus = MilvusStore(milvus_cfg)
            milvus.upsert_embeddings(milvus_cfg.reviews_collection, [review.review_id for review in reviews], review_embeddings.tolist())
        else:
            review_index = FaissIndex(
                f"data/index/reviews__{venue_id.replace('/', '_')}.faiss",
                f"data/index/reviews__{venue_id.replace('/', '_')}.meta.json",
            )
            review_index.build(np.array(review_embeddings), [review.review_id for review in reviews])
            review_index.save()

    logger.info("Index built for %s", venue_id)


def review_paper(
    config_path: str,
    paper_id: str | None,
    target_year: int | None,
    parquet_path: str | None,
    parquet_row: int,
) -> None:
    pipeline = ReviewPipeline(config_path)
    if parquet_path:
        target = load_parquet_paper(parquet_path, parquet_row, venue_id=pipeline.venue_id)
        result = pipeline._run_review(target, target_year or target.year)
        ground_truth = load_parquet_ground_truth(parquet_path, parquet_row)
        result.trace["ground_truth_reviews"] = ground_truth
        coverage_cfg = pipeline.config.get("coverage_eval", {"enabled": False})
        if coverage_cfg.get("enabled", False):
            result.trace["coverage_eval"] = evaluate_coverage(
                result.strengths,
                result.weaknesses,
                ground_truth,
                pipeline.llm,
                pipeline.embedding_client,
                coverage_cfg,
            )
            _store_coverage_gaps(pipeline, target, result.trace.get("coverage_eval", {}), coverage_cfg)
    else:
        if paper_id is None:
            raise ValueError("paper_id is required when parquet_path is not provided")
        result = pipeline.review_paper(paper_id, target_year)
    output = result.model_dump()
    write_json("data/processed/last_review.json", output)
    print(json.dumps(output, indent=2, ensure_ascii=True))


def evaluate(config_path: str, target_year: int) -> None:
    metrics = run_evaluation(config_path, target_year)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Conference-aware multi-agent peer review system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build_index", help="Fetch OpenReview data and build FAISS indexes")
    build_parser.add_argument("--venue_id", required=True)
    build_parser.add_argument("--embedding_backend", default="sentence-transformers")
    build_parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    build_parser.add_argument("--embedding_base_url", default=None)
    build_parser.add_argument("--parquet_paths", nargs="*", default=None)
    build_parser.add_argument("--skip_review_index", action="store_true")
    build_parser.add_argument("--max_embed_chars", type=int, default=6000)
    build_parser.add_argument("--vector_store_backend", default="faiss")
    build_parser.add_argument("--milvus_host", default=None)
    build_parser.add_argument("--milvus_port", type=int, default=19530)
    build_parser.add_argument("--milvus_papers_collection", default=None)
    build_parser.add_argument("--milvus_reviews_collection", default=None)

    review_parser = subparsers.add_parser("review_paper", help="Review a paper by ID")
    review_parser.add_argument("--config", default="configs/iclr.yaml")
    review_parser.add_argument("--paper_id")
    review_parser.add_argument("--target_year", type=int)
    review_parser.add_argument("--parquet_path")
    review_parser.add_argument("--parquet_row", type=int, default=0)

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation on a year split")
    eval_parser.add_argument("--config", default="configs/iclr.yaml")
    eval_parser.add_argument("--target_year", type=int, required=True)

    args = parser.parse_args()
    if args.command == "build_index":
        build_index(
            args.venue_id,
            args.embedding_backend,
            args.embedding_model,
            args.embedding_base_url,
            args.parquet_paths,
            args.skip_review_index,
            args.max_embed_chars,
            args.vector_store_backend,
            args.milvus_host,
            args.milvus_port,
            args.milvus_papers_collection,
            args.milvus_reviews_collection,
        )
    elif args.command == "review_paper":
        review_paper(args.config, args.paper_id, args.target_year, args.parquet_path, args.parquet_row)
    elif args.command == "evaluate":
        evaluate(args.config, args.target_year)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _store_coverage_gaps(pipeline: ReviewPipeline, target: Paper, coverage_eval: dict, coverage_cfg: dict) -> None:
    if not coverage_cfg.get("store_gaps", False):
        return
    vector_store = pipeline.config.get("vector_store", {})
    if vector_store.get("backend") != "milvus":
        return
    gaps: list[str] = []
    for aspect in ("strengths", "weaknesses"):
        aspect_eval = coverage_eval.get(aspect, {})
        gaps.extend(aspect_eval.get("unmatched_points", []) or [])
    if not gaps:
        return
    context_prefix = f"{target.title}\n{target.abstract}"
    contexts = [f"{context_prefix}\n{gap}" for gap in gaps]
    embeddings = pipeline.embedding_client.embed(contexts)
    collection = vector_store.get("coverage_gaps_collection", "coverage_gaps_iclr")
    milvus_cfg = MilvusConfig(
        host=vector_store.get("host", "localhost"),
        port=int(vector_store.get("port", 19530)),
        papers_collection=vector_store.get("papers_collection", "papers_iclr"),
        reviews_collection=vector_store.get("reviews_collection", "reviews_iclr"),
    )
    milvus = MilvusStore(milvus_cfg)
    ids = [str(uuid.uuid4()) for _ in gaps]
    milvus.upsert_embeddings(collection, ids, embeddings.tolist(), texts=gaps)
