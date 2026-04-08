#!/usr/bin/env python3
"""构建 DeepReview-13K 数据集的 PaperCase 记忆

将 DeepReview-13K 数据集转换为 PaperCase 格式，作为热插拔记忆源
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from common.utils import read_yaml
from storage.deepreview_store import DeepReviewCaseStore
from storage.milvus_store import MilvusConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build DeepReview-13K PaperCase memory")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/iclr.yaml",
        help="配置文件路径（用于获取 embedding 配置）",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/mnt/data/zzh/datasets/DeepReview-13K/data/stru/dataset_with_area.jsonl",
        help="DeepReview-13K JSONL 输入路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/deepreview_cases.jsonl",
        help="输出的 PaperCase JSONL 路径",
    )
    parser.add_argument(
        "--venue_id",
        type=str,
        default="DeepReview",
        help="venue 标识",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理数量（用于测试）",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="跳过 embedding 生成",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    config = read_yaml(args.config) if Path(args.config).exists() else {}

    # 创建 embedding client
    embedding_client = None
    if not args.skip_embedding:
        embedding_cfg = config.get("embedding", {})
        embedding_client = EmbeddingClient(
            EmbeddingConfig(
                backend=embedding_cfg.get("backend", "sentence-transformers"),
                model=embedding_cfg.get("model", "all-MiniLM-L6-v2"),
                base_url=embedding_cfg.get("base_url", ""),
                vllm_base_url=embedding_cfg.get("vllm_base_url", "http://10.20.49.150:8001/v1"),
            )
        )

    # 创建 Milvus config（从 vector_store 配置读取）
    milvus_config = None
    vector_cfg = config.get("vector_store", {})
    if vector_cfg.get("backend") == "milvus":
        milvus_config = MilvusConfig(
            host=vector_cfg.get("host", os.getenv("MILVUS_HOST", "localhost")),
            port=int(vector_cfg.get("port", 19530)),
            papers_collection="",  # DeepReview 用独立 collection
            reviews_collection="",
        )

    # 构建记忆
    logger.info("Building DeepReview-13K PaperCase memory...")
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)

    store = DeepReviewCaseStore.from_deepreview_jsonl(
        jsonl_path=args.input,
        output_path=args.output,
        embedding_client=embedding_client,
        milvus_config=milvus_config,
        venue_id=args.venue_id,
        limit=args.limit,
    )

    # 打印统计信息
    logger.info("Built %d cases", len(store.cases))
    logger.info("Primary areas: %d unique", len(store.list_areas()))
    for area in store.list_areas()[:10]:
        logger.info("  %s: %d cases", area, len(store.get_cases_by_area(area)))


if __name__ == "__main__":
    main()