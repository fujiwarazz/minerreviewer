#!/usr/bin/env python
"""重试 crosseval 处理失败的论文

读取 failures.jsonl，重新推断 primary_area
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.utils import read_yaml
from clients.llm_client import LLMClient, LLMConfig
from common.types import Paper
from pipeline.parse_paper_v3 import PaperParserV3, DOMAIN_TO_AREA_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_failures(failures_file: Path) -> list[dict]:
    """加载失败记录"""
    failures = []
    with open(failures_file) as f:
        for line in f:
            failures.append(json.loads(line))
    return failures


def retry_paper(
    parser: PaperParserV3,
    paper: dict,
    max_retries: int = 5,
) -> dict | None:
    """重试单篇论文"""
    paper_obj = Paper(
        paper_id=paper["paper_id"],
        title=paper["title"],
        abstract=paper.get("abstract", ""),
        venue_id=paper.get("venue_id", ""),
        year=paper.get("year", 2024),
        authors=[],
        fulltext=None,
    )

    for attempt in range(max_retries):
        try:
            signature = parser.parse(paper_obj)
            domain = signature.domain
            primary_area = DOMAIN_TO_AREA_MAP.get(domain, "other topics in machine learning (i.e., none of the above)")

            logger.info(f"Success: {paper['paper_id']} -> {primary_area[:40]}...")
            return {
                "paper_id": paper["paper_id"],
                "primary_area": primary_area,
                "domain": domain,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {paper['paper_id']}: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Still failed: {paper['paper_id']}")
                return None


def main():
    parser = argparse.ArgumentParser(description="重试 crosseval 失败论文")
    parser.add_argument("--failures-file", default="data/crosseval_failures.jsonl")
    parser.add_argument("--base-url", default="https://one-api.bltcy.top/v1")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    failures_file = Path(args.failures_file)
    if not failures_file.exists():
        logger.error(f"Failures file not found: {failures_file}")
        return

    # 加载配置
    import os
    if not args.api_key:
        config = read_yaml("configs/iclr.yaml")
        args.api_key = os.environ.get("OPENAI_API_KEY") or config.get("llm", {}).get("api_key", "")

    # 初始化 parser
    llm_config = LLMConfig(
        backend="openai",
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0.1,
    )
    llm = LLMClient(llm_config)
    paper_parser = PaperParserV3(llm)

    # 加载失败记录
    failures = load_failures(failures_file)
    logger.info(f"Loaded {len(failures)} failed papers")

    # 重试
    success_count = 0
    still_failed = []

    for paper in failures:
        logger.info(f"\nRetrying: {paper['paper_id']}")
        result = retry_paper(paper_parser, paper, args.max_retries)

        if result:
            success_count += 1
            # 更新对应的 parquet 文件（如果需要）
            # TODO: 可选实现
        else:
            still_failed.append(paper)

    logger.info(f"\n{'='*60}")
    logger.info(f"Retry completed: {success_count}/{len(failures)} success")
    logger.info(f"Still failed: {len(still_failed)}")

    # 保存仍然失败的
    if still_failed:
        new_failures_file = failures_file.parent / f"{failures_file.stem}_remaining.jsonl"
        with open(new_failures_file, "w") as f:
            for item in still_failed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Remaining failures saved to: {new_failures_file}")


if __name__ == "__main__":
    main()