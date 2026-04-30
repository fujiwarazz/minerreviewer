#!/usr/bin/env python
"""为 crosseval 数据集添加 primary_area 字段（支持双 token 并行 + retry）

使用 PaperParserV3 基于论文标题+摘要推断 primary_area
- 支持 2 个 API token 轮流使用
- retry=3，失败后记录到 failures.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.utils import read_yaml
from clients.llm_client import LLMClient, LLMConfig
from common.types import Paper
from pipeline.parse_paper_v3 import PaperParserV3, DOMAIN_TO_AREA_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 线程安全的 parser 池
_parser_pool = {}
_parser_lock = threading.Lock()
_failures = []
_failures_lock = threading.Lock()


def get_parser(token_id: int, base_url: str, model: str, api_key: str) -> PaperParserV3:
    """获取或创建 parser（线程安全）"""
    with _parser_lock:
        if token_id not in _parser_pool:
            llm_config = LLMConfig(
                backend="openai",
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=0.1,
            )
            llm = LLMClient(llm_config)
            _parser_pool[token_id] = PaperParserV3(llm)
        return _parser_pool[token_id]


def infer_with_retry(
    parser: PaperParserV3,
    paper_id: str,
    title: str,
    abstract: str,
    venue_id: str,
    year: int,
    max_retries: int = 3,
) -> dict:
    """带 retry 的推断"""
    paper = Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        venue_id=venue_id,
        year=year,
        authors=[],
        fulltext=None,
    )

    for attempt in range(max_retries):
        try:
            signature = parser.parse(paper)
            domain = signature.domain
            primary_area = DOMAIN_TO_AREA_MAP.get(domain, "other topics in machine learning (i.e., none of the above)")

            return {
                "paper_id": paper_id,
                "primary_area": primary_area,
                "domain": domain,
                "paper_type": signature.paper_type,
                "tasks": signature.tasks,
                "method_family": signature.method_family,
                "success": True,
                "attempts": attempt + 1,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 递增等待时间
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {paper_id} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                # 最终失败，记录到 failures
                logger.error(f"Failed after {max_retries} retries: {paper_id}")
                with _failures_lock:
                    _failures.append({
                        "paper_id": paper_id,
                        "title": title,
                        "abstract": abstract,
                        "venue_id": venue_id,
                        "year": year,
                        "error": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                return {
                    "paper_id": paper_id,
                    "primary_area": "other topics in machine learning (i.e., none of the above)",
                    "domain": "other",
                    "success": False,
                    "attempts": max_retries,
                    "error": str(e),
                }


def process_parquet(
    parquet_path: Path,
    base_url: str,
    model: str,
    api_keys: list[str],
    n_workers: int = 10,
    max_retries: int = 3,
    limit: int | None = None,
) -> pd.DataFrame:
    """处理单个 parquet 文件"""
    df = pd.read_parquet(parquet_path)

    if limit:
        df = df.head(limit)

    logger.info(f"Loaded {parquet_path.name}: {len(df)} papers")

    # 并行推断，轮流使用多个 token
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for idx, row in df.iterrows():
            # 轮流分配 token
            token_id = idx % len(api_keys)
            api_key = api_keys[token_id]
            parser = get_parser(token_id, base_url, model, api_key)

            future = executor.submit(
                infer_with_retry,
                parser,
                row["paper_id"],
                row["title"],
                row.get("abstract", ""),
                row.get("conf", ""),
                row.get("year", 2024),
                max_retries,
            )
            futures[future] = row["paper_id"]

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 100 == 0:
                logger.info(f"Progress: {completed}/{len(df)}")

    # 构建 DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("paper_id")

    # 合并
    df["primary_area"] = df["paper_id"].map(lambda x: results_df.loc[x, "primary_area"] if x in results_df.index else "other")
    df["domain"] = df["paper_id"].map(lambda x: results_df.loc[x, "domain"] if x in results_df.index else "other")

    return df


def main():
    parser = argparse.ArgumentParser(description="为 crosseval 数据集添加 primary_area（双 token + retry）")
    parser.add_argument("--input-dir", default="/mnt/data/zzh/datasets/crosseval/crosseval_std")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-url", default="https://one-api.bltcy.top/v1")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--api-key-1", default=None, help="第一个 API key")
    parser.add_argument("--api-key-2", default=None, help="第二个 API key")
    parser.add_argument("--n-workers", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--file-pattern", default="*.parquet")
    parser.add_argument("--failures-file", default="data/crosseval_failures.jsonl")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    failures_file = Path(args.failures_file)

    # 从环境变量或配置读取 API key
    import os
    if not args.api_key_1:
        config = read_yaml("configs/iclr.yaml")
        args.api_key_1 = os.environ.get("OPENAI_API_KEY") or config.get("llm", {}).get("api_key", "")

    if not args.api_key_2:
        args.api_key_2 = os.environ.get("OPENAI_API_KEY_2", "")

    # 构建 API keys 列表
    api_keys = [args.api_key_1]
    if args.api_key_2:
        api_keys.append(args.api_key_2)
        logger.info(f"使用 {len(api_keys)} 个 token 并行")
    else:
        logger.info(f"使用单 token")

    logger.info(f"配置: base_url={args.base_url}, model={args.model}, retries={args.max_retries}")

    # 处理文件
    parquet_files = sorted(input_dir.glob(args.file_pattern))
    # 过滤掉已处理的 _sw 文件
    parquet_files = [f for f in parquet_files if "_sw" not in f.name]
    logger.info(f"Found {len(parquet_files)} parquet files")

    total_papers = 0
    for parquet_path in parquet_files:
        # 快速统计论文数
        df_count = pd.read_parquet(parquet_path, columns=["paper_id"])
        total_papers += len(df_count)

    logger.info(f"Total papers to process: {total_papers}")
    logger.info(f"Estimated time: {total_papers * 1 / args.n_workers / 60:.0f} - {total_papers * 2 / args.n_workers / 60:.0f} minutes")

    start_time = time.time()

    for parquet_path in parquet_files:
        # 跳过已处理的文件（已存在 _sw 版本）
        output_name = parquet_path.stem + "_sw.parquet"
        output_path = output_dir / output_name
        if output_path.exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"Skipping: {parquet_path.name} (already processed)")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {parquet_path.name}")

        df = process_parquet(
            parquet_path,
            args.base_url,
            args.model,
            api_keys,
            args.n_workers,
            args.max_retries,
            args.limit,
        )

        # 统计
        logger.info(f"Primary area distribution:")
        area_counts = df["primary_area"].value_counts()
        for area, count in area_counts.head(10).items():
            logger.info(f"  {area[:50]}... : {count}")

        # 保存（添加 _sw 后缀）
        output_name = parquet_path.stem + "_sw.parquet"
        output_path = output_dir / output_name
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved to: {output_path}")

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Processed: {total_papers} papers")
    logger.info(f"Failures: {len(_failures)}")

    # 保存失败记录
    if _failures:
        failures_file.parent.mkdir(parents=True, exist_ok=True)
        with open(failures_file, "w") as f:
            for item in _failures:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Failures saved to: {failures_file}")
        logger.info(f"Run retry with: PYTHONPATH=src python scripts/retry_crosseval_failures.py --failures-file {failures_file}")


if __name__ == "__main__":
    main()