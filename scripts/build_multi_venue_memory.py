#!/usr/bin/env python
"""
构建多会议记忆

从 parquet 文件加载历史论文，构建 PaperCase 并插入记忆。
每个会议只保留最新年份用于测试，其余全部插入记忆。

Usage:
    python scripts/build_multi_venue_memory.py --config configs/iclr.yaml --dry-run
    python scripts/build_multi_venue_memory.py --config configs/iclr.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from clients.llm_client import LLMClient, LLMConfig
from common.types import Paper, PaperCase, Review
from common.utils import read_yaml
from storage.case_store import CaseStore
from storage.memory_store import MemoryStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据路径配置
DATA_ROOT = Path("/mnt/data/zzh/datasets/crosseval/crosseval_std")

VENUE_CONFIG = {
    "ICLR": {
        "test_year": 2024,  # 最新年份用于测试
    },
    "NeurIPS": {
        "test_year": 2024,
    },
}


def load_parquet_venue(venue_id: str, include_test: bool = False) -> list[dict[str, Any]]:
    """加载某个会议的所有 parquet 数据"""
    test_year = VENUE_CONFIG.get(venue_id, {}).get("test_year", 2024)

    all_data = []

    # 加载 DATA_ROOT 目录下的 parquet 文件
    for f in sorted(DATA_ROOT.glob(f"{venue_id}_*.parquet")):
        year = int(f.stem.split("_")[-1])
        if not include_test and year >= test_year:
            logger.info(f"  Skipping {venue_id} {year} (test year)")
            continue

        df = pd.read_parquet(f)
        df["_source_file"] = str(f)
        all_data.append(df)
        logger.info(f"  Loaded {venue_id} {year}: {len(df)} papers")

    if all_data:
        return pd.concat(all_data, ignore_index=True).to_dict('records')
    return []


def parse_strengths_weaknesses(text: str) -> tuple[list[str], list[str]]:
    """解析 strengths 和 weaknesses，支持多种格式

    支持格式:
    - ### Strength / ### Weakness
    - Strengths: / Weaknesses:
    - Strength: / Weakness:
    """
    strengths = []
    weaknesses = []

    if not text:
        return strengths, weaknesses

    # 使用简单可靠的方法：找到关键词位置，然后确定段落边界
    def find_section_start(keyword: str) -> int:
        """找到关键词所在行的起始位置"""
        match = re.search(keyword, text, re.IGNORECASE)
        if not match:
            return -1
        # 从匹配位置向前找行首
        pos = match.start()
        line_start = text.rfind('\n', 0, pos) + 1
        return line_start

    def find_section_end(start_pos: int, next_keywords: list[str]) -> int:
        """找到段落的结束位置（下一个关键词之前）"""
        min_pos = len(text)
        for kw in next_keywords:
            match = re.search(kw, text[start_pos:], re.IGNORECASE)
            if match:
                abs_pos = start_pos + match.start()
                # 找到这一行的起始位置
                line_start = text.rfind('\n', 0, abs_pos) + 1
                min_pos = min(min_pos, line_start)
        return min_pos

    # 找到各段落的位置
    strength_start = find_section_start(r'strength(?:s)?\b')
    weakness_start = find_section_start(r'weakness(?:es)?\b')

    # 提取 strengths 段落
    if strength_start >= 0:
        # 结束位置：weakness 段落之前，或者文档结尾
        end_pos = weakness_start if weakness_start > strength_start else len(text)
        strength_section = text[strength_start:end_pos]
        strengths = _extract_list_items(strength_section)

    # 提取 weaknesses 段落
    if weakness_start >= 0:
        # 结束位置：questions/summary/correctness 之前，或者文档结尾
        end_pos = find_section_end(weakness_start, [r'questions?(?:\s|$)', r'summary(?:\s|$)', r'correctness(?:\s|$)'])
        weakness_section = text[weakness_start:end_pos]
        weaknesses = _extract_list_items(weakness_section)

    return strengths, weaknesses


def _extract_list_items(text: str) -> list[str]:
    """从文本中提取列表项"""
    items = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # 匹配 bullet 格式
        if line.startswith('-') or line.startswith('*') or line.startswith('•'):
            cleaned = line.lstrip('-*• ').strip()
            # 去掉可能的 markdown 加粗 **...**
            cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
            if cleaned and len(cleaned) > 5:
                items.append(cleaned)
        # 匹配数字编号格式 (1., 2., etc.)
        elif re.match(r'^\d+[\.\)]\s*', line):
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
            if cleaned and len(cleaned) > 5:
                items.append(cleaned)
    return items


def parse_review_text(text: str) -> dict[str, Any]:
    """解析 review 文本，提取评分、strengths、weaknesses"""
    result = {
        "rating": None,
        "strengths": [],
        "weaknesses": [],
    }

    if not isinstance(text, str):
        return result

    text_lower = text.lower()

    # 提取评分 - 多种格式
    rating_patterns = [
        r'rating[:\s]+(\d+(?:\.\d+)?)',
        r'score[:\s]+(\d+(?:\.\d+)?)',
        r'recommendation[:\s]+(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*/\s*10',
    ]
    for pattern in rating_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                result["rating"] = float(match.group(1))
                break
            except ValueError:
                pass

    # 使用新的解析函数提取 strengths/weaknesses
    result["strengths"], result["weaknesses"] = parse_strengths_weaknesses(text)

    return result


def row_to_paper(row: dict) -> Paper:
    """将 parquet 行转换为 Paper 对象"""
    return Paper(
        paper_id=str(row.get("paper_id", "")),
        venue_id=str(row.get("venue", "Unknown")),
        year=int(row.get("year", 0)) if row.get("year") else None,
        title=str(row.get("title", "")),
        abstract=str(row.get("abstract", "")),
        authors=[],
        keywords=[],
        sections=[],
        references=[],
        pdf_url=None,
    )


def row_to_reviews(row: dict) -> list[Review]:
    """将 parquet 行转换为 Review 列表"""
    reviews = []

    venue = str(row.get("venue", "Unknown"))
    year = int(row.get("year", 0)) if row.get("year") else None
    paper_id = str(row.get("paper_id", ""))

    # 尝试从 reviews_json 解析
    reviews_json = row.get("reviews_json")
    if isinstance(reviews_json, str):
        try:
            review_list = json.loads(reviews_json)
            for i, rev in enumerate(review_list):
                if not isinstance(rev, dict):
                    continue

                # 提取评分
                rating = None
                scores = rev.get("scores", {})
                if isinstance(scores, dict):
                    rating_val = scores.get("rating")
                    if isinstance(rating_val, (int, float)):
                        rating = float(rating_val)

                # 提取文本内容和 strengths/weaknesses
                content = rev.get("content", {})
                text_parts = []
                strengths = []
                weaknesses = []

                if isinstance(content, dict):
                    # 首先检查合并的 strength_and_weaknesses 字段 (ICLR 2023+)
                    saw = content.get("strength_and_weaknesses")
                    if saw:
                        saw_text = saw if isinstance(saw, str) else (saw.get("value", "") if isinstance(saw, dict) else "")
                        if saw_text:
                            text_parts.append(f"strength_and_weaknesses: {saw_text}")
                            s_list, w_list = parse_strengths_weaknesses(saw_text)
                            strengths.extend(s_list)
                            weaknesses.extend(w_list)

                    # 检查 NeurIPS 的 strengths_and_weaknesses 字段（复数形式）
                    saw_plural = content.get("strengths_and_weaknesses")
                    if saw_plural and not saw:  # 避免重复处理
                        saw_text = saw_plural if isinstance(saw_plural, str) else (saw_plural.get("value", "") if isinstance(saw_plural, dict) else "")
                        if saw_text:
                            text_parts.append(f"strengths_and_weaknesses: {saw_text}")
                            s_list, w_list = parse_strengths_weaknesses(saw_text)
                            strengths.extend(s_list)
                            weaknesses.extend(w_list)

                    # 检查 main_review 字段 (ICLR 2022)
                    main_review = content.get("main_review")
                    if main_review:
                        mr_text = main_review if isinstance(main_review, str) else (main_review.get("value", "") if isinstance(main_review, dict) else "")
                        if mr_text:
                            text_parts.append(f"main_review: {mr_text}")
                            # 如果还没有提取到 strengths/weaknesses，尝试从 main_review 提取
                            if not strengths and not weaknesses:
                                s_list, w_list = parse_strengths_weaknesses(mr_text)
                                strengths.extend(s_list)
                                weaknesses.extend(w_list)

                    # 检查 review 字段 (ICLR 2018-2021)
                    review_text = content.get("review")
                    if review_text:
                        rt_text = review_text if isinstance(review_text, str) else (review_text.get("value", "") if isinstance(review_text, dict) else "")
                        if rt_text:
                            text_parts.append(f"review: {rt_text}")
                            # 如果还没有提取到 strengths/weaknesses，尝试从 review 提取
                            if not strengths and not weaknesses:
                                s_list, w_list = parse_strengths_weaknesses(rt_text)
                                strengths.extend(s_list)
                                weaknesses.extend(w_list)

                    # 然后检查其他字段
                    for key in ["summary_of_the_paper", "summary_of_the_review", "questions", "correctness"]:
                        val = content.get(key, {})
                        if isinstance(val, dict) and "value" in val:
                            text_parts.append(f"{key}: {val['value']}")
                        elif isinstance(val, str) and val:
                            text_parts.append(f"{key}: {val}")

                    # 也检查分开的 strengths/weaknesses 字段（某些会议可能用这种格式，如 NeurIPS 2023）
                    for key in ["strengths", "weaknesses"]:
                        val = content.get(key, {})
                        if isinstance(val, dict) and "value" in val:
                            text_parts.append(f"{key}: {val['value']}")
                            if key == "strengths":
                                s_list, _ = parse_strengths_weaknesses(f"strengths: {val['value']}")
                                strengths.extend(s_list)
                            elif key == "weaknesses":
                                _, w_list = parse_strengths_weaknesses(f"weaknesses: {val['value']}")
                                weaknesses.extend(w_list)
                        elif isinstance(val, str) and val:
                            text_parts.append(f"{key}: {val}")
                            # 直接从字符串解析
                            if key == "strengths":
                                s_list, _ = parse_strengths_weaknesses(f"strengths: {val}")
                                strengths.extend(s_list)
                            elif key == "weaknesses":
                                _, w_list = parse_strengths_weaknesses(f"weaknesses: {val}")
                                weaknesses.extend(w_list)

                text = "\n\n".join(text_parts) if text_parts else ""

                # 创建 Review 并附加 strengths/weaknesses
                review = Review(
                    review_id=rev.get("reply_id") or f"{paper_id}_review_{i}",
                    paper_id=paper_id,
                    venue_id=venue,
                    year=year,
                    rating=rating,
                    text=text,
                    decision=None,  # Decision is at paper level
                )
                # 附加解析好的 strengths/weaknesses
                review._strengths = strengths
                review._weaknesses = weaknesses
                reviews.append(review)
        except json.JSONDecodeError:
            pass

    # 如果没有 reviews_json，尝试 review 列（旧格式）
    if not reviews:
        review_data = row.get("review", [])
        if isinstance(review_data, str):
            review_data = [review_data]

        for i, rev_text in enumerate(review_data):
            if not rev_text:
                continue

            parsed = parse_review_text(str(rev_text))

            review = Review(
                review_id=f"{paper_id}_review_{i}",
                paper_id=paper_id,
                venue_id=venue,
                year=year,
                rating=parsed["rating"],
                text=str(rev_text),
                decision=str(row.get("decision", "")) if i == 0 else None,
            )
            review._strengths = parsed["strengths"]
            review._weaknesses = parsed["weaknesses"]
            reviews.append(review)

    return reviews


def extract_paper_rating(row: dict) -> float | None:
    """从 scores_json 提取平均评分"""
    scores_json = row.get("scores_json")
    if isinstance(scores_json, str):
        try:
            scores = json.loads(scores_json)
            rating_info = scores.get("rating", {})
            if isinstance(rating_info, dict):
                mean_rating = rating_info.get("mean")
                if isinstance(mean_rating, (int, float)):
                    return float(mean_rating)
        except json.JSONDecodeError:
            pass
    return None


def infer_rating_from_decision(decision: str | None) -> float | None:
    """基于决策推断评分

    Accept → 6.5 (边际接受)
    Reject → 4.0 (边际拒绝)
    """
    if not decision:
        return None

    dec_lower = decision.lower()
    if 'accept' in dec_lower:
        # 区分不同类型的 accept
        if 'oral' in dec_lower or 'spotlight' in dec_lower:
            return 8.0
        elif 'poster' in dec_lower:
            return 7.0
        else:
            return 6.5
    elif 'reject' in dec_lower:
        return 4.0
    elif 'workshop' in dec_lower:
        return 5.0
    else:
        return None


def is_rating_decision_consistent(rating: float | None, decision: str | None) -> bool:
    """检查评分与决策是否一致

    过滤异常案例:
    - Rating <= 3 且 Accept → 矛盾
    - Rating >= 7 且 Reject → 矛盾
    """
    if rating is None or decision is None:
        return True  # 无法检查，默认通过

    dec_lower = decision.lower()

    # Accept 但评分很低 → 异常
    if 'accept' in dec_lower and rating <= 3.0:
        return False

    # Reject 但评分很高 → 异常
    if 'reject' in dec_lower and rating >= 7.0:
        return False

    return True


def build_paper_case(
    paper: Paper,
    reviews: list[Review],
    paper_decision: str | None = None,
    paper_rating: float | None = None,  # 从 scores_json 提取的平均评分
    filter_inconsistent: bool = True,
) -> PaperCase | None:
    """从论文和审稿构建 PaperCase

    Args:
        filter_inconsistent: 是否过滤评分与决策矛盾的案例

    Returns:
        PaperCase 或 None (如果不一致且被过滤)
    """

    # 提取 strengths/weaknesses
    all_strengths = []
    all_weaknesses = []
    all_ratings = []

    for review in reviews:
        if review.rating:
            all_ratings.append(review.rating)

        # 优先使用直接提取的 strengths/weaknesses
        if hasattr(review, '_strengths') and review._strengths:
            all_strengths.extend(review._strengths)
        if hasattr(review, '_weaknesses') and review._weaknesses:
            all_weaknesses.extend(review._weaknesses)

        # 如果没有直接提取的，从 text 中解析
        if not hasattr(review, '_strengths') and not hasattr(review, '_weaknesses'):
            parsed = parse_review_text(review.text or "")
            all_strengths.extend(parsed["strengths"])
            all_weaknesses.extend(parsed["weaknesses"])

    # 优先使用 scores_json 中的平均评分
    if paper_rating is not None and 1 <= paper_rating <= 10:
        final_rating = paper_rating
    else:
        # 否则从中位数计算
        final_rating = statistics.median(all_ratings) if all_ratings else None

    # 如果没有评分，从决策推断
    if final_rating is None or final_rating < 1 or final_rating > 10:
        final_rating = infer_rating_from_decision(paper_decision)

    # 数据质量检查：过滤评分与决策矛盾的案例
    if filter_inconsistent and not is_rating_decision_consistent(final_rating, paper_decision):
        logger.debug(f"Skipping inconsistent case: rating={final_rating}, decision={paper_decision}, title={paper.title[:50]}...")
        return None

    # 去重
    unique_strengths = list(dict.fromkeys(all_strengths))[:5]
    unique_weaknesses = list(dict.fromkeys(all_weaknesses))[:5]

    return PaperCase(
        case_id=str(uuid.uuid4()),
        paper_id=paper.paper_id,
        venue_id=paper.venue_id,
        year=paper.year,
        title=paper.title,
        abstract=paper.abstract,
        paper_signature=None,  # 不解析 signature，太慢
        top_strengths=unique_strengths,
        top_weaknesses=unique_weaknesses,
        decisive_issues=unique_weaknesses[:3],  # 用 weaknesses 作为决定性问题
        review_consensus=None,
        decision=paper_decision,
        rating=final_rating,
        source_review_ids=[r.review_id for r in reviews],
        transferable_criteria=[],
        failure_patterns=[],
        embedding=None,
    )


def build_multi_venue_memory(
    config_path: str,
    venues: list[str] | None = None,
    dry_run: bool = False,
    limit_per_venue: int | None = None,
) -> dict[str, int]:
    """构建多会议记忆"""

    config = read_yaml(config_path)

    # 初始化组件
    embedding_client = EmbeddingClient(EmbeddingConfig(**config["embedding"]))
    case_store_path = config.get("memory", {}).get("case_store_path", "data/processed/cases.jsonl")
    case_store = CaseStore(case_store_path, embedding_client=embedding_client)

    venues = venues or list(VENUE_CONFIG.keys())

    stats = {"total_papers": 0, "total_cases": 0, "by_venue": {}}

    for venue_id in venues:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {venue_id}")
        logger.info(f"{'='*60}")

        # 加载数据
        rows = load_parquet_venue(venue_id, include_test=False)
        if not rows:
            logger.warning(f"No data found for {venue_id}")
            continue

        if limit_per_venue:
            rows = rows[:limit_per_venue]

        venue_cases = []
        skipped_inconsistent = 0

        for i, row in enumerate(rows):
            if i % 100 == 0:
                logger.info(f"  Processing {i}/{len(rows)}...")

            try:
                paper = row_to_paper(row)
                reviews = row_to_reviews(row)
                paper_decision = row.get("decision")
                paper_rating = extract_paper_rating(row)  # 从 scores_json 提取

                if not reviews:
                    continue

                case = build_paper_case(
                    paper, reviews, paper_decision, paper_rating,
                    filter_inconsistent=True
                )

                # 跳过不一致的案例
                if case is None:
                    skipped_inconsistent += 1
                    continue

                if not dry_run:
                    case_store.add_case(case)

                venue_cases.append(case)

            except Exception as e:
                logger.warning(f"Failed to process row {i}: {e}")
                continue

        stats["by_venue"][venue_id] = len(venue_cases)
        stats["total_cases"] += len(venue_cases)
        stats["total_papers"] += len(rows)

        logger.info(f"  Built {len(venue_cases)} cases for {venue_id} (skipped {skipped_inconsistent} inconsistent)")

    if not dry_run:
        # 保存
        case_store._save()
        logger.info(f"\nSaved {stats['total_cases']} cases to {case_store_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build multi-venue memory")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--venues", nargs="+", help="Venues to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't save to disk")
    parser.add_argument("--limit", type=int, help="Limit papers per venue")
    args = parser.parse_args()

    stats = build_multi_venue_memory(
        config_path=args.config,
        venues=args.venues,
        dry_run=args.dry_run,
        limit_per_venue=args.limit,
    )

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total papers: {stats['total_papers']}")
    print(f"Total cases built: {stats['total_cases']}")
    print("\nBy venue:")
    for venue, count in stats["by_venue"].items():
        print(f"  {venue}: {count}")


if __name__ == "__main__":
    main()