#!/usr/bin/env python
"""
从 reviewer2 数据集构建记忆

数据集路径: /mnt/data/zzh/datasets/reviewer2
格式: 每个 review 一个 JSON 文件

Usage:
    python scripts/build_reviewer2_memory.py --config configs/iclr.yaml --dry-run
    python scripts/build_reviewer2_memory.py --config configs/iclr.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import uuid
from pathlib import Path
from typing import Any

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from clients.llm_client import LLMClient, LLMConfig
from common.types import Paper, PaperCase, Review
from common.utils import read_yaml
from storage.case_store import CaseStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据路径
DATA_ROOT = Path("/mnt/data/zzh/datasets/reviewer2")


def _extract_list_items(text: str) -> list[str]:
    """从文本中提取列表项"""
    items = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-') or line.startswith('*') or line.startswith('•'):
            cleaned = line.lstrip('-*• ').strip()
            cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
            if cleaned and len(cleaned) > 5:
                items.append(cleaned)
        elif re.match(r'^\d+[\.\)]\s*', line):
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
            if cleaned and len(cleaned) > 5:
                items.append(cleaned)
    return items


def parse_strengths_weaknesses(text: str) -> tuple[list[str], list[str]]:
    """解析 strengths 和 weaknesses"""
    strengths = []
    weaknesses = []
    if not text:
        return strengths, weaknesses

    def find_section_start(keyword: str) -> int:
        match = re.search(keyword, text, re.IGNORECASE)
        if not match:
            return -1
        pos = match.start()
        line_start = text.rfind('\n', 0, pos) + 1
        return line_start

    def find_section_end(start_pos: int, next_keywords: list[str]) -> int:
        min_pos = len(text)
        for kw in next_keywords:
            match = re.search(kw, text[start_pos:], re.IGNORECASE)
            if match:
                abs_pos = start_pos + match.start()
                line_start = text.rfind('\n', 0, abs_pos) + 1
                min_pos = min(min_pos, line_start)
        return min_pos

    strength_start = find_section_start(r'strength(?:s)?\b')
    weakness_start = find_section_start(r'weakness(?:es)?\b')

    if strength_start >= 0:
        end_pos = weakness_start if weakness_start > strength_start else len(text)
        strength_section = text[strength_start:end_pos]
        strengths = _extract_list_items(strength_section)

    if weakness_start >= 0:
        end_pos = find_section_end(weakness_start, [r'questions?(?:\s|$)', r'summary(?:\s|$)', r'correctness(?:\s|$)', r'limitations?(?:\s|$)'])
        weakness_section = text[weakness_start:end_pos]
        weaknesses = _extract_list_items(weakness_section)

    return strengths, weaknesses


def parse_rating(rating_str: str) -> float | None:
    """从 rating 字符串提取评分"""
    if not rating_str:
        return None

    # 尝试提取数字
    match = re.search(r'(\d+(?:\.\d+)?)', str(rating_str))
    if match:
        return float(match.group(1))
    return None


def parse_review_file(file_path: Path) -> dict[str, Any] | None:
    """解析单个 review 文件"""
    try:
        with open(file_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None

    paper_id = data.get('id', '')
    reviews = data.get('reviews', [])
    meta_review = data.get('metaReview', '')

    # meta_review 可能是 dict 或 str
    if isinstance(meta_review, dict):
        meta_review = meta_review.get('text', '') or str(meta_review)

    if not reviews:
        return None

    # 从文件路径提取 venue 和 year
    parts = file_path.parts
    venue = None
    year = None
    for i, p in enumerate(parts):
        if 'ICLR' in p:
            venue = 'ICLR'
            year_match = re.search(r'(\d{4})', p)
            if year_match:
                year = int(year_match.group(1))
        elif 'NIPS' in p or 'NeurIPS' in p:
            venue = 'NeurIPS'
            year_match = re.search(r'(\d{4})', p)
            if year_match:
                year = int(year_match.group(1))

    # 收集所有 review 的信息
    all_strengths = []
    all_weaknesses = []
    all_ratings = []
    all_texts = []

    for rev in reviews:
        # 查找 S/W 字段
        sw_text = None
        for k, v in rev.items():
            if 'strength' in k.lower() and 'weakness' in k.lower():
                sw_text = v if isinstance(v, str) else str(v)
                break

        if sw_text:
            s, w = parse_strengths_weaknesses(sw_text)
            all_strengths.extend(s)
            all_weaknesses.extend(w)

        # 查找 rating
        for k, v in rev.items():
            if 'rating' in k.lower() or 'recommendation' in k.lower():
                rating = parse_rating(v)
                if rating:
                    all_ratings.append(rating)
                break

        # 收集文本
        for k, v in rev.items():
            if isinstance(v, str) and len(v) > 50:
                all_texts.append(v)

    # 去重
    unique_strengths = list(dict.fromkeys(all_strengths))[:5]
    unique_weaknesses = list(dict.fromkeys(all_weaknesses))[:5]

    # 计算平均评分
    avg_rating = statistics.mean(all_ratings) if all_ratings else None

    # 从 meta_review 推断 decision
    decision = None
    if meta_review:
        meta_lower = meta_review.lower()
        if 'accept' in meta_lower:
            decision = 'Accept'
        elif 'reject' in meta_lower:
            decision = 'Reject'

    return {
        'paper_id': paper_id,
        'venue': venue,
        'year': year,
        'strengths': unique_strengths,
        'weaknesses': unique_weaknesses,
        'rating': avg_rating,
        'decision': decision,
        'has_sw': len(unique_strengths) > 0 or len(unique_weaknesses) > 0,
    }


def build_paper_case(
    paper_id: str,
    venue: str,
    year: int,
    strengths: list[str],
    weaknesses: list[str],
    rating: float | None,
    decision: str | None,
) -> PaperCase:
    """构建 PaperCase"""
    return PaperCase(
        case_id=str(uuid.uuid4()),
        paper_id=paper_id,
        venue_id=venue,
        year=year,
        title="",  # reviewer2 中 title 在 paper 文件，review 文件没有
        abstract="",
        paper_signature=None,
        top_strengths=strengths,
        top_weaknesses=weaknesses,
        decisive_issues=weaknesses[:3],
        review_consensus=None,
        decision=decision,
        rating=rating,
        source_review_ids=[],
        transferable_criteria=[],
        failure_patterns=[],
        embedding=None,
    )


def build_reviewer2_memory(
    config_path: str,
    venues: list[str] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """从 reviewer2 构建记忆"""

    config = read_yaml(config_path)

    # 初始化组件
    embedding_client = EmbeddingClient(EmbeddingConfig(**config["embedding"]))
    case_store_path = config.get("memory", {}).get("case_store_path", "data/processed/cases.jsonl")
    case_store = CaseStore(case_store_path, embedding_client=embedding_client)

    venues = venues or ['ICLR', 'NIPS']

    stats = {"total_files": 0, "total_cases": 0, "with_sw": 0, "by_venue": {}}

    for venue in venues:
        venue_path = DATA_ROOT / venue
        if not venue_path.exists():
            logger.warning(f"Venue path not found: {venue_path}")
            continue

        # 找所有 review 文件
        review_files = list(venue_path.glob('**/*_review/*.json'))

        # 过滤有 S/W 的年份
        if venue == 'ICLR':
            # ICLR 只有 2023 有 S/W
            review_files = [f for f in review_files if 'ICLR_2023' in str(f)]

        if limit:
            review_files = review_files[:limit]

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {venue}: {len(review_files)} files")
        logger.info(f"{'='*60}")

        venue_cases = []

        for i, f in enumerate(review_files):
            if i % 500 == 0:
                logger.info(f"  Processing {i}/{len(review_files)}...")

            result = parse_review_file(f)
            if not result:
                continue

            stats["total_files"] += 1

            if result['has_sw']:
                stats["with_sw"] += 1

                case = build_paper_case(
                    paper_id=result['paper_id'],
                    venue=result['venue'],
                    year=result['year'],
                    strengths=result['strengths'],
                    weaknesses=result['weaknesses'],
                    rating=result['rating'],
                    decision=result['decision'],
                )

                if not dry_run:
                    case_store.add_case(case)

                venue_cases.append(case)

        stats["by_venue"][venue] = len(venue_cases)
        stats["total_cases"] += len(venue_cases)

        logger.info(f"  Built {len(venue_cases)} cases for {venue}")

    if not dry_run:
        case_store._save()
        logger.info(f"\nSaved {stats['total_cases']} cases to {case_store_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build memory from reviewer2 dataset")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--venues", nargs="+", help="Venues to process (ICLR, NIPS)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save to disk")
    parser.add_argument("--limit", type=int, help="Limit files per venue")
    args = parser.parse_args()

    stats = build_reviewer2_memory(
        config_path=args.config,
        venues=args.venues,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Files with S/W: {stats['with_sw']}")
    print(f"Total cases built: {stats['total_cases']}")
    print("\nBy venue:")
    for venue, count in stats["by_venue"].items():
        print(f"  {venue}: {count}")


if __name__ == "__main__":
    main()