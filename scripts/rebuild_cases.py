#!/usr/bin/env python
"""
正确地从 reviewer2 数据集构建完整 memory

修复之前的问题：
1. 正确提取所有有 decision 的论文
2. 平衡 Accept/Reject 案例
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

sys_path = __import__('sys')
sys_path.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.types import PaperCase
from common.utils import write_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_ROOT = Path("/mnt/data/zzh/datasets/reviewer2")


def parse_rating(rating_str: str) -> float | None:
    if not rating_str:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)', str(rating_str))
    if match:
        return float(match.group(1))
    return None


def parse_strengths_weaknesses(text: str) -> tuple[list[str], list[str]]:
    """解析 strengths 和 weaknesses"""
    strengths, weaknesses = [], []
    if not text:
        return strengths, weaknesses

    # 找 strengths 和 weaknesses 部分
    strength_start = -1
    weakness_start = -1

    for match in re.finditer(r'(strength(?:s)?\b)', text, re.IGNORECASE):
        if 'weakness' not in text[max(0, match.start()-10):match.start()].lower():
            strength_start = text.rfind('\n', 0, match.start()) + 1
            break

    for match in re.finditer(r'(weakness(?:es)?\b)', text, re.IGNORECASE):
        weakness_start = text.rfind('\n', 0, match.start()) + 1
        break

    # 提取列表项
    def extract_items(section_text):
        items = []
        for line in section_text.split('\n'):
            line = line.strip()
            if line.startswith(('-', '*', '•')) or re.match(r'^\d+[\.\)]\s', line):
                cleaned = re.sub(r'^[-*•\d\.\)\s]+', '', line).strip()
                cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
                if cleaned and len(cleaned) > 10:
                    items.append(cleaned)
        return items

    if strength_start >= 0:
        end = weakness_start if weakness_start > strength_start else len(text)
        strengths = extract_items(text[strength_start:end])[:5]

    if weakness_start >= 0:
        # 找 weakness section 的结束
        end_match = re.search(r'(?:^|\n)\s*(?:questions?|summary|correctness|limitations?)(?:\s|$)', text[weakness_start:], re.IGNORECASE)
        end = weakness_start + end_match.start() if end_match else len(text)
        weaknesses = extract_items(text[weakness_start:end])[:5]

    return strengths, weaknesses


def parse_review_file(file_path: Path) -> dict | None:
    """解析单个 review 文件"""
    try:
        with open(file_path) as f:
            data = json.load(f)
    except:
        return None

    paper_id = data.get('id', '')
    reviews = data.get('reviews', [])
    meta_review = data.get('metaReview', '')

    if not reviews:
        return None

    # 处理 meta_review
    if isinstance(meta_review, dict):
        meta_review = meta_review.get('text', '') or str(meta_review)

    # 提取 venue/year
    venue, year = None, None
    for part in file_path.parts:
        if 'ICLR' in part:
            venue = 'ICLR'
            ym = re.search(r'(\d{4})', part)
            if ym:
                year = int(ym.group(1))
        elif 'NIPS' in part or 'NeurIPS' in part:
            venue = 'NeurIPS'
            ym = re.search(r'(\d{4})', part)
            if ym:
                year = int(ym.group(1))

    # 提取 S/W
    all_strengths, all_weaknesses = [], []
    for rev in reviews:
        for k, v in rev.items():
            if 'strength' in k.lower() and 'weakness' in k.lower() and isinstance(v, str):
                s, w = parse_strengths_weaknesses(v)
                all_strengths.extend(s)
                all_weaknesses.extend(w)

    # 去重
    unique_strengths = list(dict.fromkeys(all_strengths))[:5]
    unique_weaknesses = list(dict.fromkeys(all_weaknesses))[:5]

    # 提取 rating
    all_ratings = []
    for rev in reviews:
        for k, v in rev.items():
            if 'rating' in k.lower() or 'recommendation' in k.lower():
                r = parse_rating(v)
                if r and r < 15:  # 过滤异常值
                    all_ratings.append(r)

    avg_rating = statistics.mean(all_ratings) if all_ratings else None

    # 提取 decision
    decision = None
    if meta_review and isinstance(meta_review, str):
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
        'has_sw': bool(unique_strengths or unique_weaknesses),
    }


def build_cases(config_path: str, output_path: str, balance: bool = True):
    """构建 cases"""

    all_cases = []
    stats = defaultdict(lambda: {'accept': 0, 'reject': 0, 'no_decision': 0})

    for venue in ['ICLR', 'NIPS']:
        venue_path = DATA_ROOT / venue
        if not venue_path.exists():
            continue

        # 只处理有 S/W 的年份
        if venue == 'ICLR':
            patterns = ['ICLR_2023']
        else:
            patterns = ['NIPS_2022', 'NeurIPS_2022']

        for pattern in patterns:
            files = list(venue_path.glob(f'**/{pattern}*_review/*.json'))
            if not files:
                files = list(venue_path.glob(f'**/{pattern}/*_review/*.json'))

            logger.info(f"Processing {pattern}: {len(files)} files")

            for f in files:
                result = parse_review_file(f)
                if not result:
                    continue

                key = f"{result['venue']}_{result['year']}"

                if result['decision']:
                    if result['decision'] == 'Accept':
                        stats[key]['accept'] += 1
                    else:
                        stats[key]['reject'] += 1

                    case = PaperCase(
                        case_id=str(uuid.uuid4()),
                        paper_id=result['paper_id'],
                        venue_id=result['venue'],
                        year=result['year'],
                        title="",
                        abstract="",
                        paper_signature=None,
                        top_strengths=result['strengths'],
                        top_weaknesses=result['weaknesses'],
                        decisive_issues=result['weaknesses'][:3],
                        review_consensus=None,
                        decision=result['decision'],
                        rating=result['rating'],
                        source_review_ids=[],
                        transferable_criteria=[],
                        failure_patterns=[],
                        embedding=None,
                    )
                    all_cases.append(case)
                else:
                    stats[key]['no_decision'] += 1

    # 打印统计
    print("\n=== 数据统计 ===")
    for key, s in sorted(stats.items()):
        total = s['accept'] + s['reject']
        print(f"{key}: Accept={s['accept']}, Reject={s['reject']}, NoDecision={s['no_decision']}")

    # 平衡 Accept/Reject
    if balance:
        accept_cases = [c for c in all_cases if c.decision == 'Accept']
        reject_cases = [c for c in all_cases if c.decision == 'Reject']

        print(f"\n平衡前: Accept={len(accept_cases)}, Reject={len(reject_cases)}")

        # 限制 Accept 数量，保持合理比例
        min_count = min(len(accept_cases), len(reject_cases) * 3)  # Accept 最多是 Reject 的 3 倍
        if len(accept_cases) > min_count:
            import random
            random.seed(42)
            accept_cases = random.sample(accept_cases, min_count)

        all_cases = accept_cases + reject_cases
        print(f"平衡后: Accept={len(accept_cases)}, Reject={len(reject_cases)}, Total={len(all_cases)}")

    # 保存
    data = [c.model_dump() for c in all_cases]
    write_json(output_path, data)
    logger.info(f"Saved {len(all_cases)} cases to {output_path}")

    return all_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/iclr.yaml")
    parser.add_argument("--output", default="data/processed/cases.jsonl")
    parser.add_argument("--no-balance", action="store_true", help="Don't balance Accept/Reject")
    args = parser.parse_args()

    build_cases(args.config, args.output, balance=not args.no_balance)