#!/usr/bin/env python
"""
从合并数据构建热插拔记忆库

按会议年份分模块，支持热插拔：
- PaperCase: 论文审稿案例（strengths/weaknesses/rating/decision）
- ExperienceCard: policy/critique/failure经验卡片

Usage:
    python scripts/build_memory_from_merged.py --config configs/iclr.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.types import PaperCase, ExperienceCard, PaperSignature

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 热插拔记忆库目录结构
MEMORY_BASE_PATH = Path("data/processed/memory")


def is_generalizable_memory(text: str, title: str = "", primary_area: str = "") -> bool:
    """过滤明显依赖原论文实体的句子，避免直接升格成通用 memory"""
    text_lower = text.lower().strip()
    if len(text_lower) < 25:
        return False

    # 过滤包含图表引用的
    entity_like_patterns = [
        r"\btable\s+\d+\b",
        r"\bfigure\s+\d+\b",
        r"\bsection\s+\d+\b",
        r"\b\d+(\.\d+)?%\b",
        r"\b\d+x\b",
        r"\bstate[- ]of[- ]the[- ]art\b",
    ]
    if any(re.search(pattern, text_lower) for pattern in entity_like_patterns):
        return False

    # 过滤包含论文特定引用的
    non_transferable_markers = [
        "this paper",
        "the paper",
        "our method",
        "our approach",
        "authors",
        "figure",
        "table",
        "appendix",
        "section",
        "manuscript",
        "submission",
    ]
    if any(marker in text_lower for marker in non_transferable_markers):
        return False

    # 过滤包含标题关键词的（可能是特定方法名）
    if title:
        title_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", title)
        }
        # 排除常见通用词
        common_words = {"learning", "method", "model", "network", "approach", "system", "algorithm", "framework"}
        title_tokens = title_tokens - common_words
        if any(token in text_lower for token in title_tokens if len(token) > 4):
            return False

    return True


def normalize_memory_text(text: str) -> str:
    """轻量归一化，减少具体论文痕迹"""
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"\bthis paper\b", "the work", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bour method\b", "the proposed method", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bour approach\b", "the proposed approach", normalized, flags=re.IGNORECASE)
    return normalized


def infer_theme(text: str) -> str:
    """推断主题"""
    text_lower = text.lower()
    theme_keywords = {
        "novelty": ["novel", "new", "original", "contribution", "first", "unique"],
        "quality": ["quality", "correctness", "accuracy", "performance", "result"],
        "clarity": ["clarity", "writing", "presentation", "explain", "clear", "well-written", "readable"],
        "significance": ["significant", "impact", "important", "useful", "practical"],
        "reproducibility": ["reproducib", "code", "implementation", "detail", "experiment"],
        "soundness": ["sound", "theory", "proof", "mathematical", "rigorous"],
        "empirical": ["empirical", "experiment", "evaluation", "benchmark", "dataset"],
        "comparison": ["comparison", "baseline", "state-of-the-art", "sota"],
        "motivation": ["motivation", "problem", "application", "real-world"],
        "methodology": ["method", "approach", "technique", "algorithm", "architecture"],
    }
    for theme, keywords in theme_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return theme
    return "general"


def extract_rating_from_scores(scores_json: dict | str | None) -> float:
    """从scores_json提取平均rating"""
    if scores_json is None:
        return 0.0

    if isinstance(scores_json, str):
        try:
            scores_json = json.loads(scores_json)
        except:
            return 0.0

    if isinstance(scores_json, dict):
        rating_info = scores_json.get("rating", {})
        if isinstance(rating_info, dict):
            return rating_info.get("mean", 0.0)

    return 0.0


def build_paper_case(row: pd.Series) -> PaperCase:
    """从一行数据构建PaperCase"""
    # 处理strengths/weaknesses
    strengths = []
    weaknesses = []

    for s in row.get('strengths', []):
        if isinstance(s, dict):
            strengths.append(s.get('value', str(s)))
        else:
            strengths.append(str(s))

    for w in row.get('weaknesses', []):
        if isinstance(w, dict):
            weaknesses.append(w.get('value', str(w)))
        else:
            weaknesses.append(str(w))

    # 处理decision
    decision = row.get('decision', '')
    if pd.isna(decision):
        decision = 'Unknown'

    # 提取rating
    rating = extract_rating_from_scores(row.get('scores_json'))

    # 构建paper_signature（简化版，只有primary_area）
    signature = None
    if row.get('primary_area'):
        signature = PaperSignature(
            paper_type="unknown",
            tasks=["unknown"],
            domain=row.get('primary_area', ''),
            method_family=["unknown"],
            datasets=["unknown"],
        )

    case = PaperCase(
        case_id=str(uuid.uuid4()),
        paper_id=row.get('paper_id', ''),
        venue_id=row.get('conf', row.get('venue', 'Unknown')),
        year=int(row.get('year', 2024)),
        title=row.get('title', ''),
        abstract=row.get('abstract', ''),
        paper_signature=signature,
        top_strengths=strengths[:5],
        top_weaknesses=weaknesses[:5],
        decisive_issues=weaknesses[:3] if decision == 'Reject' else [],
        decision=decision,
        rating=rating,
        review_consensus=decision,
        source_review_ids=[],
        transferable_criteria=[],  # 后面提取
        failure_patterns=[],
        primary_area=row.get('primary_area', ''),
    )

    return case


def build_experience_cards(
    case: PaperCase,
    strengths: list[str],
    weaknesses: list[str],
) -> list[ExperienceCard]:
    """从strengths/weaknesses构建ExperienceCards"""
    cards = []

    # 确定scope
    scope = "domain" if case.primary_area else "venue"
    venue_id = case.venue_id

    # 从strengths提取policy cards
    for strength in strengths:
        if not is_generalizable_memory(strength, case.title, case.primary_area):
            continue

        card = ExperienceCard(
            card_id=str(uuid.uuid4()),
            kind="policy",
            scope=scope,
            venue_id=venue_id,
            theme=infer_theme(strength),
            content=normalize_memory_text(strength),
            trigger=[],
            utility=max(case.rating / 10.0, 0.5),
            confidence=0.5,
            use_count=0,
            source_ids=[case.case_id],
            created_at=datetime.utcnow(),
            active=True,
            metadata={
                "memory_year": case.year,
                "memory_type": "generic_policy",
                "memory_domain": case.primary_area,
            },
        )
        cards.append(card)

    # 从weaknesses提取critique/failure cards
    for weakness in weaknesses:
        if not is_generalizable_memory(weakness, case.title, case.primary_area):
            continue

        # 判断是critique还是failure
        is_failure_pattern = any(
            kw in weakness.lower()
            for kw in ["missing", "lacks", "insufficient", "incomplete", "unclear", "not", "fails", "limited"]
        )

        kind = "failure" if (is_failure_pattern and case.decision == 'Reject') else "critique"

        card = ExperienceCard(
            card_id=str(uuid.uuid4()),
            kind=kind,
            scope=scope,
            venue_id=venue_id,
            theme=infer_theme(weakness),
            content=normalize_memory_text(weakness),
            trigger=[],
            utility=0.3 if kind == "failure" else 0.5,
            confidence=0.5,
            use_count=0,
            source_ids=[case.case_id],
            created_at=datetime.utcnow(),
            active=True,
            metadata={
                "memory_year": case.year,
                "memory_type": f"generic_{kind}",
                "memory_domain": case.primary_area,
            },
        )
        cards.append(card)

    return cards[:10]  # 每篇论文最多10张卡片


def serialize_model(obj):
    """序列化对象，处理datetime"""
    if hasattr(obj, 'model_dump'):
        data = obj.model_dump()
    else:
        data = obj

    # 处理datetime
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()

    return data


def build_memory_for_venue_year(
    parquet_path: str,
    output_dir: Path,
) -> dict:
    """为单个会议年份构建记忆库"""

    logger.info(f"Processing {parquet_path}")

    df = pd.read_parquet(parquet_path)

    cases = []
    all_cards = []

    stats = {
        'total_papers': len(df),
        'total_cases': 0,
        'total_cards': 0,
        'policy_cards': 0,
        'critique_cards': 0,
        'failure_cards': 0,
        'rejected_papers': 0,
    }

    for idx, row in df.iterrows():
        # 构建PaperCase
        case = build_paper_case(row)
        cases.append(case)

        if case.decision == 'Reject':
            stats['rejected_papers'] += 1

        # 构建ExperienceCards
        strengths = case.top_strengths
        weaknesses = case.top_weaknesses

        cards = build_experience_cards(case, strengths, weaknesses)
        all_cards.extend(cards)

        if (idx + 1) % 1000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)} papers, {len(all_cards)} cards")

    stats['total_cases'] = len(cases)
    stats['total_cards'] = len(all_cards)

    # 统计卡片类型
    for card in all_cards:
        if card.kind == 'policy':
            stats['policy_cards'] += 1
        elif card.kind == 'critique':
            stats['critique_cards'] += 1
        elif card.kind == 'failure':
            stats['failure_cards'] += 1

    # 保存cases
    cases_file = output_dir / "cases.jsonl"
    with open(cases_file, 'w') as f:
        for case in cases:
            f.write(json.dumps(serialize_model(case), ensure_ascii=False) + '\n')

    # 保存cards
    cards_file = output_dir / "policy_cards.jsonl"
    with open(cards_file, 'w') as f:
        for card in all_cards:
            f.write(json.dumps(serialize_model(card), ensure_ascii=False) + '\n')

    logger.info(f"  Saved {len(cases)} cases, {len(all_cards)} cards to {output_dir}")

    return stats


def update_registry(registry_path: Path, venue: str, year: int, stats: dict):
    """更新registry.json"""

    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {'memories': {}, 'active_memories': []}

    memory_id = f"{venue}_{year}"

    registry['memories'][memory_id] = {
        'path': f"memory/{memory_id}",
        'venue': venue,
        'year': year,
        'cases_count': stats['total_cases'],
        'cards_count': stats['total_cards'],
        'policy_count': stats['policy_cards'],
        'critique_count': stats['critique_cards'],
        'failure_count': stats['failure_cards'],
        'rejected_papers': stats['rejected_papers'],
        'created_at': datetime.utcnow().isoformat(),
        'active': True,
    }

    # 确保在active_memories中
    if memory_id not in registry['active_memories']:
        registry['active_memories'].append(memory_id)

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Updated registry: {memory_id}")


def main():
    parser = argparse.ArgumentParser(description="Build memory from merged data")
    parser.add_argument('--input_dir', default='data/merged_cases', help='Input parquet directory')
    parser.add_argument('--output_dir', default='data/processed/memory', help='Output memory directory')
    parser.add_argument('--registry_path', default='data/processed/registry.json', help='Registry file path')
    parser.add_argument('--venues', nargs='+', default=None, help='Specific venues to process')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    registry_path = Path(args.registry_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有parquet文件
    parquet_files = sorted(input_dir.glob("*_merged.parquet"))

    if args.venues:
        parquet_files = [f for f in parquet_files if any(v in f.name for v in args.venues)]

    logger.info(f"Found {len(parquet_files)} parquet files to process")

    total_stats = {
        'total_papers': 0,
        'total_cases': 0,
        'total_cards': 0,
        'policy_cards': 0,
        'critique_cards': 0,
        'failure_cards': 0,
        'rejected_papers': 0,
    }

    for parquet_file in parquet_files:
        # 从文件名提取venue和year
        name = parquet_file.stem.replace('_merged', '')
        parts = name.split('_')
        venue = parts[0]
        year = int(parts[1])

        # 创建输出目录
        memory_dir = output_dir / f"{venue}_{year}"
        memory_dir.mkdir(parents=True, exist_ok=True)

        # 构建记忆
        stats = build_memory_for_venue_year(str(parquet_file), memory_dir)

        # 更新registry
        update_registry(registry_path, venue, year, stats)

        # 累计统计
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    # 打印总统计
    logger.info("\n" + "=" * 70)
    logger.info("总统计")
    logger.info("=" * 70)
    logger.info(f"  处理文件数: {len(parquet_files)}")
    logger.info(f"  总论文数: {total_stats['total_papers']}")
    logger.info(f"  总cases: {total_stats['total_cases']}")
    logger.info(f"  总cards: {total_stats['total_cards']}")
    logger.info(f"    policy: {total_stats['policy_cards']}")
    logger.info(f"    critique: {total_stats['critique_cards']}")
    logger.info(f"    failure: {total_stats['failure_cards']}")
    logger.info(f"  rejected论文数: {total_stats['rejected_papers']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()