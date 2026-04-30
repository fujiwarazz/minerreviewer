#!/usr/bin/env python
"""
记忆训练脚本：运行pipeline并从差距中沉淀经验

训练流程：
1. 遍历有GT的论文数据
2. Pipeline推理生成pred_review
3. 对比 pred vs GT (strengths/weaknesses/decision/rating)
4. 从差距中提取有价值的经验
5. 写入记忆库

Usage:
    python scripts/train_memory.py --config configs/iclr.yaml --limit 100 --parquet data/merged_cases/ICLR_2024_merged.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.types import Paper, PaperSignature, ExperienceCard, PaperCase, ArbiterOutput
from pipeline.review_pipeline import ReviewPipeline
from pipeline.distill_experience import ExperienceDistiller, DistillationResult
from pipeline.memory_editor import MemoryEditor
from storage.multi_memory_store import MultiMemoryStore
from storage.multi_case_store import MultiCaseStore
from storage.memory_registry import MemoryRegistry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_papers_with_gt(parquet_path: str, limit: int = None) -> list[dict]:
    """加载有GT的论文数据"""
    df = pd.read_parquet(parquet_path)

    # 过滤有GT的论文（有decision且不是nan）
    df_with_gt = df[df['decision'].notna() | df['strengths'].apply(lambda x: len(x) > 0 if hasattr(x, '__len__') else False)]

    if limit:
        df_with_gt = df_with_gt.head(limit)

    papers = []
    for _, row in df_with_gt.iterrows():
        paper = {
            'paper_id': row['paper_id'],
            'title': row['title'],
            'abstract': row['abstract'],
            'venue_id': row.get('conf', 'ICLR'),
            'year': int(row['year']),
            'gt_decision': row['decision'],
            'gt_strengths': row['strengths'],
            'gt_weaknesses': row['weaknesses'],
            'gt_rating': extract_rating(row['scores_json']),
            'primary_area': row.get('primary_area', ''),
        }
        papers.append(paper)

    return papers


def extract_rating(scores_json: dict | str | None) -> float:
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


def compare_pred_gt(pred: ArbiterOutput, gt: dict) -> dict:
    """对比pred和GT，计算差距"""

    # Decision差距
    pred_decision = pred.decision_recommendation or ""
    gt_decision = gt['gt_decision'] or ""
    decision_match = (
        ("accept" in pred_decision.lower() and "accept" in gt_decision.lower())
        or ("reject" in pred_decision.lower() and "reject" in gt_decision.lower())
        or pred_decision == gt_decision
    )

    # Rating差距
    pred_rating = pred.raw_rating
    gt_rating = gt['gt_rating']
    rating_gap = abs(pred_rating - gt_rating)

    # Strengths/Weaknesses覆盖度
    pred_strengths_set = set(s.lower()[:50] for s in pred.strengths)
    gt_strengths_set = set(s.get('value', s).lower()[:50] if isinstance(s, dict) else s.lower()[:50] for s in gt['gt_strengths'])

    pred_weaknesses_set = set(w.lower()[:50] for w in pred.weaknesses)
    gt_weaknesses_set = set(w.get('value', w).lower()[:50] if isinstance(w, dict) else w.lower()[:50] for w in gt['gt_weaknesses'])

    # 计算overlap
    strengths_overlap = len(pred_strengths_set & gt_strengths_set) / max(len(gt_strengths_set), 1)
    weaknesses_overlap = len(pred_weaknesses_set & gt_weaknesses_set) / max(len(gt_weaknesses_set), 1)

    return {
        'decision_match': decision_match,
        'pred_decision': pred_decision,
        'gt_decision': gt_decision,
        'rating_gap': rating_gap,
        'pred_rating': pred_rating,
        'gt_rating': gt_rating,
        'strengths_overlap': strengths_overlap,
        'weaknesses_overlap': weaknesses_overlap,
        'pred_strengths_count': len(pred.strengths),
        'gt_strengths_count': len(gt['gt_strengths']),
        'pred_weaknesses_count': len(pred.weaknesses),
        'gt_weaknesses_count': len(gt['gt_weaknesses']),
    }


def distill_from_gap(
    pred: ArbiterOutput,
    gt: dict,
    paper: Paper,
    signature: PaperSignature | None,
    gap_analysis: dict,
) -> DistillationResult:
    """从pred vs GT差距中提取经验"""

    result = DistillationResult()

    # 1. 提取PaperCase（基于GT）
    from common.types import PaperCase
    case = PaperCase(
        case_id=str(uuid.uuid4()),
        paper_id=paper.paper_id,
        venue_id=paper.venue_id,
        year=paper.year,
        title=paper.title,
        abstract=paper.abstract,
        paper_signature=signature,
        top_strengths=[s.get('value', s) if isinstance(s, dict) else s for s in gt['gt_strengths'][:5]],
        top_weaknesses=[w.get('value', w) if isinstance(w, dict) else w for w in gt['gt_weaknesses'][:5]],
        decisive_issues=[w.get('value', w) if isinstance(w, dict) else w for w in gt['gt_weaknesses'][:3]] if 'reject' in str(gt['gt_decision']).lower() else [],
        decision=gt['gt_decision'],
        rating=gt['gt_rating'],
        review_consensus=gt['gt_decision'],
        source_review_ids=[],
        transferable_criteria=[],  # TODO: 提取
        failure_patterns=[],
        primary_area=gt.get('primary_area', ''),
    )
    result.paper_case = case

    # 2. 从GT strengths提取policy cards
    for strength in gt['gt_strengths']:
        text = strength.get('value', strength) if isinstance(strength, dict) else strength

        # 检查是否是pred漏掉的strength
        was_missed = not any(text[:50].lower() in ps.lower() for ps in pred.strengths)

        if is_generalizable(text, paper.title):
            card = ExperienceCard(
                card_id=str(uuid.uuid4()),
                kind="policy",
                scope="domain" if gt.get('primary_area') else "venue",
                venue_id=paper.venue_id,
                theme=infer_theme(text),
                content=normalize_text(text),
                trigger=[],
                utility=gt['gt_rating'] / 10.0 if gt['gt_rating'] else 0.5,
                confidence=0.6 if was_missed else 0.5,  # 漏掉的更重要
                use_count=0,
                source_ids=[paper.paper_id],
                created_at=datetime.utcnow(),
                active=True,
                metadata={
                    "memory_year": paper.year,
                    "memory_type": "gt_policy",
                    "memory_domain": gt.get('primary_area', ''),
                    "was_missed_by_pred": was_missed,
                },
            )
            result.policy_updates.append(card)

    # 3. 从GT weaknesses提取critique/failure cards
    for weakness in gt['gt_weaknesses']:
        text = weakness.get('value', weakness) if isinstance(weakness, dict) else weakness

        was_missed = not any(text[:50].lower() in pw.lower() for pw in pred.weaknesses)
        is_failure = 'reject' in str(gt['gt_decision']).lower()

        if is_generalizable(text, paper.title):
            kind = "failure" if (is_failure and was_missed) else "critique"

            card = ExperienceCard(
                card_id=str(uuid.uuid4()),
                kind=kind,
                scope="domain" if gt.get('primary_area') else "venue",
                venue_id=paper.venue_id,
                theme=infer_theme(text),
                content=normalize_text(text),
                trigger=[],
                utility=0.3 if kind == "failure" else 0.5,
                confidence=0.7 if was_missed else 0.5,  # 漏掉的weakness更重要
                use_count=0,
                source_ids=[paper.paper_id],
                created_at=datetime.utcnow(),
                active=True,
                metadata={
                    "memory_year": paper.year,
                    "memory_type": f"gt_{kind}",
                    "memory_domain": gt.get('primary_area', ''),
                    "was_missed_by_pred": was_missed,
                },
            )
            result.critique_cases.append(card)

    return result


def is_generalizable(text: str, title: str = "") -> bool:
    """检查文本是否可迁移"""
    text_lower = text.lower().strip()
    if len(text_lower) < 25:
        return False

    non_transferable = [
        "this paper", "the paper", "our method", "our approach",
        "authors", "figure", "table", "appendix", "section",
        "manuscript", "submission",
    ]
    if any(marker in text_lower for marker in non_transferable):
        return False

    return True


def normalize_text(text: str) -> str:
    """归一化文本"""
    import re
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"\bthis paper\b", "the work", normalized, flags=re.IGNORECASE)
    return normalized


def infer_theme(text: str) -> str:
    """推断主题"""
    text_lower = text.lower()
    theme_keywords = {
        "novelty": ["novel", "new", "original", "contribution"],
        "quality": ["quality", "correctness", "accuracy", "performance"],
        "clarity": ["clarity", "writing", "presentation", "clear"],
        "significance": ["significant", "impact", "important"],
        "empirical": ["empirical", "experiment", "evaluation", "benchmark"],
        "methodology": ["method", "approach", "technique", "algorithm"],
    }
    for theme, keywords in theme_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return theme
    return "general"


def train_one_paper(
    pipeline: ReviewPipeline,
    paper_data: dict,
    memory_editor: MemoryEditor,
) -> dict:
    """训练一篇论文"""

    # 1. 构建Paper对象
    paper = Paper(
        paper_id=paper_data['paper_id'],
        title=paper_data['title'],
        abstract=paper_data['abstract'],
        venue_id=paper_data['venue_id'],
        year=paper_data['year'],
        authors=[],
        fulltext=None,
    )

    # 2. Pipeline推理
    try:
        pred = pipeline.review_paper(paper.paper_id, target_year=paper.year)
    except Exception as e:
        logger.warning(f"Pipeline推理失败: {paper.paper_id} - {e}")
        return {'success': False, 'error': str(e)}

    # 3. 对比pred vs GT
    gap = compare_pred_gt(pred, paper_data)

    # 4. 从差距中提取经验
    experience = distill_from_gap(
        pred=pred,
        gt=paper_data,
        paper=paper,
        signature=pred.trace.get('paper_signature'),
        gap_analysis=gap,
    )

    # 5. 写入记忆库
    updates = {
        'paper_case': None,
        'policy_cards': [],
        'critique_cards': [],
        'failure_cards': [],
    }

    if experience.paper_case:
        if memory_editor.admit_paper_case(experience.paper_case):
            updates['paper_case'] = experience.paper_case.case_id

    for card in experience.all_cards():
        result = memory_editor.admit(card)
        if result in ["admitted_long", "admitted_short"]:
            updates[f'{card.kind}_cards'].append(card.card_id)

    return {
        'success': True,
        'paper_id': paper.paper_id,
        'gap': gap,
        'updates': updates,
        'pred': {
            'decision': pred.decision_recommendation,
            'rating': pred.raw_rating,
        },
        'gt': {
            'decision': paper_data['gt_decision'],
            'rating': paper_data['gt_rating'],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Train memory from GT data")
    parser.add_argument('--config', default='configs/iclr.yaml', help='Config file')
    parser.add_argument('--parquet', default='data/merged_cases/ICLR_2024_merged.parquet', help='Parquet file with GT')
    parser.add_argument('--limit', type=int, default=100, help='Number of papers to train')
    parser.add_argument('--output', default='data/processed/train_results.json', help='Output file')
    parser.add_argument('--dry_run', action='store_true', help='Dry run (no memory update)')
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config}")
    logger.info(f"Loading papers from {args.parquet}")
    logger.info(f"Limit: {args.limit}")
    logger.info(f"Dry run: {args.dry_run}")

    # 初始化Pipeline（直接传路径）
    pipeline = ReviewPipeline(args.config)
    config = pipeline.config

    # 初始化MemoryEditor
    registry = MemoryRegistry(config['memory']['registry_path'])
    memory_store = MultiMemoryStore(registry)
    case_store = MultiCaseStore(registry)
    memory_editor = MemoryEditor(
        memory_store=memory_store,
        case_store=case_store,
    )

    # 加载论文
    papers = load_papers_with_gt(args.parquet, limit=args.limit)
    logger.info(f"Loaded {len(papers)} papers with GT")

    # 训练
    results = []
    stats = {
        'total': len(papers),
        'success': 0,
        'decision_match': 0,
        'avg_rating_gap': 0.0,
        'avg_strengths_overlap': 0.0,
        'avg_weaknesses_overlap': 0.0,
        'policy_cards_added': 0,
        'critique_cards_added': 0,
        'failure_cards_added': 0,
    }

    for i, paper_data in enumerate(papers):
        logger.info(f"Training {i+1}/{len(papers)}: {paper_data['paper_id']}")

        result = train_one_paper(pipeline, paper_data, memory_editor)
        results.append(result)

        if result['success']:
            stats['success'] += 1
            gap = result['gap']
            stats['decision_match'] += gap['decision_match']
            stats['avg_rating_gap'] += gap['rating_gap']
            stats['avg_strengths_overlap'] += gap['strengths_overlap']
            stats['avg_weaknesses_overlap'] += gap['weaknesses_overlap']
            stats['policy_cards_added'] += len(result['updates']['policy_cards'])
            stats['critique_cards_added'] += len(result['updates']['critique_cards'])
            stats['failure_cards_added'] += len(result['updates']['failure_cards'])

    # 计算平均值
    if stats['success'] > 0:
        stats['avg_rating_gap'] /= stats['success']
        stats['avg_strengths_overlap'] /= stats['success']
        stats['avg_weaknesses_overlap'] /= stats['success']
        stats['decision_match_rate'] = stats['decision_match'] / stats['success']

    # 保存结果
    output = {
        'config': args.config,
        'parquet': args.parquet,
        'limit': args.limit,
        'stats': stats,
        'results': results,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {args.output}")

    # 打印统计
    logger.info("\n" + "=" * 70)
    logger.info("训练统计")
    logger.info("=" * 70)
    logger.info(f"  总论文数: {stats['total']}")
    logger.info(f"  成功数: {stats['success']}")
    logger.info(f"  Decision匹配率: {stats['decision_match_rate']*100:.1f}%")
    logger.info(f"  平均Rating差距: {stats['avg_rating_gap']:.2f}")
    logger.info(f"  平均Strengths覆盖: {stats['avg_strengths_overlap']*100:.1f}%")
    logger.info(f"  平均Weaknesses覆盖: {stats['avg_weaknesses_overlap']*100:.1f}%")
    logger.info(f"  新增Policy卡片: {stats['policy_cards_added']}")
    logger.info(f"  新增Critique卡片: {stats['critique_cards_added']}")
    logger.info(f"  新增Failure卡片: {stats['failure_cards_added']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()