#!/usr/bin/env python
"""
从 reviewer2 数据集构建完整记忆系统

构建内容:
1. PaperCase - 论文案例（含 strengths/weaknesses/rating/decision）
2. PolicyCards - 评审标准/政策（从 accept/reject 论文中提取）
3. CritiqueCards - 批评建议（从 review comments 中提取）
4. FailureCards - 失败模式（从 rejected 论文中提取）

Usage:
    python scripts/build_full_memory.py --config configs/iclr.yaml --dry-run
    python scripts/build_full_memory.py --config configs/iclr.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from clients.llm_client import LLMClient, LLMConfig
from common.types import PaperCase, ExperienceCard
from common.utils import read_yaml, write_json
from storage.case_store import CaseStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据路径
DATA_ROOT = Path("/mnt/data/zzh/datasets/reviewer2")

# 主题映射
THEME_MAPPING = {
    'correctness': ['correctness', 'soundness', 'validity', 'error', 'bug', 'flaw'],
    'clarity': ['clarity', 'writing', 'presentation', 'organization', 'readability'],
    'novelty': ['novelty', 'originality', 'new', 'novel', 'contribution'],
    'significance': ['significance', 'impact', 'importance', 'usefulness'],
    'empirical': ['empirical', 'experiment', 'evaluation', 'baseline', 'ablation'],
    'technical': ['technical', 'method', 'algorithm', 'theory', 'proof'],
    'reproducibility': ['reproducibility', 'reproducible', 'code', 'implementation'],
    'comparison': ['comparison', 'baseline', 'sota', 'state-of-the-art'],
    'motivation': ['motivation', 'justification', 'why', 'problem'],
    'ethics': ['ethics', 'ethical', 'bias', 'fairness', 'privacy'],
}


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
    match = re.search(r'(\d+(?:\.\d+)?)', str(rating_str))
    if match:
        return float(match.group(1))
    return None


def infer_theme(text: str) -> str:
    """从文本推断主题"""
    text_lower = text.lower()
    scores = {}
    for theme, keywords in THEME_MAPPING.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[theme] = score
    if scores:
        return max(scores, key=scores.get)
    return 'general'


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
    all_recommendations = []

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

        # 查找 rating/recommendation
        for k, v in rev.items():
            if 'rating' in k.lower() or 'recommendation' in k.lower():
                rating = parse_rating(v)
                if rating:
                    all_ratings.append(rating)
                    all_recommendations.append(str(v))
                break

        # 收集文本
        for k, v in rev.items():
            if isinstance(v, str) and len(v) > 50:
                all_texts.append({'key': k, 'text': v})

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
        'reviews': reviews,
        'meta_review': meta_review,
        'all_texts': all_texts,
        'all_recommendations': all_recommendations,
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
        title="",
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


def distill_policy_card(
    llm_client: LLMClient,
    review_data: dict[str, Any],
    venue: str,
) -> ExperienceCard | None:
    """从 review 中蒸馏评审标准"""
    reviews = review_data.get('reviews', [])
    decision = review_data.get('decision')
    rating = review_data.get('rating')

    if not reviews or not decision:
        return None

    # 收集关键 review 内容
    review_texts = []
    for rev in reviews[:3]:
        for k, v in rev.items():
            if isinstance(v, str) and len(v) > 100:
                review_texts.append(f"[{k}]: {v[:500]}")

    if not review_texts:
        return None

    # 用 LLM 提取评审标准
    prompt = f"""Analyze the following review of a paper that was {decision} (rating: {rating}).

Extract ONE clear, reusable review criterion or standard that can guide future reviews.

Format your response as:
CRITERION: <one clear criterion statement>
THEME: <one of: correctness, clarity, novelty, significance, empirical, technical, reproducibility, comparison, motivation, ethics, general>

Reviews:
{chr(10).join(review_texts[:3])}

Focus on actionable, transferable review standards, not paper-specific details.
"""

    try:
        response = llm_client.generate(prompt)
        lines = response.strip().split('\n')

        criterion = None
        theme = 'general'

        for line in lines:
            if line.startswith('CRITERION:'):
                criterion = line.replace('CRITERION:', '').strip()
            elif line.startswith('THEME:'):
                theme = line.replace('THEME:', '').strip().lower()

        if not criterion or len(criterion) < 20:
            return None

        return ExperienceCard(
            card_id=str(uuid.uuid4()),
            kind="policy",
            scope="venue",
            venue_id=venue,
            theme=theme,
            content=criterion,
            trigger=[],
            utility=0.5,
            confidence=0.5,
            use_count=0,
            source_ids=[review_data['paper_id']],
            metadata={
                'decision': decision,
                'rating': rating,
                'year': review_data.get('year'),
            }
        )
    except Exception as e:
        logger.warning(f"Failed to distill policy: {e}")
        return None


def distill_critique_card(
    llm_client: LLMClient,
    review_data: dict[str, Any],
    venue: str,
) -> ExperienceCard | None:
    """从 review 中蒸馏批评建议"""
    weaknesses = review_data.get('weaknesses', [])
    if not weaknesses:
        return None

    # 选择最有代表性的 weakness
    weakness_text = weaknesses[0] if weaknesses else None
    if not weakness_text or len(weakness_text) < 30:
        return None

    # 推断主题
    theme = infer_theme(weakness_text)

    return ExperienceCard(
        card_id=str(uuid.uuid4()),
        kind="critique",
        scope="venue",
        venue_id=venue,
        theme=theme,
        content=weakness_text[:500],
        trigger=[],
        utility=0.5,
        confidence=0.5,
        use_count=0,
        source_ids=[review_data['paper_id']],
        metadata={
            'decision': review_data.get('decision'),
            'rating': review_data.get('rating'),
            'year': review_data.get('year'),
        }
    )


def distill_failure_card(
    llm_client: LLMClient,
    review_data: dict[str, Any],
    venue: str,
) -> ExperienceCard | None:
    """从 rejected 论文中提取失败模式"""
    decision = review_data.get('decision')
    if decision and 'reject' not in decision.lower():
        return None

    weaknesses = review_data.get('weaknesses', [])
    rating = review_data.get('rating')

    if not weaknesses or (rating and rating > 4):
        return None

    # 用 LLM 提取失败模式
    prompt = f"""Analyze the following weaknesses from a REJECTED paper (rating: {rating}).

Extract ONE clear failure pattern that can warn future reviews.

Format your response as:
PATTERN: <one clear failure pattern statement>
SEVERITY: <critical/major/minor>

Weaknesses:
{chr(10).join(f'- {w}' for w in weaknesses[:3])}

Focus on common rejection reasons, not paper-specific details.
"""

    try:
        response = llm_client.generate(prompt)
        lines = response.strip().split('\n')

        pattern = None
        severity = 'major'

        for line in lines:
            if line.startswith('PATTERN:'):
                pattern = line.replace('PATTERN:', '').strip()
            elif line.startswith('SEVERITY:'):
                severity = line.replace('SEVERITY:', '').strip().lower()

        if not pattern or len(pattern) < 20:
            return None

        return ExperienceCard(
            card_id=str(uuid.uuid4()),
            kind="failure",
            scope="venue",
            venue_id=venue,
            theme=infer_theme(pattern),
            content=pattern,
            trigger=[severity],
            utility=0.6,
            confidence=0.6,
            use_count=0,
            source_ids=[review_data['paper_id']],
            metadata={
                'severity': severity,
                'rating': rating,
                'year': review_data.get('year'),
            }
        )
    except Exception as e:
        logger.warning(f"Failed to distill failure: {e}")
        return None


def build_full_memory(
    config_path: str,
    venues: list[str] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    skip_cases: bool = False,
    skip_policies: bool = False,
    skip_critiques: bool = False,
    skip_failures: bool = False,
) -> dict[str, int]:
    """从 reviewer2 构建完整记忆"""

    config = read_yaml(config_path)

    # 初始化组件
    embedding_client = EmbeddingClient(EmbeddingConfig(**config["embedding"]))
    llm_client = LLMClient(LLMConfig(**config["llm"]))

    case_store_path = config.get("memory", {}).get("case_store_path", "data/processed/cases.jsonl")
    memory_store_path = config.get("memory", {}).get("memory_store_path", "data/processed/memory_store.json")

    case_store = CaseStore(case_store_path, embedding_client=embedding_client)

    venues = venues or ['ICLR', 'NIPS']

    stats = {
        "total_files": 0,
        "cases": {"total": 0, "with_sw": 0},
        "policy_cards": 0,
        "critique_cards": 0,
        "failure_cards": 0,
        "by_venue": {},
    }

    all_cards = []

    for venue in venues:
        venue_path = DATA_ROOT / venue
        if not venue_path.exists():
            logger.warning(f"Venue path not found: {venue_path}")
            continue

        # 找所有 review 文件
        review_files = list(venue_path.glob('**/*_review/*.json'))

        # 过滤有 S/W 的年份
        if venue == 'ICLR':
            review_files = [f for f in review_files if 'ICLR_2023' in str(f)]

        if limit:
            review_files = review_files[:limit]

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {venue}: {len(review_files)} files")
        logger.info(f"{'='*60}")

        venue_cases = []
        venue_policies = []
        venue_critiques = []
        venue_failures = []

        for i, f in enumerate(review_files):
            if i % 100 == 0:
                logger.info(f"  Processing {i}/{len(review_files)}...")

            result = parse_review_file(f)
            if not result:
                continue

            stats["total_files"] += 1

            # 1. Build PaperCase
            if not skip_cases and result['has_sw']:
                stats["cases"]["with_sw"] += 1

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

            # 2. Distill Policy Card (from papers with decision)
            if not skip_policies and result.get('decision'):
                card = distill_policy_card(llm_client, result, venue)
                if card:
                    venue_policies.append(card)
                    stats["policy_cards"] += 1

            # 3. Distill Critique Card
            if not skip_critiques and result['weaknesses']:
                card = distill_critique_card(llm_client, result, venue)
                if card:
                    venue_critiques.append(card)
                    stats["critique_cards"] += 1

            # 4. Distill Failure Card (from rejected papers)
            if not skip_failures:
                card = distill_failure_card(llm_client, result, venue)
                if card:
                    venue_failures.append(card)
                    stats["failure_cards"] += 1

        stats["cases"]["total"] += len(venue_cases)
        stats["by_venue"][venue] = {
            "cases": len(venue_cases),
            "policies": len(venue_policies),
            "critiques": len(venue_critiques),
            "failures": len(venue_failures),
        }

        all_cards.extend(venue_policies)
        all_cards.extend(venue_critiques)
        all_cards.extend(venue_failures)

        logger.info(f"  Built {len(venue_cases)} cases, {len(venue_policies)} policies, "
                   f"{len(venue_critiques)} critiques, {len(venue_failures)} failures for {venue}")

    # Save cases
    if not dry_run:
        case_store._save()
        logger.info(f"\nSaved {stats['cases']['total']} cases to {case_store_path}")

    # Save memory cards
    if not dry_run and all_cards:
        cards_data = [card.model_dump() for card in all_cards]
        write_json(memory_store_path, cards_data)
        logger.info(f"Saved {len(all_cards)} cards to {memory_store_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build full memory from reviewer2 dataset")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--venues", nargs="+", help="Venues to process (ICLR, NIPS)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save to disk")
    parser.add_argument("--limit", type=int, help="Limit files per venue")
    parser.add_argument("--skip-cases", action="store_true", help="Skip building cases")
    parser.add_argument("--skip-policies", action="store_true", help="Skip building policy cards")
    parser.add_argument("--skip-critiques", action="store_true", help="Skip building critique cards")
    parser.add_argument("--skip-failures", action="store_true", help="Skip building failure cards")
    args = parser.parse_args()

    stats = build_full_memory(
        config_path=args.config,
        venues=args.venues,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_cases=args.skip_cases,
        skip_policies=args.skip_policies,
        skip_critiques=args.skip_critiques,
        skip_failures=args.skip_failures,
    )

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"\nCases:")
    print(f"  Total: {stats['cases']['total']}")
    print(f"  With S/W: {stats['cases']['with_sw']}")
    print(f"\nMemory Cards:")
    print(f"  Policy cards: {stats['policy_cards']}")
    print(f"  Critique cards: {stats['critique_cards']}")
    print(f"  Failure cards: {stats['failure_cards']}")
    print("\nBy venue:")
    for venue, vstats in stats["by_venue"].items():
        print(f"  {venue}: {vstats}")


if __name__ == "__main__":
    main()