#!/usr/bin/env python
"""
记忆库构建脚本 V2

支持：
1. 按会议/年份独立构建记忆库
2. 完整提取 reviewer2 的 S/W 内容
3. 热插拔式管理

Usage:
    # 构建 ICLR 2023 记忆库
    python scripts/build_memory_v2.py --venue ICLR --year 2023 --config configs/iclr.yaml

    # 构建所有 ICLR 记忆库
    python scripts/build_memory_v2.py --venue ICLR --all-years --config configs/iclr.yaml

    # 列出所有可用记忆库
    python scripts/build_memory_v2.py --list
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from common.types import PaperCase, ExperienceCard
from common.utils import read_yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_ROOT = Path("/mnt/data/zzh/datasets/reviewer2")
OUTPUT_ROOT = Path("data/processed/memory")
REGISTRY_PATH = Path("data/processed/registry.json")


# =============================================================================
# 数据解析函数
# =============================================================================

def extract_list_items(text: str) -> list[str]:
    """从文本中提取列表项"""
    items = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # 移除列表标记
        cleaned = re.sub(r'^[-*•]\s*', '', line)
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
        # 移除 markdown 加粗
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
        if cleaned and len(cleaned) > 10:
            items.append(cleaned.strip())
    return items


def parse_strengths_weaknesses(text: str) -> tuple[list[str], list[str]]:
    """解析混合的 Strengths And Weaknesses 文本

    支持格式：
    1. "Strengths And Weaknesses:" 混合字段（ICLR 2023）
    2. "Main Review:" 字段中的 strengths/weaknesses 部分
    3. 单独的 "Strengths:" 和 "Weaknesses:" 字段
    """
    strengths = []
    weaknesses = []

    if not text:
        return strengths, weaknesses

    text_lower = text.lower()

    # 找 strengths 和 weaknesses 的位置
    strength_pos = text_lower.find('strength')
    weakness_pos = text_lower.find('weakness')

    # 同时搜索单数形式
    if weakness_pos == -1:
        weakness_pos = text_lower.find('weakness')

    if strength_pos == -1 and weakness_pos == -1:
        return strengths, weaknesses

    # 提取各部分
    if strength_pos >= 0 and weakness_pos >= 0:
        if strength_pos < weakness_pos:
            strength_section = text[strength_pos:weakness_pos]
            weakness_section = text[weakness_pos:]
        else:
            weakness_section = text[weakness_pos:strength_pos]
            strength_section = text[strength_pos:]
    elif strength_pos >= 0:
        strength_section = text[strength_pos:]
        weakness_section = ""
    else:
        strength_section = ""
        weakness_section = text[weakness_pos:]

    # 提取列表项
    strengths = extract_list_items(strength_section)
    weaknesses = extract_list_items(weakness_section)

    # 过滤掉标题行
    strengths = [s for s in strengths if 'strength' not in s.lower()[:10]]
    weaknesses = [w for w in weaknesses if 'weakness' not in w.lower()[:10]]

    return strengths, weaknesses


def parse_rating(rating_str: str) -> float | None:
    """从 rating 字符串提取评分"""
    if not rating_str:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)', str(rating_str))
    return float(match.group(1)) if match else None


def parse_review_file(file_path: Path) -> dict[str, Any] | None:
    """解析单个 review 文件"""
    try:
        with open(file_path, encoding='utf-8') as f:
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
    venue = None
    year = None
    for part in file_path.parts:
        if 'ICLR' in part:
            venue = 'ICLR'
            ym = re.search(r'(\d{4})', part)
            year = int(ym.group(1)) if ym else None
        elif 'NIPS' in part or 'NeurIPS' in part:
            venue = 'NeurIPS'
            ym = re.search(r'(\d{4})', part)
            year = int(ym.group(1)) if ym else None

    # 收集所有 review 的信息
    all_strengths = []
    all_weaknesses = []
    all_ratings = []

    for rev in reviews:
        if not isinstance(rev, dict):
            continue

        # 查找 S/W 字段 - 按优先级尝试不同格式
        sw_text = None
        main_review_text = None

        # 格式1: "Strengths And Weaknesses:" 混合字段（ICLR 2023）
        for key in rev.keys():
            key_lower = key.lower()
            if 'strength' in key_lower and 'weakness' in key_lower:
                sw_text = rev[key]
                break

        # 格式2: "Main Review:" 字段（ICLR 2022）
        if not sw_text:
            for key in rev.keys():
                key_lower = key.lower()
                if 'main review' in key_lower or 'review' in key_lower and 'summary' not in key_lower:
                    main_review_text = rev[key]
                    # 从 Main Review 中提取 S/W
                    s, w = parse_strengths_weaknesses(main_review_text)
                    all_strengths.extend(s)
                    all_weaknesses.extend(w)
                    break

        # 格式3: 单独的 strengths/weaknesses 字段
        if not sw_text and not main_review_text:
            for key in rev.keys():
                key_lower = key.lower()
                if key_lower == 'strengths' or key_lower == 'strength':
                    strengths_text = rev[key]
                    if isinstance(strengths_text, str):
                        all_strengths.extend(extract_list_items(strengths_text))
                elif key_lower == 'weaknesses' or key_lower == 'weakness':
                    weaknesses_text = rev[key]
                    if isinstance(weaknesses_text, str):
                        all_weaknesses.extend(extract_list_items(weaknesses_text))

        if sw_text:
            s, w = parse_strengths_weaknesses(sw_text)
            all_strengths.extend(s)
            all_weaknesses.extend(w)

        # 查找 rating
        for key in rev:
            if 'rating' in key.lower() or 'recommendation' in key.lower():
                rating = parse_rating(rev[key])
                if rating:
                    all_ratings.append(rating)
                break

    # 去重并限制数量
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
        'meta_review': meta_review,
    }


# =============================================================================
# 记忆库管理
# =============================================================================

def get_memory_id(venue: str, year: int) -> str:
    """生成记忆库 ID"""
    return f"{venue.lower()}_{year}"


def load_registry() -> dict:
    """加载注册表"""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"memories": {}, "active_memories": []}


def save_registry(registry: dict) -> None:
    """保存注册表"""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)


def register_memory(memory_id: str, venue: str, year: int,
                    cases_count: int, cards_count: int) -> None:
    """注册记忆库"""
    registry = load_registry()
    registry["memories"][memory_id] = {
        "path": f"memory/{memory_id}",
        "venue": venue,
        "year": year,
        "cases_count": cases_count,
        "cards_count": cards_count,
        "created_at": datetime.now().isoformat(),
        "active": True,
    }
    if memory_id not in registry["active_memories"]:
        registry["active_memories"].append(memory_id)
    save_registry(registry)


def list_memories() -> None:
    """列出所有记忆库"""
    registry = load_registry()
    if not registry["memories"]:
        print("没有已构建的记忆库")
        return

    print("=" * 70)
    print("已构建的记忆库")
    print("=" * 70)
    print(f"{'ID':<20} {'Venue':<10} {'Year':<6} {'Cases':<8} {'Cards':<8} {'Active'}")
    print("-" * 70)
    for mid, info in registry["memories"].items():
        active = "✓" if mid in registry["active_memories"] else "✗"
        print(f"{mid:<20} {info['venue']:<10} {info['year']:<6} {info['cases_count']:<8} {info['cards_count']:<8} {active}")


# =============================================================================
# 构建函数
# =============================================================================

# =============================================================================
# 辅助函数
# =============================================================================

def extract_abstract_from_paper(paper_data: dict) -> str:
    """从 paper 文件的 sections 提取 abstract/introduction"""
    sections = paper_data.get('metadata', {}).get('sections', [])
    if not sections:
        return ""

    # 优先找 abstract 或 introduction
    for sec in sections:
        heading = sec.get('heading', '').lower()
        if 'abstract' in heading or 'introduction' in heading:
            text = sec.get('text', '')
            # 限制长度
            return text[:1000] if text else ""

    # 如果没找到，取第一个 section
    if sections:
        return sections[0].get('text', '')[:1000]

    return ""


def classify_theme(content: str) -> str:
    """根据内容关键词分类 theme"""
    content_lower = content.lower()

    # 定义主题关键词映射
    theme_keywords = {
        "clarity": ["writing", "presentation", "clarity", "clear", "motivation", "explain", "organization"],
        "quality": ["soundness", "correctness", "proof", "theorem", "theory", "technical", "rigorous"],
        "originality": ["novelty", "new", "original", "first", "innovative", "unique"],
        "significance": ["impact", "importance", "significance", "contribution", "valuable", "useful"],
        "experiments": ["experiment", "empirical", "evaluation", "benchmark", "comparison", "results", "dataset"],
        "reproducibility": ["reproducibility", "code", "implementation", "detail", "parameter", "setup"],
    }

    for theme, keywords in theme_keywords.items():
        for kw in keywords:
            if kw in content_lower:
                return theme

    return "general"


def build_paper_case(result: dict, title: str = "", abstract: str = "") -> PaperCase:
    """构建 PaperCase"""
    # 使用 title + abstract 生成文本表示
    case_text = f"{title}\n{abstract}" if title or abstract else ""

    return PaperCase(
        case_id=str(uuid.uuid4()),
        paper_id=result['paper_id'],
        venue_id=result['venue'],
        year=result['year'],
        title=title,
        abstract=abstract,
        paper_signature=None,  # TODO: 需要后续用 LLM 提取
        top_strengths=result['strengths'],
        top_weaknesses=result['weaknesses'],
        decisive_issues=result['weaknesses'][:3],
        review_consensus=result.get('meta_review'),
        decision=result['decision'],
        rating=result['rating'],
        source_review_ids=[],
        transferable_criteria=[],
        failure_patterns=[],
        embedding=None,
    )


def extract_policy_cards(result: dict) -> list[ExperienceCard]:
    """从 review 结果提取 Policy Cards"""
    cards = []

    # 从 weaknesses 提取失败模式
    for i, w in enumerate(result.get('weaknesses', [])):
        if len(w) > 30:
            # 根据内容自动分类 theme
            theme = classify_theme(w)

            card = ExperienceCard(
                card_id=str(uuid.uuid4()),
                kind="policy",
                scope="venue",
                venue_id=result['venue'],
                theme=theme,
                content=f"Papers should avoid: {w[:200]}",
                trigger=[],
                utility=0.5,
                confidence=0.5,
                use_count=0,
                source_ids=[result['paper_id']],
                version=1,
                active=True,
                created_at=datetime.now().isoformat(),  # 转为字符串
                source_trace={},
                metadata={
                    "decision": result['decision'],
                    "rating": result['rating'],
                    "year": result['year'],
                }
            )
            cards.append(card)

    return cards


def build_memory(venue: str, year: int, config_path: str,
                 embedding_client: EmbeddingClient | None = None) -> dict:
    """构建指定会议/年份的记忆库"""

    memory_id = get_memory_id(venue, year)
    output_dir = OUTPUT_ROOT / memory_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找 review 文件
    venue_path = DATA_ROOT / venue
    pattern = f"**/{venue}_{year}*_review/*.json"
    review_files = list(venue_path.glob(pattern))

    # 也尝试 NIPS 别名
    if venue == "NeurIPS" and not review_files:
        venue_path = DATA_ROOT / "NIPS"
        review_files = list(venue_path.glob(f"**/NIPS_{year}*_review/*.json"))

    logger.info(f"找到 {len(review_files)} 个 review 文件")

    # 预加载 metadata 和 paper 文件路径
    metadata_files = {}
    paper_files = {}

    # 直接查找 *_metadata 和 *_paper 目录
    for meta_dir in venue_path.glob(f"**/{venue}_{year}_metadata"):
        if meta_dir.is_dir():
            for mf in meta_dir.glob("*.json"):
                paper_id = mf.stem.replace("_metadata", "")
                metadata_files[paper_id] = mf

    for paper_dir in venue_path.glob(f"**/{venue}_{year}_paper"):
        if paper_dir.is_dir():
            for pf in paper_dir.glob("*.json"):
                paper_id = pf.stem.replace("_paper", "")
                paper_files[paper_id] = pf

    logger.info(f"找到 {len(metadata_files)} 个 metadata 文件, {len(paper_files)} 个 paper 文件")

    # 解析所有文件
    cases = []
    all_cards = []
    stats = {"total": 0, "with_sw": 0, "accept": 0, "reject": 0, "with_title": 0}

    for i, f in enumerate(review_files):
        if i % 500 == 0:
            logger.info(f"处理 {i}/{len(review_files)}...")

        result = parse_review_file(f)
        if not result:
            continue

        stats["total"] += 1

        if result['has_sw']:
            stats["with_sw"] += 1

            # 加载 title 和 abstract
            title = ""
            abstract = ""
            paper_id = result['paper_id']

            # 从 metadata 获取 title
            meta_path = metadata_files.get(paper_id)
            if meta_path and meta_path.exists():
                try:
                    with open(meta_path, encoding='utf-8') as mf:
                        meta_data = json.load(mf)
                        title = meta_data.get('title', '') or ""
                        # 也从 metadata 获取更准确的 decision
                        if not result['decision'] and meta_data.get('decision'):
                            result['decision'] = meta_data.get('decision')
                except Exception as e:
                    pass

            # 从 paper 文件获取 abstract
            paper_path = paper_files.get(paper_id)
            if paper_path and paper_path.exists():
                try:
                    with open(paper_path, encoding='utf-8') as pf:
                        paper_data = json.load(pf)
                        abstract = extract_abstract_from_paper(paper_data)
                except Exception as e:
                    pass

            if title:
                stats["with_title"] += 1

            # 构建 PaperCase
            case = build_paper_case(result, title=title, abstract=abstract)
            cases.append(case)

            # 提取 ExperienceCards
            cards = extract_policy_cards(result)
            all_cards.extend(cards)

            # 统计 decision
            if result['decision'] == 'Accept':
                stats['accept'] += 1
            elif result['decision'] == 'Reject':
                stats['reject'] += 1

    # 为 cases 生成 embedding
    if embedding_client and cases:
        logger.info(f"为 {len(cases)} 个 cases 生成 embedding...")
        texts = []
        for c in cases:
            # 优先使用 title + abstract，否则使用 S/W
            if c.title or c.abstract:
                text = f"{c.title}\n{c.abstract}"
            else:
                text = " ".join(c.top_strengths + c.top_weaknesses)
            texts.append(text)

        embeddings = embedding_client.embed(texts)
        for i, c in enumerate(cases):
            c.embedding = embeddings[i].tolist()

    # 保存文件
    cases_path = output_dir / "cases.jsonl"
    with open(cases_path, 'w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c.model_dump(mode='json'), ensure_ascii=False) + '\n')

    cards_path = output_dir / "policy_cards.jsonl"
    with open(cards_path, 'w', encoding='utf-8') as f:
        for card in all_cards:
            f.write(json.dumps(card.model_dump(mode='json'), ensure_ascii=False) + '\n')

    # 注册记忆库
    register_memory(memory_id, venue, year, len(cases), len(all_cards))

    logger.info(f"\n构建完成:")
    logger.info(f"  - 总文件: {stats['total']}")
    logger.info(f"  - 有 S/W: {stats['with_sw']}")
    logger.info(f"  - 有 title: {stats.get('with_title', 0)}")
    logger.info(f"  - Accept: {stats['accept']}, Reject: {stats['reject']}")
    logger.info(f"  - Cases: {len(cases)}")
    logger.info(f"  - Cards: {len(all_cards)}")
    logger.info(f"  - 输出目录: {output_dir}")

    return {
        "memory_id": memory_id,
        "stats": stats,
        "cases_count": len(cases),
        "cards_count": len(all_cards),
    }


def main():
    parser = argparse.ArgumentParser(description="构建记忆库 V2")
    parser.add_argument("--venue", help="会议名称 (ICLR, NeurIPS)")
    parser.add_argument("--year", type=int, help="年份")
    parser.add_argument("--all-years", action="store_true", help="处理所有年份")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--list", action="store_true", help="列出所有记忆库")
    args = parser.parse_args()

    if args.list:
        list_memories()
        return

    if not args.venue:
        print("请指定 --venue 参数")
        return

    # 初始化 embedding client
    embedding_client = None
    if args.config:
        config = read_yaml(args.config)
        embedding_client = EmbeddingClient(EmbeddingConfig(**config["embedding"]))

    if args.all_years:
        # 处理所有年份
        years = []
        venue_path = DATA_ROOT / args.venue
        if venue_path.exists():
            for d in venue_path.glob(f"{args.venue}_*"):
                ym = re.search(r'(\d{4})', d.name)
                if ym:
                    years.append(int(ym.group(1)))

        years = sorted(set(years))
        logger.info(f"找到年份: {years}")

        for year in years:
            logger.info(f"\n{'='*60}")
            logger.info(f"构建 {args.venue} {year}")
            logger.info(f"{'='*60}")
            build_memory(args.venue, year, args.config, embedding_client)
    elif args.year:
        build_memory(args.venue, args.year, args.config, embedding_client)
    else:
        print("请指定 --year 或 --all-years")


if __name__ == "__main__":
    main()