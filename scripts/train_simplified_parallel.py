#!/usr/bin/env python
"""
多进程简化训练脚本

简化流程：
1. Parse paper → signature
2. Retrieval (cases + cards)
3. Mine criteria
4. Theme agents → strengths/weaknesses
5. 汇总strengths/weaknesses
6. 从GT对比提取经验
7. 写入记忆库

去掉：Arbiter decision, Verification, Calibration等

Usage:
    python scripts/train_simplified_parallel.py --tokens tokens.txt --parquet data/merged_cases/ICLR_2024_merged.parquet --workers 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.types import (
    Paper, PaperSignature, ExperienceCard, PaperCase,
    ArbiterOutput, ThemeOutput, Criterion, ActivatedCriterion,
)
from clients.llm_client import LLMClient, LLMConfig
from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from pipeline.parse_paper import PaperParser
from pipeline.retrieve import Retriever
from pipeline.mine_criteria import CriteriaMiner
from pipeline.plan_criteria import CriteriaPlanner
from pipeline.rewrite_criteria import CriteriaRewriter
from agents.theme_agent import ThemeAgent
from agents.base import AgentConfig
from storage.memory_registry import MemoryRegistry
from storage.memory_store import MemoryStore
from storage.multi_memory_store import MultiMemoryStore
from storage.multi_case_store import MultiCaseStore
from pipeline.memory_editor import MemoryEditor
from storage.doc_store import DocStore
from pipeline.learn_from_comparison import ComparisonLearner  # 对比学习模块

logger = logging.getLogger(__name__)


def load_tokens(tokens_file: str) -> list[str]:
    """加载token列表"""
    with open(tokens_file) as f:
        content = f.read().strip()
        # 支持逗号分隔或换行分隔
        if ',' in content:
            tokens = [t.strip() for t in content.split(',') if t.strip()]
        else:
            tokens = [t.strip() for t in content.split('\n') if t.strip()]
    return tokens


def get_processed_paper_ids(cards_path: str) -> set:
    """获取已处理的论文ID（从已生成的卡片中提取）"""
    processed = set()
    if Path(cards_path).exists():
        with open(cards_path) as f:
            for line in f:
                try:
                    card = json.loads(line)
                    source_ids = card.get('source_ids', [])
                    processed.update(source_ids)
                except:
                    continue
    return processed


def load_papers_with_gt(parquet_path: str, limit: int = None, skip_processed: bool = False) -> list[dict]:
    """加载有GT的论文数据，可选跳过已处理的"""
    df = pd.read_parquet(parquet_path)
    # 过滤有GT的
    df_with_gt = df[
        df['decision'].notna() |
        df['strengths'].apply(lambda x: len(x) > 0 if hasattr(x, '__len__') else False)
    ]

    # 跳过已处理的论文
    if skip_processed:
        cards_path = "data/processed/memory/ICLR_2024_learned/policy_cards.jsonl"
        processed_ids = get_processed_paper_ids(cards_path)
        logger.info(f"Found {len(processed_ids)} already processed papers")
        df_with_gt = df_with_gt[~df_with_gt['paper_id'].isin(processed_ids)]
        logger.info(f"After skipping: {len(df_with_gt)} papers remaining")

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


def extract_rating(scores_json) -> float:
    """提取rating"""
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


class SimplifiedPipeline:
    """简化Pipeline，只到strengths/weaknesses汇总"""

    def __init__(self, config: dict, token: str, worker_id: int):
        self.config = config
        self.worker_id = worker_id

        # 使用独立的LLM client（每个worker有自己的token）
        llm_config = LLMConfig(**config["llm"])
        llm_config.api_key = token
        self.llm = LLMClient(llm_config)

        self.embedding_client = EmbeddingClient(EmbeddingConfig(**config["embedding"]))

        # 初始化组件
        self.paper_parser = PaperParser(self.llm)

        # 初始化记忆库（持续读写：加载之前训练产出的learned_cards）
        registry_path = config['memory']['registry_path']
        self.registry = MemoryRegistry(registry_path)

        # 创建 learned 记忆库目录（用于存储训练产出的cards）
        learned_memory_path = Path("data/processed/memory/ICLR_2024_learned/policy_cards.jsonl")
        learned_memory_path.parent.mkdir(parents=True, exist_ok=True)

        # 只加载 learned 记忆库（不加载其他预制记忆）
        # 这样第二次训练能用到第一次产出的经验
        self.learned_memory_store = MemoryStore(learned_memory_path)
        logger.info(f"Loaded {len(self.learned_memory_store.cards)} learned cards from previous training")

        # case_store 保持 skip_load=True（不加载预制cases）
        self.case_store = MultiCaseStore(
            registry=self.registry,
            embedding_client=self.embedding_client,
            skip_load=True,
        )

        self.retriever = Retriever(
            venue_id=config['venue_id'],
            embedding_cfg=EmbeddingConfig(**config['embedding']),
            vector_store=config.get('vector_store'),
            case_store=self.case_store,
            memory_store=self.learned_memory_store,
        )

        self.criteria_miner = CriteriaMiner(self.llm, self.embedding_client, config.get("vector_store"))
        self.criteria_planner = CriteriaPlanner(self.llm)
        self.criteria_rewriter = CriteriaRewriter(self.llm)

        self.theme_agent = ThemeAgent(
            config=AgentConfig(name="theme", llm=self.llm),
            theme="general",
            use_fulltext=False,
        )

        self.doc_store = DocStore()

    def run(self, paper_data: dict) -> dict:
        """运行简化流程，记录每一步trace"""
        paper = Paper(
            paper_id=paper_data['paper_id'],
            title=paper_data['title'],
            abstract=paper_data['abstract'],
            venue_id=paper_data['venue_id'],
            year=paper_data['year'],
            authors=[],
            fulltext=None,
        )

        # 记录每一步的trace
        trace = {'paper_id': paper.paper_id}

        try:
            # Step 1: Parse paper signature
            signature = self.paper_parser.parse(paper)
            trace['step1_parse'] = {
                'input': {'title': paper.title[:100], 'abstract': paper.abstract[:200]},
                'output': signature.model_dump() if signature else {},
            }

            # Step 2: Retrieval
            retrieval_cfg = self.config.get('retrieval', {})
            bundle = self.retriever.retrieve(
                target_paper=paper,
                top_k_papers=retrieval_cfg.get('top_k_papers', 4),
                top_k_reviews=retrieval_cfg.get('top_k_reviews', 12),
                unrelated_k=retrieval_cfg.get('unrelated_k', 0),
                similarity_threshold=retrieval_cfg.get('similarity_threshold', 0.35),
                target_year=paper_data['year'],
                paper_signature=signature,
                use_case_memory=retrieval_cfg.get('use_case_memory', True),
            )
            trace['step2_retrieval'] = {
                'input': {'signature': signature.model_dump() if signature else {}},
                'output': {
                    'cases': len(bundle.similar_paper_cases),
                    'policy_cards': len(bundle.policy_cards),
                    'critique_cards': len(bundle.critique_cases),
                    'failure_cards': len(bundle.failure_cards),
                    'related_papers': len(bundle.related_papers),
                    'policy_card_ids': [c.card_id for c in bundle.policy_cards],
                    'critique_card_ids': [c.card_id for c in bundle.critique_cases],
                    'failure_card_ids': [c.card_id for c in bundle.failure_cards],
                },
            }

            # Step 3: Mine criteria
            # 修复：用primary_area（数据集标注）而不是signature.domain（推断）
            primary_area = paper_data.get('primary_area', '')
            domain_for_criteria = primary_area if primary_area else (signature.domain if signature else None)

            content_criteria = self.criteria_miner.mine_content_criteria(
                domain_for_criteria, bundle.related_papers, bundle.related_reviews
            )
            policy_criteria = self.criteria_miner.mine_policy_criteria(
                bundle.venue_policy, []
            )
            trace['step3_mine_criteria'] = {
                'input': {'domain': domain_for_criteria},
                'output': {
                    'content_criteria': [c.text[:80] for c in content_criteria],
                    'policy_criteria': [c.text[:80] for c in policy_criteria],
                },
            }

            # Step 4: Plan criteria
            activated = self.criteria_planner.plan(
                signature, bundle, content_criteria, policy_criteria
            )
            trace['step4_plan_criteria'] = {
                'input': {'content_count': len(content_criteria), 'policy_count': len(policy_criteria)},
                'output': {
                    'activated_count': len(activated),
                    'sources': [a.source for a in activated],
                    'activated_criteria': [{'theme': a.theme, 'criterion': a.criterion, 'source': a.source, 'priority': a.priority} for a in activated],
                },
            }

            # Step 5: Convert activated criteria to Criterion format
            criteria = []
            for a in activated:
                criteria.append(Criterion(
                    criterion_id=f"{a.source}_{a.theme}",
                    text=a.criterion,
                    theme=a.theme,
                    kind="policy" if a.source == "memory" else "content",
                    source_ids=[a.source],
                ))
            trace['step5_rewrite_criteria'] = {
                'input': {'activated_count': len(activated)},
                'output': {
                    'criteria_count': len(criteria),
                    'criteria': [{'theme': c.theme, 'text': c.text} for c in criteria],
                },
            }

            # Step 6: Run theme agents
            theme_outputs = self._run_theme_agents(paper, criteria, bundle.policy_cards, bundle.critique_cases)
            trace['step6_theme_agents'] = {
                'input': {'themes': list(set(c.theme for c in criteria))},
                'output': {
                    'themes_run': len(theme_outputs),
                    'theme_details': [
                        {
                            'theme': o.theme,
                            'strengths': o.strengths,
                            'weaknesses': o.weaknesses,
                        }
                        for o in theme_outputs
                    ],
                    'strengths_count': sum(len(o.strengths) for o in theme_outputs),
                    'weaknesses_count': sum(len(o.weaknesses) for o in theme_outputs),
                },
            }

            # Step 7: 汇总strengths/weaknesses
            all_strengths = []
            all_weaknesses = []
            for output in theme_outputs:
                all_strengths.extend(output.strengths)
                all_weaknesses.extend(output.weaknesses)

            return {
                'success': True,
                'paper_id': paper.paper_id,
                'signature': signature.model_dump() if signature else {},
                'pred_strengths': all_strengths,
                'pred_weaknesses': all_weaknesses,
                'gt_strengths': paper_data['gt_strengths'],
                'gt_weaknesses': paper_data['gt_weaknesses'],
                'gt_decision': paper_data['gt_decision'],
                'gt_rating': paper_data['gt_rating'],
                'primary_area': paper_data['primary_area'],
                'retrieval_stats': {
                    'cases': len(bundle.similar_paper_cases),
                    'policy_cards': len(bundle.policy_cards),
                    'critique_cards': len(bundle.critique_cases),
                    'failure_cards': len(bundle.failure_cards),
                },
                'trace': trace,  # 详细trace记录
            }

        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] Pipeline failed for {paper.paper_id}: {e}")
            return {
                'success': False,
                'paper_id': paper.paper_id,
                'error': str(e),
                'trace': trace,
            }

    def _run_theme_agents(self, paper: Paper, criteria: list[Criterion], policy_cards, critique_cards) -> list[ThemeOutput]:
        """运行theme agents"""
        # 核心 themes
        themes = ["Clarity", "Quality", "Originality", "Significance", "Experiments"]

        outputs = []
        for theme in themes:
            # 分配criteria给theme
            theme_criteria = [c for c in criteria if c.theme.lower() == theme.lower()]
            if not theme_criteria:
                # 借用其他criteria
                theme_criteria = criteria[:2]

            try:
                output = self.theme_agent.review(paper, theme_criteria, policy_cards, critique_cards)
                outputs.append(output)
            except Exception as e:
                logger.warning(f"[Worker {self.worker_id}] Theme {theme} failed: {e}")
                # 使用空输出
                outputs.append(ThemeOutput(
                    theme=theme,
                    strengths=[],
                    weaknesses=[],
                    severity_tags=[],
                    criteria_used=[],
                ))

        return outputs


def distill_from_result(result: dict, llm: LLMClient | None = None) -> list[ExperienceCard]:
    """从结果提取经验卡片（含对比学习）"""
    cards = []

    if not result['success']:
        return cards

    paper_id = result['paper_id']
    primary_area = result.get('primary_area', '')
    venue_id = 'ICLR'
    year = result.get('year', 2024)

    pred_strengths = result.get('pred_strengths', [])
    pred_weaknesses = result.get('pred_weaknesses', [])
    gt_strengths = result.get('gt_strengths', [])
    gt_weaknesses = result.get('gt_weaknesses', [])

    # 1. 对比学习（核心改进！）
    # 修复：检查长度而不是直接布尔判断（避免numpy array歧义）
    if llm and len(pred_strengths) > 0 and len(gt_strengths) > 0:
        comparison_learner = ComparisonLearner(llm)

        paper_info = {
            'paper_id': paper_id,
            'primary_area': primary_area,
            'venue_id': venue_id,
            'year': year,
            'gt_decision': result.get('gt_decision', ''),
            'pred_weaknesses': pred_weaknesses,
            'gt_weaknesses': gt_weaknesses,
        }

        learned_cards = comparison_learner.learn(
            pred_strengths=pred_strengths,
            pred_weaknesses=pred_weaknesses,
            gt_strengths=gt_strengths,
            gt_weaknesses=gt_weaknesses,
            paper_info=paper_info,
        )
        cards.extend(learned_cards)
        result['learned_cards'] = len(learned_cards)  # 记录对比学习产出

        # 添加对比学习trace
        if 'trace' not in result:
            result['trace'] = {}
        result['trace']['step7_comparison_learning'] = {
            'input': {
                'pred_strengths': pred_strengths,
                'pred_weaknesses': pred_weaknesses,
                'gt_strengths': gt_strengths,
                'gt_weaknesses': gt_weaknesses,
            },
            'output': {
                'learned_cards_count': len(learned_cards),
                'learned_cards': [
                    {
                        'kind': c.kind,
                        'theme': c.theme,
                        'content': c.content,
                        'utility': c.utility,
                        'confidence': c.confidence,
                        'source_ids': c.source_ids,
                        'metadata': c.metadata,
                    }
                    for c in learned_cards
                ],
            },
        }

    # 2. 基础cards：从GT提取（作为补充）
    gt_strengths_text = [
        s.get('value', s) if isinstance(s, dict) else s
        for s in gt_strengths
    ]
    gt_weaknesses_text = [
        w.get('value', w) if isinstance(w, dict) else w
        for w in gt_weaknesses
    ]

    # 从GT weaknesses提取 failure cards（仅reject论文）
    is_reject = 'reject' in str(result.get('gt_decision', '')).lower()

    if is_reject:
        for text in gt_weaknesses_text[:3]:
            if len(text) > 25 and is_generalizable(text):
                card = ExperienceCard(
                    card_id=str(uuid.uuid4()),
                    kind="failure",
                    scope="domain" if primary_area else "venue",
                    venue_id=venue_id,
                    theme=infer_theme(text),
                    content=f"⚠️ {normalize_text(text)}",
                    trigger=[],
                    utility=0.4,  # 基础utility较低，对比学习的failure card utility更高(0.8)
                    confidence=0.5,
                    use_count=0,
                    source_ids=[paper_id],
                    created_at=datetime.utcnow(),
                    active=True,
                    source_trace={"source": "gt_direct"},
                    metadata={
                        "memory_year": year,
                        "memory_type": "gt_failure",
                        "memory_domain": primary_area,
                    },
                )
                cards.append(card)

    return cards


def is_generalizable(text: str) -> bool:
    """检查是否可迁移"""
    text_lower = text.lower().strip()
    if len(text_lower) < 25:
        return False
    non_transferable = [
        "this paper", "the paper", "our method", "our approach",
        "authors", "figure", "table", "appendix", "section",
    ]
    return not any(marker in text_lower for marker in non_transferable)


def normalize_text(text: str) -> str:
    """归一化"""
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


def process_paper(args):
    """处理单篇论文（用于多进程）"""
    paper_data, config, token, worker_id = args

    # 创建简化pipeline
    pipeline = SimplifiedPipeline(config, token, worker_id)

    # 运行
    result = pipeline.run(paper_data)

    # 提取经验卡片（传入llm进行对比学习）
    cards = distill_from_result(result, llm=pipeline.llm)
    result['cards_generated'] = len(cards)

    return result, cards


class CustomEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理datetime、numpy和dict中的特殊类型"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        if pd.isna(obj):
            return None
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return super().default(obj)


def save_results(results: list[dict], output_path: str):
    """保存结果"""
    # 统计
    stats = {
        'total': len(results),
        'success': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'avg_pred_strengths': 0.0,
        'avg_pred_weaknesses': 0.0,
        'avg_gt_strengths': 0.0,
        'avg_gt_weaknesses': 0.0,
        'total_cards': sum(r.get('cards_generated', 0) for r in results),
    }

    if stats['success'] > 0:
        successful = [r for r in results if r['success']]
        stats['avg_pred_strengths'] = sum(len(r.get('pred_strengths', [])) for r in successful) / stats['success']
        stats['avg_pred_weaknesses'] = sum(len(r.get('pred_weaknesses', [])) for r in successful) / stats['success']
        stats['avg_gt_strengths'] = sum(len(r.get('gt_strengths', [])) for r in successful) / stats['success']
        stats['avg_gt_weaknesses'] = sum(len(r.get('gt_weaknesses', [])) for r in successful) / stats['success']

    output = {
        'stats': stats,
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=CustomEncoder)

    return stats


def save_cards(cards: list[ExperienceCard], output_path: str):
    """保存经验卡片到训练专用的memory目录"""
    # 写入训练专用的记忆库（不覆盖现有memory）
    memory_base = Path("data/processed/memory")

    # 使用 ICLR_2024_learned 目录存放训练产生的cards
    memory_dir = memory_base / "ICLR_2024_learned"
    memory_dir.mkdir(parents=True, exist_ok=True)

    cards_file = memory_dir / "policy_cards.jsonl"  # 统一文件名，支持持续读写

    # 追加写入
    with open(cards_file, 'a') as f:
        for card in cards:
            f.write(json.dumps(serialize_card(card), ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(cards)} learned cards to {cards_file}")


def serialize_card(card: ExperienceCard) -> dict:
    """序列化卡片"""
    data = card.model_dump()
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()
    return data


def compute_card_outcomes(result: dict) -> dict[str, str]:
    """根据对比学习结果判断本轮检索到的卡片的贡献

    原理：
    - 如果对比学习发现 missed_patterns 很少 → 检索到的卡片帮了忙 → positive
    - 如果 missed_patterns 很多 → 检索到的卡片不够 → 所涉及主题的卡片 neutral/negative
    - 如果有 validated_patterns → 那些主题的卡片有效 → positive

    Returns:
        {card_id: "positive"|"negative"|"neutral"}
    """
    outcomes: dict[str, str] = {}

    trace = result.get('trace', {})
    step7 = trace.get('step7_comparison_learning', {})
    learned_cards = step7.get('output', {}).get('learned_cards', [])

    if not learned_cards:
        return outcomes

    # 统计对比学习产出
    missed_count = sum(1 for c in learned_cards if c.get('kind') == 'failure')
    bias_count = sum(1 for c in learned_cards if c.get('kind') == 'critique')
    validated_count = sum(1 for c in learned_cards if c.get('kind') == 'strength')

    # 检索到的卡片ID（从retrieval trace）
    retrieval_stats = result.get('retrieval_stats', {})
    # 从trace中获取policy/critique/failure卡片
    step2 = trace.get('step2_retrieval', {}).get('output', {})
    policy_card_ids = step2.get('policy_card_ids', [])
    critique_card_ids = step2.get('critique_card_ids', [])
    failure_card_ids = step2.get('failure_card_ids', [])
    all_retrieved_ids = policy_card_ids + critique_card_ids + failure_card_ids

    # 总体质量判断
    if missed_count == 0 and validated_count >= 1:
        # 无遗漏且有验证 → 检索到的卡片帮助了正确评价
        for card_id in all_retrieved_ids:
            outcomes[card_id] = "positive"
    elif missed_count == 0:
        # 无遗漏但也没有验证
        for card_id in all_retrieved_ids:
            outcomes[card_id] = "neutral"
    elif missed_count >= 2:
        # 多项遗漏 → 检索到的卡片不够
        for card_id in all_retrieved_ids:
            outcomes[card_id] = "negative"
    else:
        # 少量遗漏 → neutral
        for card_id in all_retrieved_ids:
            outcomes[card_id] = "neutral"

    return outcomes


def main():
    parser = argparse.ArgumentParser(description="Simplified parallel training")
    parser.add_argument('--tokens', default='tokens.txt', help='Tokens file')
    parser.add_argument('--config', default='configs/iclr.yaml', help='Config file')
    parser.add_argument('--parquet', default='data/merged_cases/ICLR_2024_merged.parquet', help='Parquet file')
    parser.add_argument('--workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--limit', type=int, default=None, help='Limit papers')
    parser.add_argument('--skip_processed', action='store_true', help='Skip already processed papers')
    parser.add_argument('--output', default='data/processed/train_simplified_results.json', help='Output file')
    args = parser.parse_args()

    # 加载tokens
    tokens = load_tokens(args.tokens)
    logger.info(f"Loaded {len(tokens)} tokens")

    if len(tokens) < args.workers:
        logger.warning(f"Only {len(tokens)} tokens, reducing workers to {len(tokens)}")
        args.workers = len(tokens)

    # 加载config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 加载论文（可选跳过已处理）
    papers = load_papers_with_gt(args.parquet, limit=args.limit, skip_processed=args.skip_processed)
    logger.info(f"Loaded {len(papers)} papers")

    # 准备任务
    tasks = []
    for i, paper_data in enumerate(papers):
        worker_id = i % args.workers
        token = tokens[worker_id]
        tasks.append((paper_data, config, token, worker_id))

    logger.info(f"Starting {args.workers} workers for {len(tasks)} tasks")

    # 运行多进程
    results = []
    all_cards = []

    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_paper, task): task for task in tasks}

        completed = 0
        for future in as_completed(futures):
            result, cards = future.result()
            results.append(result)
            all_cards.extend(cards)

            completed += 1
            if completed % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - completed) / rate / 60 if rate > 0 else 0
                logger.info(f"Completed {completed}/{len(tasks)}, rate={rate:.2f}/s, ETA={eta:.1f}min")

    elapsed = (datetime.now() - start_time).total_seconds()

    # 保存结果
    stats = save_results(results, args.output)
    logger.info(f"Results saved to {args.output}")

    # 保存卡片到learned记忆目录（不覆盖预制记忆）
    if all_cards:
        save_cards(all_cards, "data/processed/memory/ICLR_2024_learned/policy_cards.jsonl")

    # === Effectiveness Tracking: 根据对比学习结果更新卡片质量 ===
    all_outcomes: dict[str, str] = {}
    for result in results:
        if result.get('success'):
            outcomes = compute_card_outcomes(result)
            all_outcomes.update(outcomes)

    if all_outcomes:
        positive_count = sum(1 for v in all_outcomes.values() if v == "positive")
        negative_count = sum(1 for v in all_outcomes.values() if v == "negative")
        neutral_count = sum(1 for v in all_outcomes.values() if v == "neutral")
        logger.info(
            "Effectiveness Tracking: %d outcomes (%d positive, %d negative, %d neutral)",
            len(all_outcomes), positive_count, negative_count, neutral_count,
        )

        # 尝试应用反馈到VectorMemoryStore（如果可用）
        try:
            from storage.vector_memory_store import VectorMemoryStore
            learned_path = Path("data/processed/memory/ICLR_2024_learned/agent_memories.jsonl")
            if learned_path.exists():
                store = VectorMemoryStore(learned_path)
                stats = store.apply_feedback(all_outcomes, "batch_feedback")
                logger.info(
                    "Feedback applied: %d updated, %d promoted, %d degraded, %d retired",
                    stats["updated"], stats["promoted"], stats["degraded"], stats["retired"],
                )
        except Exception as e:
            logger.warning("Failed to apply feedback: %s", e)

    # 打印统计
    logger.info("\n" + "=" * 70)
    logger.info("训练统计")
    logger.info("=" * 70)
    logger.info(f"  总论文数: {stats['total']}")
    logger.info(f"  成功数: {stats['success']}")
    logger.info(f"  失败数: {stats['failed']}")
    logger.info(f"  平均pred strengths: {stats['avg_pred_strengths']:.1f}")
    logger.info(f"  平均pred weaknesses: {stats['avg_pred_weaknesses']:.1f}")
    logger.info(f"  平均GT strengths: {stats['avg_gt_strengths']:.1f}")
    logger.info(f"  平均GT weaknesses: {stats['avg_gt_weaknesses']:.1f}")
    logger.info(f"  生成卡片数: {stats['total_cards']}")
    logger.info(f"  总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"  平均速度: {len(tasks)/elapsed:.2f} papers/s")
    logger.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()