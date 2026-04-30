#!/usr/bin/env python
"""
评估脚本（不更新记忆）

流程与train相同，但不进行对比学习和记忆更新：
1. Parse paper → signature
2. Retrieval (cases + cards)
3. Mine criteria
4. Theme agents → strengths/weaknesses
5. 汇总strengths/weaknesses
6. 计算评估指标（覆盖率、一致性等）

Usage:
    python scripts/eval_simplified.py --tokens tokens.txt --config configs/iclr.yaml --limit 50 --workers 10
    python scripts/eval_simplified.py --tokens tokens.txt --config configs/iclr.yaml --limit 50 --memory_path data/processed/memory/ICLR_2024_learned/policy_cards.jsonl
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
from storage.doc_store import DocStore

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


def load_papers_by_ids(parquet_path: str, paper_ids: list[str]) -> list[dict]:
    """加载指定paper_id的论文"""
    df = pd.read_parquet(parquet_path)
    df_selected = df[df['paper_id'].isin(paper_ids)]
    logger.info(f"Found {len(df_selected)} papers from {len(paper_ids)} IDs")

    papers = []
    for _, row in df_selected.iterrows():
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

    def __init__(self, config: dict, token: str, worker_id: int, use_memory: bool = True):
        self.config = config
        self.worker_id = worker_id
        self.use_memory = use_memory

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

        # 根据use_memory参数决定是否加载记忆
        if use_memory:
            self.learned_memory_store = MemoryStore(learned_memory_path)
            logger.info(f"Loaded {len(self.learned_memory_store.cards)} learned cards from previous training")
        else:
            # B组：空记忆
            self.learned_memory_store = MemoryStore(Path("/tmp/empty_memory.jsonl"))
            logger.info(f"Using empty memory (B group for AB testing)")

        # 加载2024的cases用于评估（不能用2025自己的数据）
        # 评估模式：加载历史cases
        self.case_store = MultiCaseStore(
            registry=self.registry,
            embedding_client=self.embedding_client,
            skip_load=False,  # 评估模式需要加载历史cases
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
    paper_data, config, token, worker_id, use_memory = args

    # 创建简化pipeline（不更新记忆）
    pipeline = SimplifiedPipeline(config, token, worker_id, use_memory=use_memory)

    # 运行（不进行对比学习，不生成卡片）
    result = pipeline.run(paper_data)

    return result


def compute_metrics(results: list[dict]) -> dict:
    """计算评估指标"""
    import numpy as np

    successful = [r for r in results if r['success']]

    if not successful:
        return {}

    metrics = {
        'total': len(results),
        'success': len(successful),
        'failed': len(results) - len(successful),
        'avg_pred_strengths': np.mean([len(r.get('pred_strengths', [])) for r in successful]),
        'avg_pred_weaknesses': np.mean([len(r.get('pred_weaknesses', [])) for r in successful]),
        'avg_gt_strengths': np.mean([len(r.get('gt_strengths', [])) for r in successful]),
        'avg_gt_weaknesses': np.mean([len(r.get('gt_weaknesses', [])) for r in successful]),
    }

    # 关键词覆盖率
    def get_keywords(text):
        if isinstance(text, dict):
            text = text.get('value', str(text))
        words = str(text).lower().split()
        return set([w for w in words if len(w) > 5 and w.isalpha()])

    coverages_s = []
    coverages_w = []
    for r in successful:
        gt_kw_s = set()
        gt_kw_w = set()
        for gt in r.get('gt_strengths', []):
            gt_kw_s.update(get_keywords(gt))
        for gt in r.get('gt_weaknesses', []):
            gt_kw_w.update(get_keywords(gt))

        pred_kw_s = set()
        pred_kw_w = set()
        for pred in r.get('pred_strengths', []):
            pred_kw_s.update(get_keywords(pred))
        for pred in r.get('pred_weaknesses', []):
            pred_kw_w.update(get_keywords(pred))

        if gt_kw_s:
            coverages_s.append(len(pred_kw_s & gt_kw_s) / len(gt_kw_s))
        if gt_kw_w:
            coverages_w.append(len(pred_kw_w & gt_kw_w) / len(gt_kw_w))

    metrics['keyword_coverage_strengths'] = np.mean(coverages_s) if coverages_s else 0
    metrics['keyword_coverage_weaknesses'] = np.mean(coverages_w) if coverages_w else 0

    # 决定一致性
    valid_dec = [r for r in successful if r.get('gt_decision') and str(r.get('gt_decision', '')) != 'nan']
    correct = 0
    for r in valid_dec:
        pred_dec = 'Reject' if len(r.get('pred_weaknesses', [])) > len(r.get('pred_strengths', [])) else 'Accept'
        gt_dec = 'Accept' if 'Accept' in str(r.get('gt_decision', '')) else 'Reject'
        if pred_dec == gt_dec:
            correct += 1

    metrics['decision_accuracy'] = correct / len(valid_dec) if valid_dec else 0
    metrics['decision_count'] = len(valid_dec)

    # 输出质量分布
    zero = len([r for r in successful if len(r.get('pred_strengths', [])) == 0])
    good = len([r for r in successful if len(r.get('pred_strengths', [])) >= 3])
    metrics['zero_output_count'] = zero
    metrics['good_output_count'] = good
    metrics['good_output_ratio'] = good / len(successful) if successful else 0

    return metrics


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
    """保存结果（eval模式，不保存卡片）"""
    # 计算完整评估指标
    metrics = compute_metrics(results)

    output = {
        'metrics': metrics,
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=CustomEncoder)

    logger.info(f"Results saved to {output_path}")
    return metrics


def serialize_card(card: ExperienceCard) -> dict:
    """序列化卡片"""
    data = card.model_dump()
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluation script (no memory update)")
    parser.add_argument('--tokens', default='tokens.txt', help='Tokens file')
    parser.add_argument('--config', default='configs/iclr.yaml', help='Config file')
    parser.add_argument('--parquet', default='data/merged_cases/ICLR_2024_merged.parquet', help='Parquet file')
    parser.add_argument('--workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--limit', type=int, default=None, help='Limit papers')
    parser.add_argument('--paper_ids', type=str, default=None, help='Comma-separated paper IDs to evaluate')
    parser.add_argument('--output', default='data/processed/eval_results.json', help='Output file')
    parser.add_argument('--memory_path', default=None, help='Custom memory path to use')
    parser.add_argument('--no_memory', action='store_true', help='Disable memory (B group for AB testing)')
    args = parser.parse_args()

    use_memory = not args.no_memory

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

    # 如果指定了memory_path，加载指定记忆
    if args.memory_path:
        logger.info(f"Using custom memory: {args.memory_path}")

    if args.no_memory:
        logger.info("=== B组：空记忆（AB测试）===")
    else:
        logger.info("=== A组：使用learned记忆（AB测试）===")

    # 加载论文
    if args.paper_ids:
        paper_ids = [pid.strip() for pid in args.paper_ids.split(',')]
        papers = load_papers_by_ids(args.parquet, paper_ids)
    else:
        papers = load_papers_with_gt(args.parquet, limit=args.limit)
    logger.info(f"Loaded {len(papers)} papers")

    # 准备任务
    tasks = []
    for i, paper_data in enumerate(papers):
        worker_id = i % args.workers
        token = tokens[worker_id]
        tasks.append((paper_data, config, token, worker_id, use_memory))

    logger.info(f"Starting {args.workers} workers for {len(tasks)} tasks")

    # 运行多进程
    results = []

    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_paper, task): task for task in tasks}

        completed = 0
        for future in as_completed(futures):
            result = future.result()  # 只返回result，不返回cards
            results.append(result)

            completed += 1
            if completed % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - completed) / rate / 60 if rate > 0 else 0
                logger.info(f"Completed {completed}/{len(tasks)}, rate={rate:.2f}/s, ETA={eta:.1f}min")

    elapsed = (datetime.now() - start_time).total_seconds()

    # 保存结果
    metrics = save_results(results, args.output)

    # 打印评估指标
    logger.info("\n" + "=" * 70)
    logger.info("评估结果")
    logger.info("=" * 70)
    logger.info(f"  总论文数: {metrics['total']}")
    logger.info(f"  成功数: {metrics['success']}")
    logger.info(f"  失败数: {metrics['failed']}")
    logger.info(f"  平均pred strengths: {metrics['avg_pred_strengths']:.1f}")
    logger.info(f"  平均pred weaknesses: {metrics['avg_pred_weaknesses']:.1f}")
    logger.info(f"  平均GT strengths: {metrics['avg_gt_strengths']:.1f}")
    logger.info(f"  平均GT weaknesses: {metrics['avg_gt_weaknesses']:.1f}")
    logger.info(f"  关键词覆盖率(strengths): {metrics['keyword_coverage_strengths']*100:.1f}%")
    logger.info(f"  关键词覆盖率(weaknesses): {metrics['keyword_coverage_weaknesses']*100:.1f}%")
    logger.info(f"  决定一致性准确率: {metrics['decision_accuracy']*100:.1f}%")
    logger.info(f"  有效决定论文数: {metrics['decision_count']}")
    logger.info(f"  正常输出论文数: {metrics['good_output_count']} ({metrics['good_output_ratio']*100:.1f}%)")
    logger.info(f"  总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"  平均速度: {len(tasks)/elapsed:.2f} papers/s")
    logger.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()