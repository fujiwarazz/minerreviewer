#!/usr/bin/env python
"""
测试 DeepReview-13K 记忆库对 ICLR 2024 论文的评审效果

使用 DeepReview-13K 作为记忆库，测试 ICLR 2024 论文
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.types import Paper
from common.utils import read_yaml
from pipeline.review_pipeline import ReviewPipeline


def load_iclr2024_papers(path: str, limit: int = 50, seed: int = 42) -> list[dict]:
    """加载 ICLR 2024 论文"""
    random.seed(seed)

    with open(path) as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} ICLR 2024 papers")

    # 随机采样
    if len(papers) > limit:
        papers = random.sample(papers, limit)

    return papers


def load_deepreview_ground_truth(jsonl_path: str, paper_ids: list[str]) -> dict[str, dict]:
    """从 DeepReview-13K 加载 ground truth（如果测试论文来自该数据集）"""
    gt_map = {}
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            pid = data.get('id')
            if pid in paper_ids:
                gt_map[pid] = {
                    'decision': data.get('decision'),
                    'rating': data.get('rating'),
                    'primary_area': data.get('primary_area'),
                }
    return gt_map


def run_test(pipeline: ReviewPipeline, paper_info: dict, target_year: int = 2024) -> dict:
    """运行单个论文的 review 并返回详细结果"""
    paper = Paper(
        paper_id=paper_info.get('paper_id', paper_info.get('id', 'unknown')),
        title=paper_info.get('title', ''),
        abstract=paper_info.get('abstract', ''),
        venue_id='ICLR',
        year=2024,
        authors=paper_info.get('authors', []),
        fulltext=paper_info.get('fulltext'),
    )

    # 调用内部方法获取完整的 bundle 和 output
    signature = pipeline._parse_paper(paper)
    bundle = pipeline._retrieve_multi_channel(paper, signature, target_year)
    content_criteria, policy_criteria = pipeline._mine_criteria(paper, bundle, target_year)
    activated = pipeline._plan_criteria(signature, bundle, content_criteria, policy_criteria)
    criteria = pipeline._rewrite_criteria(paper, activated)
    theme_outputs = pipeline._run_theme_agents(paper, criteria)
    arbiter_output = pipeline._aggregate(
        theme_outputs, bundle.policy_cards, policy_criteria, bundle.venue_policy,
        similar_cases=bundle.similar_paper_cases
    )

    # 提取相似案例信息
    similar_cases_info = []
    deepreview_cases = []
    for case in bundle.similar_paper_cases:
        case_info = {
            "paper_id": case.paper_id,
            "venue_id": case.venue_id,
            "year": case.year,
            "rating": case.rating,
            "decision": case.decision,
            "primary_area": getattr(case, 'primary_area', None),
            "top_strengths": case.top_strengths[:2] if case.top_strengths else [],
            "top_weaknesses": case.top_weaknesses[:2] if case.top_weaknesses else [],
        }
        similar_cases_info.append(case_info)
        if case.venue_id == 'DeepReview':
            deepreview_cases.append(case_info)

    # 提取 policy cards 信息
    policy_cards_info = []
    for card in bundle.policy_cards[:5]:
        policy_cards_info.append({
            "card_id": card.card_id,
            "theme": card.theme,
            "kind": card.kind,
            "content": card.content[:100] + "..." if len(card.content) > 100 else card.content,
        })

    # 计算相似案例统计
    import statistics
    valid_cases = [c for c in bundle.similar_paper_cases if c.rating]
    stats = {}
    if valid_cases:
        ratings = [c.rating for c in valid_cases]
        stats = {
            "mean_rating": round(statistics.mean(ratings), 2),
            "median_rating": round(statistics.median(ratings), 2),
            "accept_count": sum(1 for c in valid_cases if c.decision and "accept" in c.decision.lower()),
            "reject_count": sum(1 for c in valid_cases if c.decision and "reject" in c.decision.lower()),
        }

    return {
        "paper_id": paper.paper_id,
        "title": paper.title[:100] + "..." if len(paper.title) > 100 else paper.title,
        "primary_area": paper_info.get('primary_area'),
        "ground_truth": paper_info.get('decision'),
        "prediction": {
            "rating": round(arbiter_output.raw_rating, 2),
            "decision": arbiter_output.decision_recommendation,
            "acceptance_likelihood": round(arbiter_output.acceptance_likelihood, 4) if arbiter_output.acceptance_likelihood else None,
        },
        "similar_cases_count": len(bundle.similar_paper_cases),
        "deepreview_cases_count": len(deepreview_cases),
        "similar_cases": similar_cases_info[:5],
        "similar_cases_stats": stats,
        "policy_cards_count": len(bundle.policy_cards),
        "policy_cards_sample": policy_cards_info,
        "strengths": arbiter_output.strengths[:3],
        "weaknesses": arbiter_output.weaknesses[:3],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/iclr.yaml")
    parser.add_argument("--papers", default="data/processed/papers__ICLR.cc_2024_Conference.json")
    parser.add_argument("--output-dir", default="data/eval_results")
    parser.add_argument("--n-samples", type=int, default=50, help="测试论文数量")
    parser.add_argument("--target-year", type=int, default=2024, help="目标年份（记忆库使用该年份之前的数据）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 pipeline
    print("Initializing pipeline...")
    print(f"Config: {args.config}")
    pipeline = ReviewPipeline(args.config)
    print("Pipeline loaded!\n")

    # 检查 DeepReview 记忆库
    if pipeline.deepreview_store:
        print(f"DeepReview memory enabled: {len(pipeline.deepreview_store.cases)} cases")
        print(f"Primary areas: {len(pipeline.deepreview_store.list_areas())}")
    else:
        print("WARNING: DeepReview memory NOT enabled!")

    # 检查主记忆库
    main_cases = 0
    if pipeline.case_store:
        if hasattr(pipeline.case_store, 'cases'):
            main_cases = len(pipeline.case_store.cases)
        elif hasattr(pipeline.case_store, '_stores'):
            for store_id, store in pipeline.case_store._stores.items():
                if hasattr(store, 'cases'):
                    main_cases += len(store.cases)
    print(f"Main memory cases: {main_cases}")
    print()

    # 加载测试论文
    papers = load_iclr2024_papers(args.papers, limit=args.n_samples, seed=args.seed)

    # 运行测试
    results = []
    correct = 0
    has_gt = any(p.get('decision') for p in papers)

    print("="*60)
    print(f"Testing {len(papers)} ICLR 2024 papers with DeepReview memory")
    print("="*60)

    for i, paper_info in enumerate(papers):
        paper_id = paper_info.get('paper_id', paper_info.get('id', 'unknown'))
        gt_decision = paper_info.get('decision', 'N/A')

        print(f"\n[{i+1}/{len(papers)}] {paper_id}")
        print(f"  Title: {paper_info.get('title', 'N/A')[:60]}...")
        if gt_decision != 'N/A':
            print(f"  GT: {gt_decision}")

        try:
            result = run_test(pipeline, paper_info, target_year=args.target_year)
            results.append(result)

            pred_decision = result['prediction']['decision'] or 'N/A'
            print(f"  Pred: {pred_decision} (rating: {result['prediction']['rating']})")
            print(f"  Similar cases: {result['similar_cases_count']} (DeepReview: {result['deepreview_cases_count']})")

            if has_gt and gt_decision != 'N/A':
                gt_binary = 'accept' if 'accept' in gt_decision.lower() else 'reject'
                pred_binary = 'accept' if pred_decision and 'accept' in pred_decision.lower() else 'reject'
                match = gt_binary == pred_binary
                if match:
                    correct += 1
                print(f"  Match: {match}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "paper_id": paper_id,
                "error": str(e),
            })

    # 汇总结果
    summary = {
        "config": args.config,
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(papers),
        "target_year": args.target_year,
        "seed": args.seed,
        "deepreview_enabled": pipeline.deepreview_store is not None,
        "deepreview_cases": len(pipeline.deepreview_store.cases) if pipeline.deepreview_store else 0,
        "main_cases": len(pipeline.case_store.cases) if pipeline.case_store else 0,
        "results": results,
    }

    if has_gt:
        valid_results = [r for r in results if 'error' not in r]
        summary["accuracy"] = round(correct / len(valid_results), 4) if valid_results else 0
        summary["correct"] = correct
        summary["total_valid"] = len(valid_results)
        print(f"\n{'='*60}")
        print(f"Accuracy: {correct}/{len(valid_results)} = {correct/len(valid_results)*100:.1f}%")

    # 保存结果
    output_path = output_dir / f"deepreview_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()