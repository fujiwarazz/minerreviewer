#!/usr/bin/env python
"""
批量测试脚本 - 使用 reviewer2 数据集测试同会议和跨会议论文

输出 JSON 包含：
- GT decision 和 Pred decision
- 召回的相似文章
- 使用的记忆经验（policy cards）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.types import Paper
from common.utils import read_yaml
from pipeline.review_pipeline import ReviewPipeline

DATA_ROOT = Path("/mnt/data/zzh/datasets/reviewer2")


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

    # meta_review 可能是 dict 或 str
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

    # 提取 title 和 abstract（如果有）
    title = data.get('title', '')
    abstract = data.get('abstract', '')

    # 提取 decision
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
        'title': title,
        'abstract': abstract,
        'decision': decision,
        'file_path': str(file_path),
    }


def get_test_papers(venue: str, year: int, n_accept: int = 5, n_reject: int = 5, seed: int = 42) -> list[dict]:
    """获取测试论文"""
    import random
    random.seed(seed)

    # reviewer2 中 NeurIPS 用 NIPS 作为目录名
    data_venue = 'NIPS' if venue == 'NeurIPS' else venue
    venue_path = DATA_ROOT / data_venue
    if not venue_path.exists():
        print(f"Venue path not found: {venue_path}")
        return []

    # 找所有 review 文件
    # NIPS_2022 是目录名格式
    pattern = f"{data_venue}_{year}"
    files = list(venue_path.glob(f'**/{pattern}*_review/*.json'))
    if not files:
        files = list(venue_path.glob(f'**/{pattern}/*_review/*.json'))

    print(f"Found {len(files)} files for {venue} {year}")

    # 解析并分类
    accept_papers = []
    reject_papers = []

    for f in files:
        result = parse_review_file(f)
        if result and result['decision']:
            if result['decision'] == 'Accept':
                accept_papers.append(result)
            elif result['decision'] == 'Reject':
                reject_papers.append(result)

    print(f"Accept: {len(accept_papers)}, Reject: {len(reject_papers)}")

    # 采样
    sampled_accept = random.sample(accept_papers, min(n_accept, len(accept_papers)))
    sampled_reject = random.sample(reject_papers, min(n_reject, len(reject_papers)))

    return sampled_accept + sampled_reject


def run_test(pipeline: ReviewPipeline, paper_info: dict) -> dict:
    """运行单个论文的 review 并返回详细结果"""
    paper = Paper(
        paper_id=paper_info['paper_id'],
        title=paper_info.get('title', ''),
        abstract=paper_info.get('abstract', ''),
        venue_id=paper_info['venue'],
        year=paper_info['year'],
        authors=[],
        fulltext=None,
    )

    target_year = paper_info['year'] + 1  # 允许使用同年的案例（测试集本身不在 cases 里）
    target_venue = paper_info['venue']

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
    for case in bundle.similar_paper_cases:
        similar_cases_info.append({
            "paper_id": case.paper_id,
            "venue_id": case.venue_id,
            "year": case.year,
            "rating": case.rating,
            "decision": case.decision,
            "top_strengths": case.top_strengths[:2] if case.top_strengths else [],
            "top_weaknesses": case.top_weaknesses[:2] if case.top_weaknesses else [],
        })

    # 提取 policy cards 信息（只取前 5 个）
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

    gt_decision = paper_info['decision']
    gt_binary = 'accept' if gt_decision.lower() == 'accept' else 'reject'
    pred_binary = 'accept' if arbiter_output.decision_recommendation and 'accept' in arbiter_output.decision_recommendation.lower() else 'reject'

    return {
        "paper_id": paper.paper_id,
        "title": paper.title[:100] + "..." if len(paper.title) > 100 else paper.title,
        "venue": target_venue,
        "year": target_year,
        "ground_truth": gt_decision,
        "prediction": {
            "rating": round(arbiter_output.raw_rating, 2),
            "decision": arbiter_output.decision_recommendation,
            "acceptance_likelihood": round(arbiter_output.acceptance_likelihood, 4) if arbiter_output.acceptance_likelihood else None,
        },
        "match": gt_binary == pred_binary,
        "similar_cases": similar_cases_info,
        "similar_cases_stats": stats,
        "policy_cards_count": len(bundle.policy_cards),
        "policy_cards_sample": policy_cards_info,
        "strengths": arbiter_output.strengths[:3],
        "weaknesses": arbiter_output.weaknesses[:3],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/iclr.yaml")
    parser.add_argument("--output-dir", default="data/eval_results")
    parser.add_argument("--n-samples", type=int, default=10, help="每类（Accept/Reject）的样本数")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 pipeline
    print("Initializing pipeline...")
    pipeline = ReviewPipeline(args.config)
    print("Pipeline loaded!\n")

    all_results = {
        "config": args.config,
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }

    # ========== 同会议测试：ICLR 2023 ==========
    print("="*60)
    print("Testing SAME VENUE: ICLR 2023")
    print("="*60)

    papers = get_test_papers('ICLR', 2023, n_accept=args.n_samples//2, n_reject=args.n_samples//2)
    results = []
    correct = 0

    for i, paper_info in enumerate(papers):
        print(f"\n[{i+1}/{len(papers)}] Testing {paper_info['paper_id']} (GT: {paper_info['decision']})")
        try:
            result = run_test(pipeline, paper_info)
            results.append(result)
            if result['match']:
                correct += 1
            print(f"  -> Pred: {result['prediction']['decision']} ({result['prediction']['rating']}), Match: {result['match']}")
        except Exception as e:
            print(f"  -> Error: {e}")
            import traceback
            traceback.print_exc()

    all_results["tests"]["same_venue_iclr_2023"] = {
        "description": "Same venue test: ICLR 2023 papers with ICLR memory",
        "total": len(results),
        "correct": correct,
        "accuracy": round(correct / len(results), 4) if results else 0,
        "results": results,
    }
    print(f"\nICLR 2023 Accuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")

    # ========== 跨会议测试：NeurIPS 2022 ==========
    print("\n" + "="*60)
    print("Testing CROSS VENUE: NeurIPS 2022 (with ICLR memory)")
    print("="*60)

    papers = get_test_papers('NeurIPS', 2022, n_accept=args.n_samples//2, n_reject=args.n_samples//2)
    results = []
    correct = 0

    for i, paper_info in enumerate(papers):
        print(f"\n[{i+1}/{len(papers)}] Testing {paper_info['paper_id']} (GT: {paper_info['decision']})")
        try:
            result = run_test(pipeline, paper_info)
            results.append(result)
            if result['match']:
                correct += 1
            print(f"  -> Pred: {result['prediction']['decision']} ({result['prediction']['rating']}), Match: {result['match']}")
        except Exception as e:
            print(f"  -> Error: {e}")
            import traceback
            traceback.print_exc()

    all_results["tests"]["cross_venue_neurips_2022"] = {
        "description": "Cross venue test: NeurIPS 2022 papers with ICLR memory",
        "total": len(results),
        "correct": correct,
        "accuracy": round(correct / len(results), 4) if results else 0,
        "results": results,
    }
    print(f"\nNeurIPS 2022 Accuracy: {correct}/{len(results)} = {correct/max(len(results),1)*100:.1f}%")

    # 保存结果
    output_path = output_dir / f"batch_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Same venue (ICLR 2023): {all_results['tests']['same_venue_iclr_2023']['accuracy']*100:.1f}%")
    print(f"Cross venue (NeurIPS 2022): {all_results['tests']['cross_venue_neurips_2022']['accuracy']*100:.1f}%")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()