#!/usr/bin/env python
"""
详细日志测试脚本 - 展示完整的 review pipeline 运行链路
"""
from __future__ import annotations

import sys
import json
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from common.types import Paper
from common.utils import read_yaml
from pipeline.review_pipeline import ReviewPipeline
from storage.case_store import CaseStore

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

def print_subsection(title: str):
    print(f"\n--- {title} ---")

def run_detailed_review(config_path: str, paper: Paper, target_year: int = 2024):
    """运行完整的 review pipeline 并输出详细日志"""

    config = read_yaml(config_path)
    pipeline = ReviewPipeline(config_path)

    print_section("INPUT")
    print(f"Paper ID: {paper.paper_id}")
    print(f"Title: {paper.title}")
    print(f"Abstract: {paper.abstract[:300]}...")
    print(f"Venue: {paper.venue_id}, Year: {target_year}")

    # Step 1: Parse Paper
    print_section("STEP 1: PAPER PARSING")
    signature = pipeline._parse_paper(paper)
    print(f"Paper Type: {signature.paper_type}")
    print(f"Domain: {signature.domain}")
    print(f"Tasks: {signature.tasks}")
    print(f"Method Family: {signature.method_family}")

    # Step 2: Multi-channel Retrieval
    print_section("STEP 2: MULTI-CHANNEL RETRIEVAL")
    bundle = pipeline._retrieve_multi_channel(paper, signature, target_year)

    print_subsection("Similar Paper Cases Retrieved")
    if bundle.similar_paper_cases:
        for i, case in enumerate(bundle.similar_paper_cases[:5]):
            print(f"\n  [{i+1}] {case.paper_id}")
            print(f"      Rating: {case.rating}")
            print(f"      Decision: {case.decision}")
            if case.top_strengths:
                print(f"      Strengths: {case.top_strengths[0][:60]}...")
            if case.top_weaknesses:
                print(f"      Weaknesses: {case.top_weaknesses[0][:60]}...")
    else:
        print("  No similar cases found")

    print_subsection("Policy Cards Retrieved")
    if bundle.policy_cards:
        print(f"  Count: {len(bundle.policy_cards)}")
        for i, card in enumerate(bundle.policy_cards[:3]):
            print(f"\n  [{i+1}] Theme: {card.theme}")
            print(f"      Content: {card.content[:80]}...")
    else:
        print("  No policy cards")

    # Step 3: Mine Criteria
    print_section("STEP 3: CRITERIA MINING")
    content_criteria, policy_criteria = pipeline._mine_criteria(paper, bundle, target_year)

    print_subsection(f"Content Criteria ({len(content_criteria)})")
    for c in content_criteria[:5]:
        print(f"  - [{c.theme}] {c.text[:70]}...")

    print_subsection(f"Policy Criteria ({len(policy_criteria)})")
    for c in policy_criteria[:5]:
        print(f"  - [{c.theme}] {c.text[:70]}...")

    # Step 4: Plan Criteria
    print_section("STEP 4: CRITERIA PLANNING")
    activated = pipeline._plan_criteria(signature, bundle, content_criteria, policy_criteria)

    print(f"Total Activated: {len(activated)}")
    by_theme = {}
    for a in activated:
        by_theme[a.theme] = by_theme.get(a.theme, 0) + 1
    print(f"By Theme: {by_theme}")

    print_subsection("Top Activated Criteria")
    for a in activated[:5]:
        print(f"\n  [{a.theme}] Priority: {a.priority}")
        print(f"  Source: {a.source}")
        print(f"  Trigger: {a.trigger_reason}")
        print(f"  Criterion: {a.criterion[:70]}...")

    # Step 5: Rewrite Criteria
    print_section("STEP 5: CRITERIA REWRITING")
    criteria = pipeline._rewrite_criteria(paper, activated)
    print(f"Rewritten Criteria Count: {len(criteria)}")
    for c in criteria[:5]:
        print(f"  - [{c.theme}] {c.text[:70]}...")

    # Step 6: Theme Agents
    print_section("STEP 6: THEME AGENTS")
    theme_outputs = pipeline._run_theme_agents(paper, criteria)

    for i, output in enumerate(theme_outputs):
        print(f"\n  Theme: {output.theme}")
        print(f"  Strengths ({len(output.strengths)}):")
        for s in output.strengths[:2]:
            print(f"    + {s[:60]}...")
        print(f"  Weaknesses ({len(output.weaknesses)}):")
        for w in output.weaknesses[:2]:
            print(f"    - {w[:60]}...")

    # Step 7: Arbiter
    print_section("STEP 7: ARBITER JUDGMENT")

    # 显示 arbiter 的输入
    print_subsection("Arbiter Input Summary")
    total_strengths = sum(len(o.strengths) for o in theme_outputs)
    total_weaknesses = sum(len(o.weaknesses) for o in theme_outputs)
    print(f"  Total Strengths: {total_strengths}")
    print(f"  Total Weaknesses: {total_weaknesses}")

    if bundle.similar_paper_cases:
        valid_cases = [c for c in bundle.similar_paper_cases if c.rating]
        if valid_cases:
            ratings = [c.rating for c in valid_cases]
            accept_count = sum(1 for c in valid_cases if c.decision and 'accept' in c.decision.lower())
            reject_count = sum(1 for c in valid_cases if c.decision and 'reject' in c.decision.lower())
            print(f"\n  Similar Cases Reference:")
            print(f"    Mean Rating: {statistics.mean(ratings):.2f}")
            print(f"    Median Rating: {statistics.median(ratings):.2f}")
            print(f"    Accept/Reject: {accept_count}/{reject_count}")

    # 运行 arbiter
    arbiter_output = pipeline._aggregate(
        theme_outputs, bundle.policy_cards, policy_criteria, bundle.venue_policy,
        similar_cases=bundle.similar_paper_cases
    )

    print_subsection("Arbiter Decision")
    print(f"  Raw Rating: {arbiter_output.raw_rating:.1f}/10")
    print(f"  Decision: {arbiter_output.decision_recommendation}")
    print(f"  Acceptance Likelihood: {arbiter_output.acceptance_likelihood:.2%}" if arbiter_output.acceptance_likelihood else "")

    if arbiter_output.trace:
        print(f"\n  Rating Rationale: {arbiter_output.trace.get('rating_rationale', 'N/A')[:200]}...")

    print_subsection("Final Strengths")
    for i, s in enumerate(arbiter_output.strengths, 1):
        print(f"  {i}. {s}")

    print_subsection("Final Weaknesses")
    for i, w in enumerate(arbiter_output.weaknesses, 1):
        print(f"  {i}. {w}")

    return arbiter_output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/iclr.yaml")
    parser.add_argument("--paper_id", help="Specific paper ID to review")
    parser.add_argument("--parquet", default="/mnt/data/zzh/datasets/crosseval/crosseval_std/ICLR_2024.parquet")
    parser.add_argument("--row", type=int, default=0, help="Row index in parquet")
    args = parser.parse_args()

    # 加载测试论文
    df = pd.read_parquet(args.parquet)

    if args.paper_id:
        row = df[df['paper_id'] == args.paper_id].iloc[0]
    else:
        row = df[df['decision'].notna()].iloc[args.row]

    paper = Paper(
        paper_id=str(row.get('paper_id')),
        title=str(row.get('title', '')),
        abstract=str(row.get('abstract', '')),
        venue_id='ICLR',
        year=2024,
        authors=[],
        fulltext=None,
    )

    gt_decision = str(row.get('decision', ''))
    print(f"\n{'#'*70}")
    print(f"# Ground Truth: {gt_decision}")
    print(f"{'#'*70}")

    # 运行详细测试
    result = run_detailed_review(args.config, paper)

    # 最终对比
    print_section("RESULT VS GROUND TRUTH")
    gt_binary = 'accept' if 'accept' in gt_decision.lower() else 'reject'
    pred_binary = 'accept' if result.decision_recommendation and 'accept' in result.decision_recommendation.lower() else 'reject'

    print(f"Predicted Rating: {result.raw_rating:.1f}")
    print(f"Predicted Decision: {result.decision_recommendation}")
    print(f"Ground Truth Decision: {gt_decision}")
    print(f"Match: {'✓' if gt_binary == pred_binary else '✗'} (GT={gt_binary}, Pred={pred_binary})")


if __name__ == "__main__":
    main()