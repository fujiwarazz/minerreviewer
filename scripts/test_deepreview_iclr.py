#!/usr/bin/env python
"""
使用 DeepReview-13K 的 50 篇 2024 年论文测试 ICLR 记忆库

- 测试集：DeepReview-13K 中 50 篇 2024 年论文
- 记忆库：ICLR 2024 之前的记忆（不包含 DeepReview）
- primary_area：作为论文元数据存储
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from common.types import Paper
from common.utils import write_json
from eval.coverage import evaluate_coverage
from pipeline.review_pipeline import ReviewPipeline

def calc_avg_coverage(cov: dict | None) -> float | None:
    """计算平均覆盖率"""
    if not cov or "strengths" not in cov or "score" not in cov["strengths"]:
        return None
    s_score = cov["strengths"]["score"]
    w_score = cov.get("weaknesses", {}).get("score", 0)
    return (s_score + w_score) / 2


_pipeline = None
_pipeline_lock = threading.Lock()


def parse_paper_from_latex(paper_text: str) -> tuple[str, str]:
    """从 LaTeX 文本提取 title 和 abstract"""
    # 提取 title
    title_match = re.search(r"\\title\{([^}]+)\}", paper_text)
    title = title_match.group(1) if title_match else "Unknown Title"

    # 提取 abstract
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        paper_text,
        re.DOTALL,
    )
    abstract = abstract_match.group(1) if abstract_match else ""
    # 清理 LaTeX
    abstract = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", abstract)
    abstract = re.sub(r"\\[a-zA-Z]+", "", abstract)
    abstract = abstract.strip()

    return title, abstract


def load_deepreview_test_papers(
    jsonl_path: str,
    year: int = 2024,
    n_samples: int = 50,
    seed: int = 42,
) -> list[dict]:
    """从 DeepReview-13K 加载测试论文"""
    random.seed(seed)

    papers = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("year") == year:
                paper_text = data.get("paper", "")
                title, abstract = parse_paper_from_latex(paper_text)

                papers.append({
                    "paper_id": data.get("id"),
                    "title": title,
                    "abstract": abstract,
                    "primary_area": data.get("primary_area"),
                    "decision": data.get("decision"),
                    "rating_raw": data.get("rating"),
                    "year": year,
                    "reviewer_comments": data.get("reviewer_comments", []),  # 加载 ground truth
                })

    print(f"Found {len(papers)} papers from DeepReview-{year}")

    # 随机采样
    if len(papers) > n_samples:
        papers = random.sample(papers, n_samples)

    return papers


def parse_rating(rating_raw: str) -> float | None:
    """解析 rating（如 "[6, 6, 8, 8, 6]" -> 6.8）"""
    try:
        if rating_raw.startswith("["):
            ratings = json.loads(rating_raw)
            return sum(ratings) / len(ratings) if ratings else None
        else:
            return float(rating_raw)
    except (json.JSONDecodeError, ValueError):
        return None


def extract_ground_truth_reviews(reviewer_comments: list) -> list[dict]:
    """从 reviewer_comments 提取 ground truth reviews"""
    gt_reviews = []
    for rc in reviewer_comments:
        content = rc.get("content", {})
        if content.get("strengths") or content.get("weaknesses"):
            gt_reviews.append({
                "strengths": content.get("strengths", ""),
                "weaknesses": content.get("weaknesses", ""),
                "rating": content.get("rating"),
            })
    return gt_reviews


def run_test(
    paper_info: dict,
    config_path: str,
    target_year: int = 2024,
    coverage_config: dict | None = None,
) -> dict:
    """运行单个论文的 review（线程安全）"""
    global _pipeline

    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = ReviewPipeline(config_path)

    pipeline = _pipeline

    paper = Paper(
        paper_id=paper_info["paper_id"],
        title=paper_info["title"],
        abstract=paper_info["abstract"],
        venue_id="DeepReview",
        year=paper_info["year"],
        authors=[],
        fulltext=None,
    )

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

    criteria_info = []
    for c in criteria:
        criteria_info.append({
            "theme": c.theme,
            "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
            "kind": c.kind,
        })

    theme_outputs_info = []
    for to in theme_outputs:
        theme_outputs_info.append({
            "theme": to.theme,
            "strengths": to.strengths,
            "weaknesses": to.weaknesses,
            "severity_tags": to.severity_tags,
        })

    gt_decision = paper_info.get("decision")
    gt_rating = parse_rating(paper_info.get("rating_raw", ""))

    pred_decision = arbiter_output.decision_recommendation or "Unknown"
    pred_rating = arbiter_output.raw_rating

    gt_binary = "accept" if gt_decision and "accept" in gt_decision.lower() else "reject"
    pred_binary = "accept" if pred_decision and "accept" in pred_decision.lower() else "reject"
    match = gt_binary == pred_binary

    coverage_result = None
    if coverage_config and coverage_config.get("enabled", False):
        gt_reviews = extract_ground_truth_reviews(paper_info.get("reviewer_comments", []))
        if gt_reviews:
            try:
                coverage_result = evaluate_coverage(
                    strengths=arbiter_output.strengths,
                    weaknesses=arbiter_output.weaknesses,
                    ground_truth_reviews=gt_reviews,
                    llm=pipeline.llm,
                    embedding_client=pipeline.embedding_client,
                    config=coverage_config,
                )
            except Exception as e:
                coverage_result = {"error": str(e)}

    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract,
        "primary_area": paper_info.get("primary_area"),
        "ground_truth": {
            "decision": gt_decision,
            "rating": gt_rating,
            "n_reviewers": len(paper_info.get("reviewer_comments", [])),
        },
        "prediction": {
            "decision": pred_decision,
            "rating": round(pred_rating, 2),
            "acceptance_likelihood": arbiter_output.acceptance_likelihood,
            "decision_rationale": arbiter_output.decision_rationale,
            "score_rationale": arbiter_output.score_rationale,
            "key_decisive_issues": arbiter_output.key_decisive_issues,
        },
        "match": match,
        "review": {
            "strengths": arbiter_output.strengths,
            "weaknesses": arbiter_output.weaknesses,
        },
        "theme_outputs": theme_outputs_info,
        "criteria_used": criteria_info,
        "similar_cases": similar_cases_info,
        "coverage": coverage_result,
        "trace": {
            "similar_cases_count": len(bundle.similar_paper_cases),
            "policy_cards_count": len(bundle.policy_cards),
            "content_criteria_count": len(content_criteria),
            "policy_criteria_count": len(policy_criteria),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/iclr.yaml")
    parser.add_argument("--input", default="/mnt/data/zzh/datasets/DeepReview-13K/data/stru/dataset_with_area.jsonl")
    parser.add_argument("--output-dir", default="data/eval_results")
    parser.add_argument("--year", type=int, default=2024, help="测试年份")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-workers", type=int, default=5, help="并行工作线程数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage", action="store_true", help="启用覆盖率评估")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DeepReview-13K 测试")
    print("="*60)
    print(f"配置: {args.config}")
    print(f"测试集: DeepReview {args.year} ({args.n_samples} 篇)")
    print(f"记忆库: ICLR {args.year} 之前")
    print(f"覆盖率评估: {'启用' if args.coverage else '禁用'}")
    print(f"并行线程: {args.n_workers}")
    print()

    _pipeline = ReviewPipeline(args.config)
    print("Pipeline loaded!\n")

    coverage_config = None
    if args.coverage:
        coverage_config = _pipeline.config.get("coverage_eval", {"enabled": True, "method": "llm", "threshold": 0.55})

    papers = load_deepreview_test_papers(
        args.input,
        year=args.year,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    results = []
    completed_count = 0
    print_lock = threading.Lock()

    def process_paper(paper_info: dict, idx: int) -> tuple[int, dict]:
        nonlocal completed_count
        try:
            result = run_test(
                paper_info,
                config_path=args.config,
                target_year=args.year,
                coverage_config=coverage_config,
            )

            with print_lock:
                completed_count += 1
                print(f"\n[{completed_count}/{len(papers)}] {paper_info['paper_id']}")
                print(f"  Title: {paper_info['title'][:60]}...")
                print(f"  GT: {paper_info['decision']} (rating: {parse_rating(paper_info.get('rating_raw', ''))})")
                print(f"  Pred: {result['prediction']['decision']} (rating: {result['prediction']['rating']})")
                print(f"  Similar cases: {result['trace']['similar_cases_count']}")
                print(f"  Match: {result['match']}")

                if result.get("coverage"):
                    avg_cov = calc_avg_coverage(result["coverage"])
                    if avg_cov is not None:
                        print(f"  Coverage: avg={avg_cov:.2f}")

            return (idx, result)

        except Exception as e:
            with print_lock:
                completed_count += 1
                print(f"\n[{completed_count}/{len(papers)}] {paper_info['paper_id']}")
                print(f"  ERROR: {e}")
            return (idx, {"paper_id": paper_info["paper_id"], "error": str(e)})

    print(f"Starting parallel processing with {args.n_workers} workers...")
    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(process_paper, paper, i): i
            for i, paper in enumerate(papers)
        }

        for future in as_completed(futures):
            idx, result = future.result()
            results.append((idx, result))

    results.sort(key=lambda x: x[0])
    results = [r for _, r in results]

    valid_results = [r for r in results if "error" not in r]
    correct = sum(1 for r in valid_results if r.get("match"))
    accuracy = correct / len(valid_results) if valid_results else 0

    total_coverage = []
    for r in valid_results:
        avg_cov = calc_avg_coverage(r.get("coverage"))
        if avg_cov is not None:
            total_coverage.append(avg_cov)

    avg_coverage = sum(total_coverage) / len(total_coverage) if total_coverage else None

    elapsed = (datetime.now() - start_time).total_seconds()

    summary = {
        "config": args.config,
        "timestamp": datetime.now().isoformat(),
        "test_year": args.year,
        "n_samples": len(papers),
        "n_workers": args.n_workers,
        "elapsed_seconds": round(elapsed, 2),
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total_valid": len(valid_results),
        "avg_coverage": round(avg_coverage, 4) if avg_coverage else None,
        "coverage_enabled": args.coverage,
        "results": results,
    }

    output_path = output_dir / f"deepreview_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(output_path, summary)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Accuracy: {correct}/{len(valid_results)} = {accuracy*100:.1f}%")
    if avg_coverage is not None:
        print(f"Average Coverage: {avg_coverage:.4f}")
    print(f"Elapsed: {elapsed:.1f}s ({len(papers)} papers, {args.n_workers} workers)")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()