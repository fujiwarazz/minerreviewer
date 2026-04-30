#!/usr/bin/env python3
"""
运行 Pipeline 并记录每一步的输入和输出
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from pipeline.review_pipeline import ReviewPipeline
from common.types import Paper

logging.basicConfig(level=logging.WARNING)

def save_step(step_name: str, input_data, output_data, output_dir: Path):
    """保存步骤的输入和输出"""
    step_dir = output_dir / step_name
    step_dir.mkdir(parents=True, exist_ok=True)

    # 保存输入
    if input_data is not None:
        if hasattr(input_data, 'model_dump'):
            input_json = input_data.model_dump(mode='json')
        elif isinstance(input_data, dict):
            input_json = input_data
        elif isinstance(input_data, list):
            input_json = [item.model_dump(mode='json') if hasattr(item, 'model_dump') else item for item in input_data]
        else:
            input_json = str(input_data)

        with open(step_dir / 'input.json', 'w', encoding='utf-8') as f:
            json.dump(input_json, f, ensure_ascii=False, indent=2, default=str)

    # 保存输出
    if output_data is not None:
        if hasattr(output_data, 'model_dump'):
            output_json = output_data.model_dump(mode='json')
        elif isinstance(output_data, dict):
            output_json = output_data
        elif isinstance(output_data, list):
            output_json = [item.model_dump(mode='json') if hasattr(item, 'model_dump') else item for item in output_data]
        else:
            output_json = str(output_data)

        with open(step_dir / 'output.json', 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2, default=str)

    print(f"  Step {step_name}: saved to {step_dir}")


def run_pipeline_with_trace(paper: Paper, pipeline: ReviewPipeline, target_year: int):
    """运行 pipeline 并记录每一步"""

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"data/pipeline_traces/{timestamp}_{paper.paper_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Pipeline Trace: {paper.paper_id}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    retrieval_cfg = pipeline.config["retrieval"]
    distill_cfg = pipeline.config["distill"]

    # =========================================================================
    # Step 0: 输入论文
    # =========================================================================
    print("Step 0: Input Paper")
    save_step("00_input_paper", None, paper, output_dir)

    # =========================================================================
    # Step 1: PaperParser
    # =========================================================================
    print("Step 1: PaperParser")
    signature = pipeline._parse_paper(paper)
    save_step("01_paper_parser", paper, signature, output_dir)

    # =========================================================================
    # Step 2: Retriever
    # =========================================================================
    print("Step 2: Retriever")
    bundle = pipeline._retrieve_multi_channel(paper, signature, target_year)
    save_step("02_retriever", signature, bundle, output_dir)

    # =========================================================================
    # Step 3: CriteriaMiner
    # =========================================================================
    print("Step 3: CriteriaMiner")
    content_criteria, policy_criteria = pipeline._mine_criteria(paper, signature, bundle, target_year)
    save_step("03_criteria_miner", bundle, {"content_criteria": content_criteria, "policy_criteria": policy_criteria}, output_dir)

    # =========================================================================
    # Step 4: CriteriaPlanner
    # =========================================================================
    print("Step 4: CriteriaPlanner")
    activated = pipeline._plan_criteria(signature, bundle, content_criteria, policy_criteria)
    save_step("04_criteria_planner", {"signature": signature, "content_criteria": content_criteria, "policy_criteria": policy_criteria}, activated, output_dir)

    # =========================================================================
    # Step 5: CriteriaRewriter
    # =========================================================================
    print("Step 5: CriteriaRewriter")
    criteria = pipeline._rewrite_criteria(paper, activated)
    save_step("05_criteria_rewriter", activated, criteria, output_dir)

    # =========================================================================
    # Step 6: ThemeAgents
    # =========================================================================
    print("Step 6: ThemeAgents")
    theme_outputs = pipeline._run_theme_agents(paper, criteria, bundle.policy_cards)
    save_step("06_theme_agents", {"paper": paper, "criteria": criteria, "policy_cards": bundle.policy_cards[:5]}, theme_outputs, output_dir)

    # =========================================================================
    # Step 7: Arbiter
    # =========================================================================
    print("Step 7: Arbiter")
    arbiter_output = pipeline._aggregate(
        theme_outputs, bundle.policy_cards, policy_criteria, bundle.venue_policy,
        similar_cases=bundle.similar_paper_cases,
    )
    save_step("07_arbiter", {"theme_outputs": theme_outputs, "similar_cases": bundle.similar_paper_cases[:3]}, arbiter_output, output_dir)

    # =========================================================================
    # Step 8: Verifier
    # =========================================================================
    print("Step 8: Verifier")
    verification = pipeline._verify_decision(arbiter_output, paper, bundle)
    save_step("08_verifier", arbiter_output, verification, output_dir)

    # =========================================================================
    # Step 9: ScoreConsistencyChecker
    # =========================================================================
    print("Step 9: ScoreConsistencyChecker")
    consistency = pipeline._check_score_consistency(arbiter_output, bundle)
    save_step("09_score_consistency", arbiter_output, consistency, output_dir)

    # =========================================================================
    # Step 10: Calibrator
    # =========================================================================
    print("Step 10: Calibrator")
    calibration = pipeline._calibrate_multiclass(arbiter_output.raw_rating, target_year)
    save_step("10_calibrator", {"raw_rating": arbiter_output.raw_rating, "target_year": target_year}, calibration, output_dir)

    # =========================================================================
    # Step 11: Apply Calibration
    # =========================================================================
    print("Step 11: Apply Calibration")
    arbiter_output = pipeline._apply_calibration(arbiter_output, calibration)
    save_step("11_apply_calibration", calibration, arbiter_output, output_dir)

    # =========================================================================
    # Step 12: ExperienceDistiller
    # =========================================================================
    print("Step 12: ExperienceDistiller")
    experience = pipeline._distill_experience(arbiter_output, paper, signature, bundle)
    save_step("12_experience_distiller", arbiter_output, experience, output_dir)

    # =========================================================================
    # Step 13: MemoryEditor
    # =========================================================================
    print("Step 13: MemoryEditor")
    memory_updates = pipeline._update_memory(experience)
    save_step("13_memory_editor", experience, memory_updates, output_dir)

    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"Final rating: {arbiter_output.raw_rating}")
    print(f"Decision: {arbiter_output.decision_recommendation}")
    print(f"Acceptance likelihood: {arbiter_output.acceptance_likelihood}")
    print(f"Trace saved to: {output_dir}")
    print(f"{'='*60}\n")

    return arbiter_output, output_dir


def load_paper_from_deepreview(year: int = 2024) -> Paper:
    """从 DeepReview-13K 加载指定年份的论文"""
    import json
    from pathlib import Path

    deepreview_path = Path("/mnt/data/zzh/datasets/DeepReview-13K/data/stru/train_fast.jsonl")

    if not deepreview_path.exists():
        raise FileNotFoundError(f"DeepReview-13K data not found: {deepreview_path}")

    # 读取并筛选指定年份的论文
    papers_2024 = []
    with open(deepreview_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('year') == year:
                papers_2024.append(data)

    if not papers_2024:
        raise ValueError(f"No papers found for year {year}")

    # 随机选择一篇
    import random
    selected = random.choice(papers_2024)

    # 从 paper 内容提取 title 和 abstract
    paper_content = selected['paper']

    # 尝试提取 title
    title = "Unknown"
    import re
    # LaTeX \title{...}
    match = re.search(r'\\title\{([^}]+)\}', paper_content)
    if match:
        title = match.group(1).strip()
    else:
        # Markdown # Title
        match = re.search(r'^#\s+(.+)$', paper_content, re.MULTILINE)
        if match:
            title = match.group(1).strip()

    # 尝试提取 abstract
    abstract = ""
    match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', paper_content, re.DOTALL)
    if match:
        abstract = match.group(1).strip()[:2000]  # 限制长度
    else:
        # 取前 1000 字符作为 abstract
        abstract = paper_content[:1000]

    return Paper(
        paper_id=selected['id'],
        title=title,
        abstract=abstract,
        venue_id="ICLR",
        year=selected['year'],
        authors=[],
        fulltext=paper_content[:10000]  # 限制 fulltext 长度
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/iclr.yaml')
    parser.add_argument('--paper-id', default=None, help='Paper ID from doc_store')
    parser.add_argument('--deepreview', action='store_true', help='Use DeepReview-13K dataset')
    parser.add_argument('--year', type=int, default=2024, help='Year for DeepReview-13K (2024 or 2025)')
    args = parser.parse_args()

    print("Initializing pipeline...")
    pipeline = ReviewPipeline(args.config)

    # 获取测试论文
    if args.deepreview:
        print(f"Loading paper from DeepReview-13K (year={args.year})...")
        paper = load_paper_from_deepreview(args.year)
        print(f"Using paper: {paper.paper_id} - {paper.title[:50]}...")
    elif args.paper_id:
        papers = pipeline.doc_store.load_papers(pipeline.venue_id)
        paper = next((p for p in papers if p.paper_id == args.paper_id), None)
        if not paper:
            print(f"Paper {args.paper_id} not found")
            return
    else:
        # 使用 doc_store 中的论文
        papers = pipeline.doc_store.load_papers(pipeline.venue_id)
        if papers:
            paper = papers[0]
            print(f"Using paper from doc_store: {paper.paper_id} - {paper.title[:50]}...")
        else:
            print("No papers found in doc_store")
            return

    output, trace_dir = run_pipeline_with_trace(paper, pipeline, paper.year or 2024)

    # 打印摘要
    print("\n" + "="*60)
    print("PIPELINE TRACE SUMMARY")
    print("="*60)
    print(f"\nTrace directory: {trace_dir}")
    print("\nSteps recorded:")
    for step_dir in sorted(trace_dir.iterdir()):
        if step_dir.is_dir():
            input_file = step_dir / 'input.json'
            output_file = step_dir / 'output.json'
            input_size = input_file.stat().st_size if input_file.exists() else 0
            output_size = output_file.stat().st_size if output_file.exists() else 0
            print(f"  {step_dir.name}: input={input_size}B, output={output_size}B")


if __name__ == '__main__':
    main()