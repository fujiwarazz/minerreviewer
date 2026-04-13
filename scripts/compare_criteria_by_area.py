#!/usr/bin/env python
"""
对比不同研究方向的论文挖掘出的criteria差异

选择方向：
- RL (reinforcement learning)
- Theory (learning theory)
- Graph (learning on graphs)
- CV (representation learning for vision/audio/language)
- Optimization
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.types import Paper
from pipeline.review_pipeline import ReviewPipeline


TARGET_AREAS = [
    "reinforcement learning",
    "learning theory",
    "learning on graphs and other geometries & topologies",
    "representation learning for computer vision, audio, language, and other modalities",
    "optimization",
]

AREA_SHORT = {
    "reinforcement learning": "RL",
    "learning theory": "Theory",
    "learning on graphs and other geometries & topologies": "Graph",
    "representation learning for computer vision, audio, language, and other modalities": "CV/NLP",
    "optimization": "Optimization",
}


def parse_paper_from_latex(paper_text: str) -> tuple[str, str]:
    """从 LaTeX 文本提取 title 和 abstract"""
    title_match = re.search(r"\\title\{([^}]+)\}", paper_text)
    title = title_match.group(1) if title_match else "Unknown Title"

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        paper_text,
        re.DOTALL,
    )
    abstract = abstract_match.group(1) if abstract_match else ""
    abstract = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", abstract)
    abstract = re.sub(r"\\[a-zA-Z]+", "", abstract)
    abstract = abstract.strip()

    return title, abstract


def select_papers_by_area(jsonl_path: str, areas: list[str], year: int = 2024) -> dict[str, list[dict]]:
    """按研究方向选择论文"""
    area_papers = defaultdict(list)

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("year") != year:
                continue

            area = data.get("primary_area", "")
            if area in areas:
                paper_text = data.get("paper", "")
                title, abstract = parse_paper_from_latex(paper_text)

                area_papers[area].append({
                    "paper_id": data.get("id"),
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "decision": data.get("decision"),
                    "rating_raw": data.get("rating"),
                })

    return area_papers


def run_analysis(paper_info: dict, pipeline: ReviewPipeline, target_year: int = 2024) -> dict:
    """运行单个论文的criteria分析"""
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

    # 收集信息
    content_by_theme = defaultdict(list)
    for c in content_criteria:
        content_by_theme[c.theme].append(c.text[:150])

    policy_by_theme = defaultdict(list)
    for p in policy_criteria:
        policy_by_theme[p.theme].append(p.text[:150])

    activated_by_theme = defaultdict(list)
    for a in activated:
        activated_by_theme[a.theme].append({
            "text": a.criterion[:200],
            "source": a.source,
        })

    return {
        "paper_id": paper_info["paper_id"],
        "title": paper_info["title"][:80],
        "signature": {
            "paper_type": signature.paper_type,
            "tasks": signature.tasks[:3] if signature.tasks else [],
            "domain": signature.domain[:3] if signature.domain else [],
            "method_family": signature.method_family[:3] if signature.method_family else [],
        },
        "content_criteria_by_theme": dict(content_by_theme),
        "policy_criteria_by_theme": dict(policy_by_theme),
        "activated_criteria_by_theme": dict(activated_by_theme),
        "final_criteria_count": len(criteria),
        "similar_cases": [
            {
                "paper_id": c.paper_id,
                "venue": c.venue_id,
                "year": c.year,
                "rating": c.rating,
                "decision": c.decision,
            }
            for c in bundle.similar_paper_cases[:3]
        ],
        "policy_cards": [
            {"kind": p.kind, "theme": p.theme, "content": p.content[:50]}
            for p in bundle.policy_cards[:3]
        ],
    }


def main():
    jsonl_path = "/mnt/data/zzh/datasets/DeepReview-13K/data/stru/dataset_with_area.jsonl"
    config_path = "configs/iclr.yaml"
    output_path = Path("data/eval_results/criteria_comparison.json")

    print("加载论文数据...")
    area_papers = select_papers_by_area(jsonl_path, TARGET_AREAS, year=2024)

    for area, papers in area_papers.items():
        print(f"  {AREA_SHORT[area]}: {len(papers)} 篇")

    # 每个方向选2篇代表性论文
    selected = {}
    for area in TARGET_AREAS:
        papers = area_papers.get(area, [])
        if len(papers) >= 2:
            # 选择一篇accept一篇reject（如果可能）
            accepts = [p for p in papers if p.get("decision") and "accept" in p["decision"].lower()]
            rejects = [p for p in papers if p.get("decision") and "reject" in p["decision"].lower()]
            if accepts and rejects:
                selected[area] = [accepts[0], rejects[0]]
            else:
                selected[area] = papers[:2]
        elif papers:
            selected[area] = papers[:1]

    print(f"\n选择 {sum(len(v) for v in selected.values())} 篇论文进行对比分析")

    # 初始化pipeline
    print("\n加载 Pipeline...")
    pipeline = ReviewPipeline(config_path)

    results = {}
    for area, papers in selected.items():
        area_name = AREA_SHORT[area]
        results[area_name] = {"papers": []}

        for paper_info in papers:
            print(f"\n处理 [{area_name}] {paper_info['paper_id']}: {paper_info['title'][:50]}...")
            try:
                analysis = run_analysis(paper_info, pipeline, target_year=2024)
                results[area_name]["papers"].append(analysis)

                print(f"  Signature: type={analysis['signature']['paper_type']}")
                print(f"  Content criteria: {sum(len(v) for v in analysis['content_criteria_by_theme'].values())}")
                print(f"  Policy criteria: {sum(len(v) for v in analysis['policy_criteria_by_theme'].values())}")
                print(f"  Final criteria: {analysis['final_criteria_count']}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # 输出对比分析
    print("\n" + "="*80)
    print("CRITERIA 对比分析")
    print("="*80)

    for area_name, area_data in results.items():
        print(f"\n### {area_name}")
        print("-"*60)
        for paper in area_data.get("papers", []):
            print(f"\n论文: {paper['title']}")
            print(f"  类型: {paper['signature']['paper_type']}")
            print(f"  任务: {paper['signature']['tasks']}")
            print(f"  领域: {paper['signature']['domain']}")

            print(f"\n  Content Criteria themes: {list(paper['content_criteria_by_theme'].keys())}")
            print(f"  Policy Criteria themes: {list(paper['policy_criteria_by_theme'].keys())}")

            print(f"\n  Final Criteria ({paper['final_criteria_count']}):")
            for theme, crits in paper['activated_criteria_by_theme'].items():
                print(f"    [{theme}] {len(crits)} criteria")

            if paper['similar_cases']:
                print(f"\n  相似案例:")
                for c in paper['similar_cases']:
                    print(f"    {c['paper_id']} ({c['venue']} {c['year']}) rating={c['rating']} {c['decision']}")

    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()