#!/usr/bin/env python
"""测试 PaperParserV2 的 primary_area 推断能力"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clients.llm_client import LLMClient, LLMConfig
from common.types import Paper
from pipeline.parse_paper_v3 import PaperParserV3
from pipeline.parse_paper_v2 import PaperParserV2, PRIMARY_AREAS, AREA_TO_DOMAIN_MAP


def load_test_papers(jsonl_path: str, n_samples: int = 10) -> list[dict]:
    """加载测试论文（带 ground truth primary_area）"""
    import random
    random.seed(42)

    papers = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("year") == 2024 and data.get("primary_area"):
                papers.append(data)

    # 随机采样
    if len(papers) > n_samples:
        papers = random.sample(papers, n_samples)

    return papers


def extract_abstract(paper_text: str) -> str:
    """从 LaTeX 提取 abstract"""
    import re
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        paper_text,
        re.DOTALL,
    )
    if abstract_match:
        abstract = abstract_match.group(1)
        abstract = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", abstract)
        abstract = re.sub(r"\\[a-zA-Z]+", "", abstract)
        return abstract.strip()
    return ""


def main():
    # 加载配置 - 使用 read_yaml 正确处理环境变量
    from common.utils import read_yaml
    from clients.llm_client import LLMConfig
    config_path = "configs/iclr.yaml"
    config = read_yaml(config_path)

    # 初始化 LLM
    llm_config = LLMConfig(**config["llm"])
    llm = LLMClient(llm_config)
    parser = PaperParserV2(llm)

    # 加载测试论文
    jsonl_path = "/mnt/data/zzh/datasets/DeepReview-13K/data/stru/dataset_with_area.jsonl"
    test_papers = load_test_papers(jsonl_path, n_samples=10)

    print("="*80)
    print("PaperParserV2 Primary Area 推断测试")
    print("="*80)
    print(f"测试论文数: {len(test_papers)}")
    print()

    correct = 0
    results = []

    for data in test_papers:
        paper_text = data.get("paper", "")
        import re
        title_match = re.search(r"\\title\{([^}]+)\}", paper_text)
        title = title_match.group(1) if title_match else "Unknown"
        abstract = extract_abstract(paper_text)

        gt_area = data.get("primary_area")
        gt_domain = AREA_TO_DOMAIN_MAP.get(gt_area, "other")

        paper = Paper(
            paper_id=data.get("id"),
            title=title,
            abstract=abstract,
            venue_id="DeepReview",
            year=2024,
            authors=[],
            fulltext=None,
        )

        print(f"\n论文: {title[:60]}...")
        print(f"GT primary_area: {gt_area}")

        # 使用简化版推断
        signature = parser.parse(paper)

        # 打印 LLM 返回的原始 primary_area
        if parser.last_raw_response:
            raw_area = parser.last_raw_response.get('primary_area', 'N/A')
            print(f"LLM raw primary_area: {raw_area}")

        print(f"Pred domain: {signature.domain}")
        print(f"Pred paper_type: {signature.paper_type}")
        print(f"Pred tasks: {signature.tasks[:3]}")

        # 检查 domain 是否匹配
        match = signature.domain == gt_domain
        correct += match
        print(f"匹配: {'✓' if match else '✗'}")

        results.append({
            "paper_id": data.get("id"),
            "title": title[:60],
            "gt_area": gt_area,
            "gt_domain": gt_domain,
            "pred_domain": signature.domain,
            "pred_paper_type": signature.paper_type,
            "pred_tasks": signature.tasks,
            "match": match,
        })

    print("\n" + "="*80)
    print("总结")
    print("="*80)
    accuracy = correct / len(test_papers)
    print(f"Domain 匹配准确率: {correct}/{len(test_papers)} = {accuracy:.1%}")

    # 按类别统计
    area_stats = {}
    for r in results:
        area = r["gt_area"]
        if area not in area_stats:
            area_stats[area] = {"correct": 0, "total": 0}
        area_stats[area]["total"] += 1
        if r["match"]:
            area_stats[area]["correct"] += 1

    print("\n各类别准确率:")
    for area, stats in sorted(area_stats.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {area[:50]}... {stats['correct']}/{stats['total']} = {acc:.0%}")

    # 保存结果
    output_path = Path("data/eval_results/parser_v2_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()