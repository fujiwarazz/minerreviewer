#!/usr/bin/env python
"""测试 PaperParserV3 的两阶段推断"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.utils import read_yaml
from clients.llm_client import LLMClient, LLMConfig
from common.types import Paper
from pipeline.parse_paper_v3 import PaperParserV3, AREA_TO_DOMAIN_MAP


def extract_abstract(paper_text: str) -> str:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", paper_text, re.DOTALL)
    if match:
        abstract = match.group(1)
        abstract = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", abstract)
        abstract = re.sub(r"\\[a-zA-Z]+", "", abstract)
        return abstract.strip()
    return ""


def load_test_papers(jsonl_path: str, n_samples: int = 10):
    import random
    random.seed(42)

    papers = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("year") == 2024 and data.get("primary_area"):
                papers.append(data)

    if len(papers) > n_samples:
        papers = random.sample(papers, n_samples)
    return papers


def main():
    config = read_yaml("configs/iclr.yaml")
    llm = LLMClient(LLMConfig(**config["llm"]))
    parser = PaperParserV3(llm)

    jsonl_path = "/mnt/data/zzh/datasets/DeepReview-13K/data/stru/dataset_with_area.jsonl"
    test_papers = load_test_papers(jsonl_path, n_samples=10)

    print("="*80)
    print("PaperParserV3 两阶段推断测试")
    print("="*80)
    print(f"测试论文数: {len(test_papers)}\n")

    correct = 0
    results = []

    for data in test_papers:
        paper_text = data.get("paper", "")
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

        print(f"论文: {title[:50]}...")
        print(f"  GT: {gt_area[:40]}...")

        signature = parser.parse(paper)

        # 打印分析结果
        if parser.last_raw_response:
            llm_domain = parser.last_raw_response.get("domain", "N/A")
            print(f"  LLM domain: {llm_domain}")

        print(f"  Pred domain: {signature.domain}")
        match = signature.domain == gt_domain
        correct += match
        print(f"  匹配: {'✓' if match else '✗'}\n")

        results.append({
            "title": title[:50],
            "gt_area": gt_area,
            "gt_domain": gt_domain,
            "core_domain": parser.last_raw_response.get("core_technical_domain") if parser.last_raw_response else None,
            "pred_domain": signature.domain,
            "match": match,
        })

    print("="*80)
    print("总结")
    print("="*80)
    accuracy = correct / len(test_papers)
    print(f"准确率: {correct}/{len(test_papers)} = {accuracy:.1%}")

    # 保存结果
    output_path = Path("data/eval_results/parser_v3_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)

    print(f"结果保存到: {output_path}")


if __name__ == "__main__":
    main()