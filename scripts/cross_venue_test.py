#!/usr/bin/env python
"""
跨会议测试脚本 - 只测试 NeurIPS 2022（用 ICLR 训练的模型）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.types import Paper
from pipeline.review_pipeline import ReviewPipeline
from clients.llm_client import LLMClient, LLMConfig
from common.utils import read_yaml

DATA_ROOT = Path("/mnt/data/zzh/datasets/reviewer2")

# 全局 LLM client（延迟初始化）
_llm_client = None


def get_llm_client(config_path: str = "configs/iclr.yaml") -> LLMClient:
    """获取 LLM client（单例）"""
    global _llm_client
    if _llm_client is None:
        cfg = read_yaml(config_path)
        llm_cfg = LLMConfig(
            backend=cfg.get("llm", {}).get("backend", "openai"),
            model=cfg.get("llm", {}).get("model", "qwen-plus"),
            temperature=0.1,
            base_url=cfg.get("llm", {}).get("base_url"),
            api_key=cfg.get("llm", {}).get("api_key"),
        )
        _llm_client = LLMClient(llm_cfg)
    return _llm_client


def calculate_coverage_with_llm(
    pred_items: list[str],
    gt_items: list[str],
    item_type: str = "strengths",
) -> dict:
    """用 LLM 判断覆盖率

    Args:
        pred_items: 预测的 strengths/weaknesses 列表
        gt_items: ground truth 的 strengths/weaknesses 列表
        item_type: "strengths" 或 "weaknesses"

    Returns:
        {"coverage": float, "matched": int, "total": int, "details": list}
    """
    if not gt_items:
        return {"coverage": 1.0 if not pred_items else 0.0, "matched": 0, "total": 0, "details": []}

    if not pred_items:
        return {"coverage": 0.0, "matched": 0, "total": len(gt_items), "details": []}

    llm = get_llm_client()

    # 格式化输入
    gt_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(gt_items)])
    pred_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(pred_items)])

    prompt = f"""You are a reviewer evaluating how well predicted {item_type} cover the ground truth {item_type}.

Ground Truth {item_type.title()}:
{gt_text}

Predicted {item_type.title()}:
{pred_text}

Task: For each ground truth point, determine if it is covered (mentioned or addressed) in the predicted {item_type}.

Output format (JSON only, no other text):
{{
  "results": [
    {{"gt_index": 1, "covered": true, "reason": "brief reason"}},
    {{"gt_index": 2, "covered": false, "reason": "brief reason"}},
    ...
  ],
  "coverage": <number between 0 and 1>
}}

Coverage = number of covered points / total points.
"""

    try:
        response = llm.generate(prompt)
        # 解析 JSON
        import json as json_module
        # 提取 JSON 部分
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            result = json_module.loads(json_match.group())
            coverage = result.get("coverage", 0.0)
            details = result.get("results", [])
            matched = sum(1 for d in details if d.get("covered", False))
            return {
                "coverage": round(coverage, 4),
                "matched": matched,
                "total": len(gt_items),
                "details": details,
            }
    except Exception as e:
        print(f"    LLM coverage check failed: {e}")

    # 回退到简单关键词匹配
    return calculate_coverage(pred_items, gt_items)


def calculate_coverage(pred_items: list[str], gt_items: list[str], threshold: float = 0.3) -> dict:
    """计算覆盖率（基于关键词重叠，作为回退方法）"""
    if not gt_items:
        return {"coverage": 1.0 if not pred_items else 0.0, "matched": 0, "total": 0}

    # 提取关键词
    def extract_keywords(texts: list[str]) -> set[str]:
        keywords = set()
        for text in texts:
            # 简单分词并过滤
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            keywords.update(words)
        return keywords

    pred_keywords = extract_keywords(pred_items)
    gt_keywords = extract_keywords(gt_items)

    if not gt_keywords:
        return {"coverage": 1.0, "matched": 0, "total": 0}

    # 计算重叠
    overlap = pred_keywords & gt_keywords
    coverage = len(overlap) / len(gt_keywords) if gt_keywords else 0.0

    return {
        "coverage": round(coverage, 4),
        "matched": len(overlap),
        "total": len(gt_keywords),
    }


def parse_review_file(file_path: Path) -> dict | None:
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

    if isinstance(meta_review, dict):
        meta_review = meta_review.get('text', '') or str(meta_review)

    venue, year = None, None
    for part in file_path.parts:
        if 'NIPS' in part or 'NeurIPS' in part:
            venue = 'NeurIPS'
            ym = re.search(r'(\d{4})', part)
            if ym:
                year = int(ym.group(1))

    title = data.get('title', '')
    abstract = data.get('abstract', '')

    # reviewer2 数据没有 title/abstract，从 reviews 的 Summary 字段提取
    if not title or not abstract:
        summaries = []
        for review in reviews:
            if isinstance(review, dict):
                for key in ['Summary:', 'Summary', 'summary']:
                    if key in review and review[key]:
                        summaries.append(review[key])
                        break
        if summaries:
            # 使用最长的 Summary 作为 abstract
            abstract = max(summaries, key=len)
            # 从 Summary 提取标题（第一句话）
            if not title and abstract:
                first_sentence = abstract.split('.')[0]
                if len(first_sentence) > 20:
                    title = first_sentence[:100]

    # 提取 strengths 和 weaknesses
    gt_strengths = []
    gt_weaknesses = []
    for review in reviews:
        if isinstance(review, dict):
            # reviewer2 格式: "Strengths And Weaknesses:" 字段包含混合文本
            sw_text = None
            for key in ['Strengths And Weaknesses:', 'Strengths And Weaknesses', 'strengths and weaknesses']:
                if key in review and review[key]:
                    sw_text = review[key]
                    break

            if sw_text:
                # 解析混合文本，按 "Strengths" 和 "Weaknesses" 分割
                sw_lower = sw_text.lower()
                strengths_start = sw_lower.find('strengths')
                # 同时搜索单数和复数形式
                weaknesses_start = sw_lower.find('weaknesses')
                if weaknesses_start == -1:
                    weaknesses_start = sw_lower.find('weakness')
                    if weaknesses_start != -1:
                        # 确保是独立的词，不是 "weaknesses" 的一部分
                        if weaknesses_start + 9 <= len(sw_lower) and sw_lower[weaknesses_start+9:weaknesses_start+10].isalpha():
                            weaknesses_start = -1  # 是 "weaknesses" 的一部分，跳过

                if strengths_start != -1 and weaknesses_start != -1:
                    if strengths_start < weaknesses_start:
                        # Strengths 在前
                        strengths_part = sw_text[strengths_start:weaknesses_start]
                        weaknesses_part = sw_text[weaknesses_start:]
                    else:
                        # Weaknesses 在前
                        weaknesses_part = sw_text[weaknesses_start:strengths_start]
                        strengths_part = sw_text[strengths_start:]

                    # 提取要点（按换行或数字列表分割）
                    def extract_points(text):
                        points = []
                        lines = text.split('\n')
                        for line in lines:
                            line = line.strip()
                            # 跳过标题行
                            if 'strength' in line.lower() or 'weakness' in line.lower():
                                continue
                            if len(line) > 20:  # 过滤太短的行
                                points.append(line)
                        return points

                    gt_strengths.extend(extract_points(strengths_part))
                    gt_weaknesses.extend(extract_points(weaknesses_part))
                else:
                    # 无法分割，作为整体
                    if len(sw_text) > 50:
                        gt_strengths.append(sw_text[:200])

            # 也检查独立的 strengths/weaknesses 字段
            for key in ['Strengths', 'strengths', 'STRENGTHS']:
                if key in review and review[key]:
                    text = review[key]
                    if isinstance(text, str) and len(text) > 20:
                        gt_strengths.append(text[:200])
                    elif isinstance(text, list):
                        gt_strengths.extend([t[:200] for t in text if len(t) > 20])

            for key in ['Weaknesses', 'weaknesses', 'WEAKNESSES']:
                if key in review and review[key]:
                    text = review[key]
                    if isinstance(text, str) and len(text) > 20:
                        gt_weaknesses.append(text[:200])
                    elif isinstance(text, list):
                        gt_weaknesses.extend([t[:200] for t in text if len(t) > 20])

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
        'gt_strengths': gt_strengths,
        'gt_weaknesses': gt_weaknesses,
    }


def get_test_papers(n_accept=25, n_reject=25, seed=42):
    import random
    random.seed(seed)

    venue_path = DATA_ROOT / 'NIPS'
    files = list(venue_path.glob('**/NIPS_2022*_review/*.json'))
    if not files:
        files = list(venue_path.glob('**/NIPS_2022/*_review/*.json'))

    print(f"Found {len(files)} files for NeurIPS 2022")

    accept_papers = []
    reject_papers = []

    for f in files:
        result = parse_review_file(f)
        if result and result['decision']:
            # 过滤掉没有 GT Weaknesses 的论文
            if not result.get('gt_weaknesses'):
                continue
            if result['decision'] == 'Accept':
                accept_papers.append(result)
            elif result['decision'] == 'Reject':
                reject_papers.append(result)

    print(f"Accept: {len(accept_papers)}, Reject: {len(reject_papers)}")

    sampled_accept = random.sample(accept_papers, min(n_accept, len(accept_papers)))
    sampled_reject = random.sample(reject_papers, min(n_reject, len(reject_papers)))

    return sampled_accept + sampled_reject


def run_test(pipeline, paper_info, max_retries=3):
    for attempt in range(max_retries):
        try:
            paper = Paper(
                paper_id=paper_info['paper_id'],
                title=paper_info.get('title', ''),
                abstract=paper_info.get('abstract', ''),
                venue_id=paper_info['venue'],
                year=paper_info['year'],
                authors=[],
                fulltext=None,
            )

            target_year = paper_info['year'] + 1

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
                })

            import statistics
            valid_cases = [c for c in bundle.similar_paper_cases if c.rating]
            stats = {}
            if valid_cases:
                ratings = [c.rating for c in valid_cases]
                stats = {
                    "mean_rating": round(statistics.mean(ratings), 2),
                    "accept_count": sum(1 for c in valid_cases if c.decision and "accept" in c.decision.lower()),
                    "reject_count": sum(1 for c in valid_cases if c.decision and "reject" in c.decision.lower()),
                }

            gt_decision = paper_info['decision']
            gt_binary = 'accept' if gt_decision.lower() == 'accept' else 'reject'
            pred_binary = 'accept' if arbiter_output.decision_recommendation and 'accept' in arbiter_output.decision_recommendation.lower() else 'reject'

            pred_strengths = arbiter_output.strengths[:5] if arbiter_output.strengths else []
            pred_weaknesses = arbiter_output.weaknesses[:5] if arbiter_output.weaknesses else []
            gt_strengths = paper_info.get('gt_strengths', [])
            gt_weaknesses = paper_info.get('gt_weaknesses', [])

            # 计算覆盖率（使用 LLM 判断）
            strength_coverage = calculate_coverage_with_llm(pred_strengths, gt_strengths, "strengths")
            weakness_coverage = calculate_coverage_with_llm(pred_weaknesses, gt_weaknesses, "weaknesses")

            return {
                "paper_id": paper.paper_id,
                "ground_truth": gt_decision,
                "prediction": {
                    "rating": round(arbiter_output.raw_rating, 2),
                    "decision": arbiter_output.decision_recommendation,
                },
                "match": gt_binary == pred_binary,
                "strengths": pred_strengths,
                "weaknesses": pred_weaknesses,
                "gt_strengths": gt_strengths,
                "gt_weaknesses": gt_weaknesses,
                "strength_coverage": strength_coverage,
                "weakness_coverage": weakness_coverage,
                "similar_cases": similar_cases_info,
                "similar_cases_stats": stats,
            }
        except Exception as e:
            if 'timeout' in str(e).lower():
                print(f"    Retry {attempt + 1}/{max_retries}...")
                time.sleep(5)
                continue
            return {"paper_id": paper_info['paper_id'], "error": str(e), "ground_truth": paper_info['decision']}
    return {"paper_id": paper_info['paper_id'], "error": "Max retries", "ground_truth": paper_info['decision']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/iclr.yaml")
    parser.add_argument("--output-dir", default="data/eval_results")
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing pipeline...")
    pipeline = ReviewPipeline(args.config)
    print("Pipeline loaded!\n")

    print("="*60)
    print("Testing CROSS VENUE: NeurIPS 2022 (with ICLR memory)")
    print("="*60)

    n_per_class = args.n_samples // 2
    papers = get_test_papers(n_accept=n_per_class, n_reject=n_per_class)
    results = []
    correct = 0
    total_strength_coverage = 0.0
    total_weakness_coverage = 0.0
    coverage_count = 0

    for i, paper_info in enumerate(papers):
        print(f"\n[{i+1}/{len(papers)}] {paper_info['paper_id']} (GT: {paper_info['decision']})")
        result = run_test(pipeline, paper_info)
        results.append(result)
        if result.get('match'):
            correct += 1

        # 累计覆盖率
        if 'strength_coverage' in result:
            total_strength_coverage += result['strength_coverage']['coverage']
            total_weakness_coverage += result['weakness_coverage']['coverage']
            coverage_count += 1

        pred = result.get('prediction', {})
        st_cov = result.get('strength_coverage', {}).get('coverage', 0)
        wk_cov = result.get('weakness_coverage', {}).get('coverage', 0)
        print(f"  -> Pred: {pred.get('decision', 'N/A')} ({pred.get('rating', 'N/A')}), Match: {result.get('match')}")
        print(f"     Coverage: Strengths {st_cov:.1%}, Weaknesses {wk_cov:.1%}")

    # 计算平均覆盖率
    avg_strength_coverage = total_strength_coverage / coverage_count if coverage_count > 0 else 0
    avg_weakness_coverage = total_weakness_coverage / coverage_count if coverage_count > 0 else 0

    output = {
        "test_type": "cross_venue",
        "source_venue": "ICLR",
        "target_venue": "NeurIPS",
        "year": 2022,
        "timestamp": datetime.now().isoformat(),
        "total": len(papers),
        "correct": correct,
        "accuracy": round(correct / len(results), 4) if results else 0,
        "metrics": {
            "avg_strength_coverage": round(avg_strength_coverage, 4),
            "avg_weakness_coverage": round(avg_weakness_coverage, 4),
        },
        "results": results,
    }

    output_path = output_dir / f"cross_venue_neurips50_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"NeurIPS 2022 Accuracy: {correct}/{len(results)} = {correct/max(len(results),1)*100:.1f}%")
    print(f"Strength Coverage: {avg_strength_coverage:.1%}")
    print(f"Weakness Coverage: {avg_weakness_coverage:.1%}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
