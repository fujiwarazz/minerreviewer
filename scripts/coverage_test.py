#!/usr/bin/env python
"""
测试覆盖率计算脚本
计算预测的 strengths/weaknesses 与真实 review 的覆盖率
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_ROOT = Path("/mnt/data/zzh/datasets/reviewer2")


def extract_strengths_weaknesses(sw_text: str) -> tuple[list[str], list[str]]:
    """从 review 文本中提取 strengths 和 weaknesses"""
    if not sw_text:
        return [], []
    
    strengths = []
    weaknesses = []
    
    # 常见分隔符
    text_lower = sw_text.lower()
    
    # 尝试找 Strengths 和 Weaknesses 部分
    # Pattern 1: "Strengths\n...\nWeaknesses\n..."
    # Pattern 2: "Pros:\n...\nCons:\n..."
    # Pattern 3: 直接解析
    
    parts = re.split(r'\n(?=(?:weaknesses?|cons?|limitations?)\s*[:\n])', text_lower, flags=re.IGNORECASE)
    
    if len(parts) == 2:
        strength_part = parts[0]
        weakness_part = parts[1]
        
        # 提取 strengths
        strength_lines = re.split(r'\n(?=[-•*]|\d+\.)', strength_part)
        for line in strength_lines:
            line = line.strip()
            # 跳过标题
            if 'strengths' in line.lower() or 'pros' in line.lower():
                continue
            if len(line) > 20:
                strengths.append(line)
        
        # 提取 weaknesses
        weakness_lines = re.split(r'\n(?=[-•*]|\d+\.)', weakness_part)
        for line in weakness_lines:
            line = line.strip()
            if 'weakness' in line.lower() or 'cons' in line.lower() or 'limitation' in line.lower():
                continue
            if len(line) > 20:
                weaknesses.append(line)
    else:
        # 尝试按 bullet points 分割
        lines = sw_text.split('\n')
        current_type = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if 'strength' in line.lower() or 'pros' in line.lower():
                current_type = 'strength'
                continue
            if 'weakness' in line.lower() or 'cons' in line.lower() or 'limitation' in line.lower():
                current_type = 'weakness'
                continue
            if line.startswith(('- ', '• ', '* ')) or re.match(r'^\d+\.', line):
                if current_type == 'strength':
                    strengths.append(line)
                elif current_type == 'weakness':
                    weaknesses.append(line)
    
    return strengths, weaknesses


def calculate_coverage(pred_items: list[str], gt_items: list[str], threshold: float = 0.3) -> dict:
    """计算覆盖率（基于关键词重叠）"""
    if not gt_items:
        return {"coverage": 1.0, "matched": 0, "total": 0}
    
    if not pred_items:
        return {"coverage": 0.0, "matched": 0, "total": len(gt_items)}
    
    def extract_keywords(text: str) -> set:
        """提取关键词"""
        # 简单的关键词提取：移除停用词，保留有意义词汇
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                      'during', 'before', 'after', 'above', 'below', 'between', 'under',
                      'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                      'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                      'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until',
                      'while', 'this', 'that', 'these', 'those', 'it', 'its', 'paper'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return set(w for w in words if w not in stop_words)
    
    matched = 0
    for gt_item in gt_items:
        gt_keywords = extract_keywords(gt_item)
        if not gt_keywords:
            continue
        
        best_overlap = 0
        for pred_item in pred_items:
            pred_keywords = extract_keywords(pred_item)
            if not pred_keywords:
                continue
            
            # 计算 Jaccard 相似度
            intersection = len(gt_keywords & pred_keywords)
            union = len(gt_keywords | pred_keywords)
            overlap = intersection / max(union, 1)
            best_overlap = max(best_overlap, overlap)
        
        if best_overlap >= threshold:
            matched += 1
    
    coverage = matched / len(gt_items) if gt_items else 1.0
    return {"coverage": coverage, "matched": matched, "total": len(gt_items)}


def load_ground_truth(paper_id: str) -> dict:
    """加载真实 review 数据"""
    venue_path = DATA_ROOT / 'NIPS'
    files = list(venue_path.glob(f'**/{paper_id}_review/*.json'))
    if not files:
        return {"strengths": [], "weaknesses": []}
    
    with open(files[0]) as f:
        data = json.load(f)
    
    all_strengths = []
    all_weaknesses = []
    
    reviews = data.get('reviews', [])
    for review in reviews:
        sw_text = review.get('Strengths And Weaknesses:', '')
        if sw_text:
            s, w = extract_strengths_weaknesses(sw_text)
            all_strengths.extend(s)
            all_weaknesses.extend(w)
    
    return {"strengths": all_strengths, "weaknesses": all_weaknesses}


def analyze_coverage(results_path: str):
    """分析覆盖率"""
    with open(results_path) as f:
        data = json.load(f)
    
    results = data['results']
    
    strength_coverages = []
    weakness_coverages = []
    
    detailed_results = []
    
    for r in results:
        paper_id = r.get('paper_id')
        if not paper_id:
            continue
        
        # 加载真实数据
        gt = load_ground_truth(paper_id)
        
        # 获取预测的 strengths/weaknesses（如果有）
        pred_strengths = r.get('pred_strengths', [])
        pred_weaknesses = r.get('pred_weaknesses', [])
        
        if gt['strengths'] or gt['weaknesses']:
            # 计算覆盖率
            s_cov = calculate_coverage(pred_strengths, gt['strengths'])
            w_cov = calculate_coverage(pred_weaknesses, gt['weaknesses'])
            
            strength_coverages.append(s_cov['coverage'])
            weakness_coverages.append(w_cov['coverage'])
            
            detailed_results.append({
                "paper_id": paper_id,
                "gt_decision": r.get('ground_truth'),
                "strength_coverage": s_cov,
                "weakness_coverage": w_cov,
                "gt_strengths_count": len(gt['strengths']),
                "gt_weaknesses_count": len(gt['weaknesses']),
            })
    
    # 汇总统计
    import statistics
    print("="*60)
    print("覆盖率分析")
    print("="*60)
    
    print(f"\n有效样本数: {len(strength_coverages)}")
    
    if strength_coverages:
        print(f"\nStrength 覆盖率:")
        print(f"  平均: {statistics.mean(strength_coverages)*100:.1f}%")
        print(f"  中位数: {statistics.median(strength_coverages)*100:.1f}%")
        print(f"  范围: {min(strength_coverages)*100:.1f}% - {max(strength_coverages)*100:.1f}%")
    
    if weakness_coverages:
        print(f"\nWeakness 覆盖率:")
        print(f"  平均: {statistics.mean(weakness_coverages)*100:.1f}%")
        print(f"  中位数: {statistics.median(weakness_coverages)*100:.1f}%")
        print(f"  范围: {min(weakness_coverages)*100:.1f}% - {max(weakness_coverages)*100:.1f}%")
    
    # 显示部分详细结果
    print("\n详细结果示例:")
    for item in detailed_results[:5]:
        print(f"\n{item['paper_id']} ({item['gt_decision']})")
        print(f"  Strength: {item['strength_coverage']['matched']}/{item['strength_coverage']['total']} = {item['strength_coverage']['coverage']*100:.0f}%")
        print(f"  Weakness: {item['weakness_coverage']['matched']}/{item['weakness_coverage']['total']} = {item['weakness_coverage']['coverage']*100:.0f}%")
    
    return {
        "strength_coverage_mean": statistics.mean(strength_coverages) if strength_coverages else 0,
        "weakness_coverage_mean": statistics.mean(weakness_coverages) if weakness_coverages else 0,
        "detailed_results": detailed_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results JSON")
    args = parser.parse_args()
    
    analyze_coverage(args.results)
