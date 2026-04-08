#!/usr/bin/env python3
"""
使用 OpenReview API 获取论文 Primary Area

API 端点: https://api2.openreview.net/notes?paperhash={author}|{title}

使用方法:
    python fetch_primary_areas_api.py --input train_fast.jsonl --output primary_areas.json
"""

import json
import time
import re
import requests
import argparse
from pathlib import Path
from typing import Optional
from urllib.parse import quote


def extract_title_from_paper(paper_content: str) -> Optional[str]:
    """从 paper 内容中提取 title"""
    # LaTeX 格式: \title{...}
    match = re.search(r'\\title\{([^}]+)\}', paper_content)
    if match:
        return match.group(1).strip()

    # Markdown 格式: # Title
    match = re.search(r'^#\s+(.+)$', paper_content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # 尝试从开头提取
    lines = paper_content.strip().split('\n')
    for line in lines[:5]:
        line = line.strip()
        if line and not line.startswith('\\') and len(line) > 10:
            return line

    return None


def normalize_title_for_hash(title: str) -> str:
    """将 title 转换为 paperhash 格式"""
    # 转小写
    title = title.lower()
    # 移除特殊字符，只保留字母数字
    title = re.sub(r'[^a-z0-9\s]', '', title)
    # 空格替换为下划线
    title = title.replace(' ', '_')
    # 合并多个下划线
    title = re.sub(r'_+', '_', title)
    return title


def get_primary_area_via_api(paper_id: str, title: str, session: requests.Session) -> Optional[str]:
    """通过 OpenReview API 获取 primary area"""

    # 构造 paperhash (尝试多种格式)
    title_normalized = normalize_title_for_hash(title)

    # 尝试不同的 author 格式
    # 格式1: 只有 title
    paperhash_variants = [
        title_normalized,
        f"anonymous|{title_normalized}",
        f"unknown|{title_normalized}",
    ]

    for paperhash in paperhash_variants:
        try:
            url = f"https://api2.openreview.net/notes?paperhash={quote(paperhash)}"
            resp = session.get(url, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                notes = data.get('notes', [])
                if notes:
                    # 提取 primary_area
                    content = notes[0].get('content', {})
                    primary_area = content.get('primary_area', {})
                    if isinstance(primary_area, dict):
                        value = primary_area.get('value')
                        if value:
                            return value
                    elif isinstance(primary_area, str):
                        return primary_area
        except Exception as e:
            pass

    # 直接用 paper_id 尝试
    try:
        url = f"https://api2.openreview.net/notes?id={paper_id}"
        resp = session.get(url, timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            notes = data.get('notes', [])
            if notes:
                content = notes[0].get('content', {})
                primary_area = content.get('primary_area', {})
                if isinstance(primary_area, dict):
                    return primary_area.get('value')
                elif isinstance(primary_area, str):
                    return primary_area
    except:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(description="通过 API 获取论文 Primary Area")
    parser.add_argument('--input', default='train_fast.jsonl', help='输入 JSONL 文件')
    parser.add_argument('--output', default='primary_areas.json', help='输出 JSON 文件')
    parser.add_argument('--delay', type=float, default=0.5, help='请求间隔(秒)')
    parser.add_argument('--limit', type=int, default=0, help='限制处理数量 (0=全部)')
    args = parser.parse_args()

    # 读取已有数据
    output_path = Path(args.output)
    existing = {}
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        print(f"已加载 {len(existing)} 条已有记录")

    # 读取论文列表
    papers = []
    with open(args.input) as f:
        for line in f:
            papers.append(json.loads(line))

    if args.limit > 0:
        papers = papers[:args.limit]

    print(f"待处理论文数: {len(papers)}")

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json',
    })

    stats = {'found': 0, 'not_found': 0, 'error': 0}

    for i, paper in enumerate(papers):
        paper_id = paper['id']

        if paper_id in existing:
            print(f"[{i+1}/{len(papers)}] {paper_id}: 已存在")
            continue

        # 提取 title
        paper_content = paper.get('paper', '')
        title = extract_title_from_paper(paper_content)

        if not title:
            print(f"[{i+1}/{len(papers)}] {paper_id}: 无法提取 title")
            existing[paper_id] = None
            stats['error'] += 1
            continue

        print(f"[{i+1}/{len(papers)}] {paper_id}: {title[:50]}...")

        area = get_primary_area_via_api(paper_id, title, session)

        if area:
            existing[paper_id] = area
            print(f"  -> {area}")
            stats['found'] += 1
        else:
            existing[paper_id] = None
            print(f"  -> 未找到")
            stats['not_found'] += 1

        # 定期保存
        if (i + 1) % 20 == 0:
            output_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
            print(f"  已保存 ({stats['found']} found, {stats['not_found']} not found)")

        time.sleep(args.delay)

    # 最终保存
    output_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))

    print(f"\n完成!")
    print(f"  找到: {stats['found']}")
    print(f"  未找到: {stats['not_found']}")
    print(f"  错误: {stats['error']}")
    print(f"  保存到: {output_path}")


if __name__ == '__main__':
    main()