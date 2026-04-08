#!/usr/bin/env python3
"""
爬取 DeepReview-13K 论文的 Primary Area 信息

使用方法：
1. 确保安装了 requests 和 beautifulsoup4
2. 在本地运行此脚本

输出：
- data/primary_areas.json: {paper_id: primary_area}
"""

import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import argparse
from typing import Optional, Dict, List, Any


def get_primary_area(paper_id: str, session: requests.Session) -> Optional[str]:
    """从 OpenReview 页面提取 primary area"""
    url = f"https://openreview.net/forum?id={paper_id}"

    try:
        # 添加完整的浏览器 headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }

        resp = session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')

        # 方法1: 查找 keywords 元数据
        keywords_meta = soup.find('meta', {'name': 'keywords'})
        if keywords_meta and keywords_meta.get('content'):
            return keywords_meta['content']

        # 方法2: 查找页面中的 primary area 标签
        # OpenReview 通常在页面中显示 topics/keywords
        for elem in soup.find_all(['span', 'div', 'p']):
            text = elem.get_text(strip=True)
            if 'primary area' in text.lower() or 'keywords' in text.lower():
                # 提取相关内容
                parent = elem.parent
                if parent:
                    content = parent.get_text(strip=True)
                    if len(content) < 200:
                        return content

        # 方法3: 查找 JSON-LD 数据
        script = soup.find('script', {'type': 'application/ld+json'})
        if script:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    if 'keywords' in data:
                        return data['keywords']
                    if 'about' in data:
                        return data['about']
            except:
                pass

        # 方法4: 直接解析页面数据 - OpenReview 将数据嵌入在 JavaScript 中
        for script in soup.find_all('script'):
            if script.string and 'primaryArea' in script.string:
                # 尝试提取 primaryArea 字段
                import re
                match = re.search(r'"primaryArea"\s*:\s*"([^"]+)"', script.string)
                if match:
                    return match.group(1)

                match = re.search(r"'primaryArea'\s*:\s*'([^']+)'", script.string)
                if match:
                    return match.group(1)

        # 方法5: 查找 topics 字段
        for script in soup.find_all('script'):
            if script.string and 'topics' in script.string:
                import re
                match = re.search(r'"topics"\s*:\s*\[([^\]]+)\]', script.string)
                if match:
                    topics_str = match.group(1)
                    # 提取所有 topic
                    topics = re.findall(r'"([^"]+)"', topics_str)
                    if topics:
                        return ', '.join(topics[:3])  # 取前3个

        return None

    except Exception as e:
        print(f"  Error fetching {paper_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="爬取论文 Primary Area")
    parser.add_argument('--input', default='/mnt/data/zzh/datasets/DeepReview-13K/data/stru/train_fast.jsonl',
                        help='输入 JSONL 文件路径')
    parser.add_argument('--output', default='primary_areas.json',
                        help='输出 JSON 文件路径')
    parser.add_argument('--start', type=int, default=0, help='起始索引')
    parser.add_argument('--end', type=int, default=-1, help='结束索引 (-1 表示全部)')
    parser.add_argument('--delay', type=float, default=1.0, help='请求间隔(秒)')
    args = parser.parse_args()

    # 读取已处理的数据
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

    if args.end > 0:
        papers = papers[args.start:args.end]
    else:
        papers = papers[args.start:]

    print(f"待处理论文数: {len(papers)}")

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    for i, paper in enumerate(papers):
        paper_id = paper['id']

        if paper_id in existing:
            print(f"[{i+1}/{len(papers)}] {paper_id}: 已存在, 跳过")
            continue

        print(f"[{i+1}/{len(papers)}] 获取 {paper_id}...")
        area = get_primary_area(paper_id, session)

        if area:
            existing[paper_id] = area
            print(f"  -> {area[:100]}..." if len(str(area)) > 100 else f"  -> {area}")
        else:
            existing[paper_id] = None
            print(f"  -> 未找到")

        # 定期保存
        if (i + 1) % 10 == 0:
            output_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
            print(f"  已保存到 {output_path}")

        time.sleep(args.delay)

    # 最终保存
    output_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"\n完成! 共 {len(existing)} 条记录保存到 {output_path}")

    # 统计
    found = sum(1 for v in existing.values() if v)
    print(f"成功获取: {found}/{len(existing)} ({found/len(existing)*100:.1f}%)")


if __name__ == '__main__':
    main()