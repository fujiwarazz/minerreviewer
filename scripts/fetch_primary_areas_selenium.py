#!/usr/bin/env python3
"""
使用 Selenium 爬取 DeepReview-13K 论文的 Primary Area 信息

使用方法：
1. 安装: pip install selenium webdriver-manager
2. 运行: python fetch_primary_areas_selenium.py

输出：
- primary_areas.json: {paper_id: primary_area}
"""

import json
import time
import argparse
from pathlib import Path
from typing import Optional, List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def get_primary_area_selenium(paper_id: str, driver, wait) -> Optional[str]:
    """使用 Selenium 从 OpenReview 页面提取 Primary Area"""
    url = f"https://openreview.net/forum?id={paper_id}"

    try:
        driver.get(url)

        # 等待页面加载完成
        time.sleep(3)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        import re

        # 方法1: 从页面源码提取 primary area
        page_source = driver.page_source

        # 查找 "primary_area" 或 "primaryArea" 字段
        matches = re.findall(r'"primary[_-]?area"\s*:\s*"([^"]+)"', page_source, re.IGNORECASE)
        if matches:
            return matches[0]

        # 查找 JSON 中的 primary_area
        matches = re.findall(r'"primaryArea"\s*:\s*"([^"]+)"', page_source)
        if matches:
            return matches[0]

        # 方法2: 查找页面上显示的 "Primary Area" 文本
        try:
            # 找包含 "Primary Area" 的元素
            elements = driver.find_elements(By.XPATH,
                "//*[contains(text(), 'Primary Area') or contains(text(), 'primary area')]")
            for elem in elements:
                # 获取同级或下一个元素的值
                try:
                    parent = elem.find_element(By.XPATH, "..")
                    text = parent.text
                    # 解析出 Primary Area 的值
                    if text:
                        lines = text.split('\n')
                        for i, line in enumerate(lines):
                            if 'primary area' in line.lower():
                                # 返回下一行或同行冒号后的内容
                                if i + 1 < len(lines):
                                    return lines[i + 1].strip()
                                elif ':' in line:
                                    return line.split(':', 1)[1].strip()
                except:
                    continue
        except:
            pass

        return None

    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="使用 Selenium 爬取论文 Primary Area")
    parser.add_argument('--input', default='train_fast.jsonl', help='输入 JSONL 文件路径')
    parser.add_argument('--output', default='primary_areas.json', help='输出 JSON 文件路径')
    parser.add_argument('--start', type=int, default=0, help='起始索引')
    parser.add_argument('--end', type=int, default=-1, help='结束索引 (-1 表示全部)')
    parser.add_argument('--delay', type=float, default=2.0, help='请求间隔(秒)')
    parser.add_argument('--headless', action='store_true', help='无头模式运行')
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

    # 设置 Chrome
    options = Options()
    if args.headless:
        options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 10)

    try:
        for i, paper in enumerate(papers):
            paper_id = paper['id']

            if paper_id in existing:
                print(f"[{i+1}/{len(papers)}] {paper_id}: 已存在, 跳过")
                continue

            print(f"[{i+1}/{len(papers)}] 获取 {paper_id}...")
            area = get_primary_area_selenium(paper_id, driver, wait)

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

    finally:
        driver.quit()

    # 最终保存
    output_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"\n完成! 共 {len(existing)} 条记录保存到 {output_path}")

    # 统计
    found = sum(1 for v in existing.values() if v)
    print(f"成功获取: {found}/{len(existing)} ({found/len(existing)*100:.1f}%)")


if __name__ == '__main__':
    main()