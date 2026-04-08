# DeepReview-13K 记忆库使用指南

## 概述

DeepReview-13K 是一个包含 13,379 条论文评审记录的数据集，来自多个机器学习会议。已集成到 MinerReview 作为热插拔记忆源，提供额外的论文评审案例支持。

## 特点

- **13,379 条论文案例**：涵盖 27 个不同研究方向（primary_area）
- **primary_area 字段**：论文研究方向标签，支持按领域过滤检索
- **热插拔设计**：可在配置中启用/禁用，不影响主记忆系统
- **hybrid retrieval**：支持 embedding + signature + primary_area 融合检索

## 构建 DeepReview 记忆库

### 1. 从原始 JSONL 构建 PaperCase

```bash
# 跳过 embedding（快速构建）
python scripts/build_deepreview_memory.py --skip-embedding \
    --input /mnt/data/zzh/datasets/DeepReview-13K/data/stru/dataset_with_area.jsonl \
    --output data/processed/deepreview_cases_full.jsonl

# 生成 embedding（需要 embedding 服务）
python scripts/build_deepreview_memory.py \
    --config configs/iclr.yaml \
    --input /mnt/data/zzh/datasets/DeepReview-13K/data/stru/dataset_with_area.jsonl \
    --output data/processed/deepreview_cases_full.jsonl
```

### 2. 构建结果

```
2026-04-08 - INFO - Built 13379 cases
2026-04-08 - INFO - Primary areas: 27 unique
2026-04-08 - INFO -   generative models: 1284 cases
2026-04-08 - INFO -   reinforcement learning: 890 cases
2026-04-08 - INFO -   foundation or frontier models, including LLMs: 1014 cases
...
```

## 启用 DeepReview 记忆库

在配置文件中启用：

```yaml
memory:
  # DeepReview-13K 热插拔记忆库
  deepreview:
    enabled: true  # 启用 DeepReview 记忆库
    path: "data/processed/deepreview_cases_full.jsonl"
    primary_area_weight: 0.1  # primary_area 匹配权重
```

## 检索流程

启用后，DeepReview 记忆库会作为额外案例源：

1. **主记忆库检索**：从 ICLR/NeurIPS 等会议记忆库检索
2. **DeepReview 补充检索**：从 DeepReview-13K 检索相似案例
3. **合并结果**：两源案例合并，提供更丰富的评审参考

### 按 primary_area 过滤

如果目标论文有 `primary_area` 信息，检索时会优先匹配同领域案例：

```python
deepreview_results = deepreview_store.retrieve_cases(
    query_text=query_text,
    primary_area="generative models",  # 可选：按研究方向过滤
    ...
)
```

## 数据结构

每条 PaperCase 包含：

```json
{
  "case_id": "deepreview_xxx",
  "paper_id": "xxx",
  "venue_id": "DeepReview",
  "year": 2024,
  "title": "...",
  "abstract": "...",
  "primary_area": "generative models",  // 论文研究方向
  "top_strengths": ["...", "..."],
  "top_weaknesses": ["...", "..."],
  "decision": "Reject",
  "rating": 6.8
}
```

## 27 个 primary_area 类别

| 类别 | 数量 |
|------|------|
| generative models | 1,284 |
| unsupervised/self-supervised learning | 1,185 |
| applications to vision/audio/language | 1,138 |
| foundation/LLMs | 1,014 |
| reinforcement learning | 890 |
| alignment/fairness/safety | 838 |
| datasets and benchmarks | 787 |
| ... | ... |

## 与主记忆库的区别

| 特点 | 主记忆库 (ICLR/NeurIPS) | DeepReview 记忆库 |
|------|------------------------|-------------------|
| 数据来源 | 单一会议 | 多会议混合 |
| primary_area | 无 | 有（27 类） |
| venue_id | ICLR/NeurIPS | DeepReview |
| 检索权重 | 主要源 | 补充源 |
| 配置 | registry.json | config.yaml |

## 热插拔特性

- **可随时启用/禁用**：修改配置即可
- **不影响主系统**：独立的存储和检索
- **可选 embedding**：可跳过 embedding 生成快速构建
- **可扩展**：未来可添加更多外部数据集