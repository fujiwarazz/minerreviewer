"""对比学习模块：从 pred vs GT 差异中提炼通用规律"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from clients.llm_client import LLMClient
from common.types import ExperienceCard, Paper

logger = logging.getLogger(__name__)


class ComparisonLearner:
    """对比 pred 和 GT，提炼可迁移的经验卡片"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def learn(
        self,
        pred_strengths: list[str],
        pred_weaknesses: list[str],
        gt_strengths: list[str],
        gt_weaknesses: list[str],
        paper_info: dict,
    ) -> list[ExperienceCard]:
        """
        对比分析，生成三类卡片：

        1. missed_patterns → failure card (LLM遗漏的关键问题)
        2. angle_bias → critique card (评价角度偏差)
        3. validated_patterns → 不生成新card，只记录

        Returns:
            list of ExperienceCard (failure + critique)
        """
        # 修复：检查长度而不是布尔值（避免numpy array歧义）
        if len(gt_strengths) == 0 and len(gt_weaknesses) == 0:
            return []

        # 格式化输入
        pred_s_text = self._format_list(pred_strengths) or "(无)"
        pred_w_text = self._format_list(pred_weaknesses) or "(无)"
        gt_s_text = self._format_list(gt_strengths) or "(无)"
        gt_w_text = self._format_list(gt_weaknesses) or "(无)"

        prompt = self._build_prompt(pred_s_text, pred_w_text, gt_s_text, gt_w_text, paper_info)

        try:
            response = self.llm.generate_json(prompt)
            cards = self._parse_response(response, paper_info)
            logger.info(
                f"对比学习完成: {len(cards)} cards "
                f"(failure={sum(1 for c in cards if c.kind == 'failure')}, "
                f"critique={sum(1 for c in cards if c.kind == 'critique')}, "
                f"strength={sum(1 for c in cards if c.kind == 'strength')})"
            )
            return cards
        except Exception as e:
            logger.warning(f"对比学习失败: {e}")
            return []

    def _format_list(self, items: list) -> str:
        """格式化列表为文本"""
        # 修复：检查长度而不是布尔值
        if items is None or len(items) == 0:
            return ""
        formatted = []
        for i, item in enumerate(items, 1):
            if isinstance(item, dict):
                text = item.get("value", item.get("point", str(item)))
            else:
                text = str(item)
            formatted.append(f"{i}. {text[:200]}")
        return "\n".join(formatted)

    def _build_prompt(
        self,
        pred_s: str,
        pred_w: str,
        gt_s: str,
        gt_w: str,
        paper_info: dict,
    ) -> str:
        """构建对比分析prompt"""

        domain = paper_info.get("primary_area", "general")
        title = paper_info.get("title", "Unknown")
        decision = paper_info.get("gt_decision", "Unknown")

        # 修复：不用f-string，避免JSON中大括号的格式化问题
        return """你是审稿经验分析专家。请对比分析 AI预测评价和真实评审的差异。

## 论文信息
- Domain: {domain}
- Title: {title}
- 真实决定: {decision}

## 真实评审 (Ground Truth)
### Strengths:
{gt_s}

### Weaknesses:
{gt_w}

## AI预测评价
### Strengths:
{pred_s}

### Weaknesses:
{pred_w}

## 分析任务
请仔细对比，提取三类可迁移的经验：

1. **missed_patterns**: AI完全遗漏但GT强调的关键问题
   - 只提取真正重要、会导致拒稿的问题
   - 提炼为通用规律（不提具体论文细节）
   - 示例: "Always verify baseline coverage within 2 years for empirical papers"

2. **angle_bias**: AI评价角度有偏差（关注不该关注的，或忽略关键角度）
   - 提炼为"应该关注什么"的纠正建议
   - 示例: "For method papers, novelty outweighs implementation details"

3. **validated_patterns**: AI和GT一致的评价（仅记录，不生成新卡片）
   - 验证现有经验是否有效

## 重要过滤规则
请忽略以下内容，不要生成卡片：
- **噪声内容**: 泛泛而谈的评价（如"well written", "good experiments"）、格式问题、语言问题等次要因素
- **重复内容**: 如果AI和GT都在讨论同一个问题（即使表述不同），不算遗漏，不生成卡片
- **领域特定细节**: 包含具体数据集名、方法名、参数值的内容不可迁移，不生成卡片
- **表面差异**: GT说"A方法不好"，AI说"A方法有改进空间"，实质一致，不算遗漏

只有当AI真正**忽略了一个关键评价维度**或**评价角度完全错误**时，才生成卡片。

## 输出格式 (JSON)
输出一个JSON对象，包含三个数组：
- missed_patterns: AI遗漏的关键问题数组（最多2条，严格筛选）
- angle_bias: 评价角度偏差数组（最多1条，必须有明确纠正建议）
- validated_patterns: 验证有效的模式数组（最多2条，记录但不生成卡片）

注意：
- 每条内容要简洁（50-100字），可迁移到其他论文
- 如果没有真正重要的差异，可以返回空数组
""".format(
            domain=domain,
            title=title[:100],
            decision=decision,
            gt_s=gt_s,
            gt_w=gt_w,
            pred_s=pred_s,
            pred_w=pred_w,
        )

    def _extract_item_field(self, item, field: str, default: str = "") -> str:
        """安全提取item字段，处理dict和string两种情况"""
        if isinstance(item, dict):
            return item.get(field, default)
        elif isinstance(item, str):
            # 如果是string，直接作为content使用
            return item if field in ["general_rule", "issue", "correction", "bias", "pattern"] else default
        return default

    def _parse_response(
        self,
        response: dict,
        paper_info: dict,
    ) -> list[ExperienceCard]:
        """解析LLM响应，生成cards"""
        # 修复：检查response类型（LLM可能返回list或str）
        if not isinstance(response, dict):
            # 如果是list且包含dict，取第一个dict
            if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
                response = response[0]
                logger.info("Comparison learner returned list, extracted first dict")
            else:
                logger.warning(f"Comparison learner returned non-dict: {type(response)}, content: {str(response)[:200]}")
                return []

        cards = []
        paper_id = paper_info.get("paper_id", "unknown")
        domain = paper_info.get("primary_area", "general")

        # 1. missed_patterns → failure cards（最多2条）
        for item in response.get("missed_patterns", [])[:2]:
            severity = self._extract_item_field(item, "severity", "medium")
            utility = 0.8 if severity == "high" else 0.6 if severity == "medium" else 0.4

            card = ExperienceCard(
                card_id=str(uuid.uuid4()),
                kind="failure",
                scope="domain",
                venue_id=paper_info.get("venue_id", "ICLR"),
                theme=self._extract_item_field(item, "theme", "general"),
                content=f"⚠️ {self._extract_item_field(item, 'general_rule', self._extract_item_field(item, 'issue', ''))}",
                trigger=[],
                utility=utility,
                confidence=0.7,
                use_count=0,
                source_ids=[paper_id],
                created_at=datetime.utcnow(),
                active=True,
                source_trace={
                    "source": "comparison",
                    "type": "learned_failure",
                    "pred_count": len(paper_info.get("pred_weaknesses", [])),
                    "gt_count": len(paper_info.get("gt_weaknesses", [])),
                },
                metadata={
                    "memory_year": paper_info.get("year", 2024),
                    "memory_type": "learned_failure",
                    "memory_domain": domain,
                },
            )
            cards.append(card)

        # 2. angle_bias → critique cards（最多1条）
        for item in response.get("angle_bias", [])[:1]:
            card = ExperienceCard(
                card_id=str(uuid.uuid4()),
                kind="critique",
                scope="domain",
                venue_id=paper_info.get("venue_id", "ICLR"),
                theme=self._extract_item_field(item, "theme", "general"),
                content=f"Focus check: {self._extract_item_field(item, 'correction', self._extract_item_field(item, 'bias', ''))}",
                trigger=[],
                utility=0.5,
                confidence=0.6,
                use_count=0,
                source_ids=[paper_id],
                created_at=datetime.utcnow(),
                active=True,
                source_trace={
                    "source": "comparison",
                    "type": "learned_critique",
                },
                metadata={
                    "memory_year": paper_info.get("year", 2024),
                    "memory_type": "learned_critique",
                    "memory_domain": domain,
                },
            )
            cards.append(card)

        # 3. validated_patterns → strength cards
        for item in response.get("validated_patterns", [])[:2]:
            card = ExperienceCard(
                card_id=str(uuid.uuid4()),
                kind="strength",
                scope="domain",
                venue_id=paper_info.get("venue_id", "ICLR"),
                theme=self._extract_item_field(item, "theme", "general"),
                content=f"✓ {self._extract_item_field(item, 'pattern', '')}",
                trigger=[],
                utility=0.7,  # 验证有效的正面模式
                confidence=0.8,  # 高置信度（因为AI和GT一致）
                use_count=0,
                source_ids=[paper_id],
                created_at=datetime.utcnow(),
                active=True,
                source_trace={
                    "source": "comparison",
                    "type": "learned_strength",
                },
                metadata={
                    "memory_year": paper_info.get("year", 2024),
                    "memory_type": "learned_strength",
                    "memory_domain": domain,
                },
            )
            cards.append(card)

        return cards