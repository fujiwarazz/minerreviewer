"""AgentMemoryAllocator: 将对比学习结果分配给对应agent

策略：
1. 根据卡片类型和主题推断目标agent
2. 支持多agent共享同一卡片
3. 记录分配理由到metadata
"""
from __future__ import annotations

import logging
from typing import Literal

from common.types import ExperienceCard

logger = logging.getLogger(__name__)


class AgentMemoryAllocator:
    """将对比学习结果分配给对应agent"""

    # Agent映射规则：theme -> agent_name
    AGENT_MAPPING = {
        # Theme agents
        "quality": "theme_quality",
        "novelty": "theme_originality",
        "originality": "theme_originality",
        "clarity": "theme_clarity",
        "significance": "theme_significance",
        "experiments": "theme_experiments",
        "reproducibility": "theme_experiments",
        "soundness": "theme_quality",
        "presentation": "theme_clarity",
        "technical": "theme_quality",
        "empirical": "theme_experiments",

        # Arbiter agent
        "decision": "arbiter",
        "rating": "arbiter",
        "overall": "arbiter",
    }

    # Kind默认分配规则
    KIND_DEFAULT_AGENTS = {
        "strength": ["theme_quality", "arbiter"],  # strength共享给quality + arbiter
        "critique": ["theme_quality"],  # critique默认给quality
        "failure": ["arbiter"],  # failure主要给arbiter参考
    }

    def allocate(
        self,
        cards: list[ExperienceCard],
        share_strength_with_arbiter: bool = True,
    ) -> dict[str, list[ExperienceCard]]:
        """分配卡片到各agent

        Args:
            cards: 待分配的卡片列表
            share_strength_with_arbiter: strength卡片是否共享给arbiter

        Returns:
            dict mapping agent_name to list of cards
        """
        allocation: dict[str, list[ExperienceCard]] = {}

        for card in cards:
            target_agents = self._infer_target_agents(
                card,
                share_strength_with_arbiter=share_strength_with_arbiter,
            )

            for agent_name in target_agents:
                # 创建卡片副本，标记owner_agent
                card_copy = card.model_copy()
                card_copy.owner_agent = agent_name
                card_copy.metadata["allocation_reason"] = self._get_allocation_reason(card)

                if agent_name not in allocation:
                    allocation[agent_name] = []
                allocation[agent_name].append(card_copy)

        logger.info(
            "Allocated %d cards to %d agents",
            sum(len(cards) for cards in allocation.values()),
            len(allocation),
        )

        # 打印分配详情
        for agent_name, agent_cards in allocation.items():
            logger.info(
                "  Agent '%s': %d cards (%s)",
                agent_name,
                len(agent_cards),
                ", ".join(set(c.kind for c in agent_cards)),
            )

        return allocation

    def _infer_target_agents(
        self,
        card: ExperienceCard,
        share_strength_with_arbiter: bool = True,
    ) -> list[str]:
        """推断卡片应该分配给哪些agent"""
        agents = []

        # 1. 根据主题推断
        theme_lower = card.theme.lower() if card.theme else ""
        for key, agent_name in self.AGENT_MAPPING.items():
            if key in theme_lower:
                agents.append(agent_name)
                break

        # 2. 根据类型推断（如果没有从theme推断出来）
        if not agents:
            default_agents = self.KIND_DEFAULT_AGENTS.get(card.kind, ["theme_quality"])
            agents.extend(default_agents)

        # 3. strength卡片特殊处理：是否共享给arbiter
        if card.kind == "strength":
            if share_strength_with_arbiter and "arbiter" not in agents:
                agents.append("arbiter")

        # 4. failure卡片：强制给arbiter
        if card.kind == "failure":
            if "arbiter" not in agents:
                agents.append("arbiter")
            # 移除其他agent（failure只给arbiter）
            agents = ["arbiter"]

        # 5. 去重
        agents = list(set(agents))

        return agents

    def _get_allocation_reason(self, card: ExperienceCard) -> str:
        """获取分配理由"""
        reasons = []

        if card.theme:
            reasons.append(f"theme={card.theme}")

        reasons.append(f"kind={card.kind}")

        if card.metadata.get("memory_type"):
            reasons.append(f"type={card.metadata['memory_type']}")

        return "allocated_by_" + "_".join(reasons)

    def get_supported_agents(self) -> list[str]:
        """获取支持的agent列表"""
        return list(set(self.AGENT_MAPPING.values()))

    def get_supported_themes(self) -> list[str]:
        """获取支持的主题列表"""
        return list(self.AGENT_MAPPING.keys())

    def allocate_by_theme(
        self,
        cards: list[ExperienceCard],
        theme: str,
    ) -> list[ExperienceCard]:
        """按主题分配卡片

        Args:
            cards: 待分配的卡片列表
            theme: 目标主题

        Returns:
            分配给该主题的卡片列表
        """
        agent_name = self.AGENT_MAPPING.get(theme.lower(), "theme_quality")

        allocated_cards = []
        for card in cards:
            card_copy = card.model_copy()
            card_copy.owner_agent = agent_name
            card_copy.metadata["allocation_reason"] = f"forced_theme_{theme}"
            allocated_cards.append(card_copy)

        return allocated_cards