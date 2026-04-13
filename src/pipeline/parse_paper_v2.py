"""PaperParserV2: 基于 primary_area 的论文解析器"""
from __future__ import annotations

import logging
from typing import Any

from clients.llm_client import LLMClient
from common.types import Paper, PaperSignature

logger = logging.getLogger(__name__)


# Primary Area 类别定义（基于 ICLR 2024-2025 submission categories）
PRIMARY_AREAS = [
    "generative models",
    "unsupervised, self-supervised, semi-supervised, and supervised representation learning",
    "applications to computer vision, audio, language, and other modalities",
    "foundation or frontier models, including LLMs",
    "reinforcement learning",
    "alignment, fairness, safety, privacy, and societal considerations",
    "datasets and benchmarks",
    "applications to physical sciences (physics, chemistry, biology, etc.)",
    "optimization",
    "representation learning for computer vision, audio, language, and other modalities",
    "transfer learning, meta learning, and lifelong learning",
    "learning on graphs and other geometries & topologies",
    "general machine learning (i.e., none of the above)",
    "other topics in machine learning (i.e., none of the above)",
    "interpretability and explainable AI",
    "learning theory",
    "societal considerations including fairness, safety, privacy",
    "probabilistic methods (Bayesian methods, variational inference, sampling, UQ, etc.)",
    "applications to robotics, autonomy, planning",
    "applications to neuroscience & cognitive science",
    "learning on time series and dynamical systems",
    "causal reasoning",
    "neurosymbolic & hybrid AI systems (physics-informed, logic & formal reasoning, etc.)",
    "visualization or interpretation of learned representations",
    "infrastructure, software libraries, hardware, systems, etc.",
    "infrastructure, software libraries, hardware, etc.",
    "metric learning, kernel learning, and sparse coding",
]

# 映射：primary_area -> 简化 domain
AREA_TO_DOMAIN_MAP = {
    "generative models": "generative",
    "unsupervised, self-supervised, semi-supervised, and supervised representation learning": "representation_learning",
    "applications to computer vision, audio, language, and other modalities": "vision",
    "foundation or frontier models, including LLMs": "llm",
    "reinforcement learning": "rl",
    "alignment, fairness, safety, privacy, and societal considerations": "alignment",
    "datasets and benchmarks": "benchmark",
    "applications to physical sciences (physics, chemistry, biology, etc.)": "science",
    "optimization": "optimization",
    "representation learning for computer vision, audio, language, and other modalities": "vision_rep",
    "transfer learning, meta learning, and lifelong learning": "transfer",
    "learning on graphs and other geometries & topologies": "graphs",
    "general machine learning (i.e., none of the above)": "general_ml",
    "other topics in machine learning (i.e., none of the above)": "other",
    "interpretability and explainable AI": "interpretability",
    "learning theory": "theory",
    "societal considerations including fairness, safety, privacy": "societal",
    "probabilistic methods (Bayesian methods, variational inference, sampling, UQ, etc.)": "probabilistic",
    "applications to robotics, autonomy, planning": "robotics",
    "applications to neuroscience & cognitive science": "neuroscience",
    "learning on time series and dynamical systems": "time_series",
    "causal reasoning": "causal",
    "neurosymbolic & hybrid AI systems (physics-informed, logic & formal reasoning, etc.)": "neurosymbolic",
    "visualization or interpretation of learned representations": "visualization",
    "infrastructure, software libraries, hardware, systems, etc.": "infrastructure",
    "infrastructure, software libraries, hardware, etc.": "infrastructure",
    "metric learning, kernel learning, and sparse coding": "metric_learning",
}

# 选择指南（替代 few-shot）
SELECTION_GUIDE = """
Selection Guidelines:
1. Focus on the MAIN CONTRIBUTION: What is the core method or application domain?
2. Distinguish similar categories:
   - "generative models" vs "foundation models, including LLMs":
     → Use "generative models" for GAN/diffusion/VAE architecture research
     → Use "foundation models" for LLM/foundation model research itself
   - "applications to X" vs "representation learning for X":
     → Use "applications to X" when applying existing methods
     → Use "representation learning for X" when proposing new methods
   - "alignment, fairness, safety..." vs "societal considerations...":
     → Use "alignment..." for comprehensive societal impact
     → Use "societal considerations" for specific fairness/safety topics
3. If unsure, choose "general machine learning"
"""


class PaperParserV2:
    """基于 primary_area 的论文解析器"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.last_raw_response: dict | None = None

    def parse(self, paper: Paper) -> PaperSignature:
        """解析论文"""
        prompt = self._build_prompt(paper)
        try:
            response = self.llm.generate_json(prompt)
            self.last_raw_response = response
            return self._parse_response(response)
        except Exception as e:
            logger.warning("Failed to parse paper %s: %s", paper.paper_id, e)
            return PaperSignature()

    def _build_prompt(self, paper: Paper) -> str:
        """构建 prompt（简化版）"""
        lines = [
            "Extract features from this paper. Return JSON with:",
            "- paper_type: [empirical, theoretical, survey, system, analysis, benchmark, other]",
            "- tasks: list (e.g., classification, generation)",
            "- primary_area: EXACT name from categories below (NOT a number)",
            "- method_family: list (e.g., transformer, gan, diffusion, gnn)",
            "- main_claims: list (max 3)",
            "",
            "=== Categories ===",
        ]
        for area in PRIMARY_AREAS:
            lines.append(f"  {area}")

        lines.append("")
        lines.append(SELECTION_GUIDE.strip())
        lines.append("")
        lines.append("=== Paper ===")
        lines.append(f"Title: {paper.title}")
        lines.append(f"Abstract: {paper.abstract}")
        lines.append("")
        lines.append("Return JSON. Use EXACT category name.")

        return "\n".join(lines)

    def _parse_response(self, response: dict[str, Any]) -> PaperSignature:
        """解析 LLM 响应"""
        primary_area = response.get("primary_area", "")

        # 处理编号
        if isinstance(primary_area, int):
            idx = primary_area - 1
            if 0 <= idx < len(PRIMARY_AREAS):
                primary_area = PRIMARY_AREAS[idx]
            else:
                primary_area = "other"

        # 处理不在列表中的情况
        if primary_area and primary_area not in PRIMARY_AREAS:
            primary_area = self._find_closest_area(primary_area)

        domain = AREA_TO_DOMAIN_MAP.get(primary_area, "other")

        return PaperSignature(
            paper_type=response.get("paper_type"),
            tasks=response.get("tasks", []),
            domain=domain,
            method_family=response.get("method_family", []),
            main_claims=response.get("main_claims", []),
        )

    def _find_closest_area(self, guessed: str) -> str:
        """关键词匹配"""
        if not isinstance(guessed, str):
            return "other"

        keywords = {
            "generative": "generative models",
            "self-supervised": "unsupervised, self-supervised, semi-supervised, and supervised representation learning",
            "vision": "applications to computer vision, audio, language, and other modalities",
            "llm": "foundation or frontier models, including LLMs",
            "reinforcement": "reinforcement learning",
            "rl": "reinforcement learning",
            "fairness": "alignment, fairness, safety, privacy, and societal considerations",
            "benchmark": "datasets and benchmarks",
            "physics": "applications to physical sciences (physics, chemistry, biology, etc.)",
            "optim": "optimization",
            "transfer": "transfer learning, meta learning, and lifelong learning",
            "graph": "learning on graphs and other geometries & topologies",
            "gnn": "learning on graphs and other geometries & topologies",
            "theory": "learning theory",
            "bayesian": "probabilistic methods (Bayesian methods, variational inference, sampling, UQ, etc.)",
            "robot": "applications to robotics, autonomy, planning",
            "time series": "learning on time series and dynamical systems",
            "causal": "causal reasoning",
            "interpret": "interpretability and explainable AI",
        }

        lower = guessed.lower()
        for kw, area in keywords.items():
            if kw in lower:
                return area
        return guessed

    def parse_with_known_area(self, paper: Paper, known_area: str) -> PaperSignature:
        """已知 primary_area 时使用"""
        domain = AREA_TO_DOMAIN_MAP.get(known_area, "other")

        prompt = "\n".join([
            "Extract features. Return JSON:",
            "- paper_type: [empirical, theoretical, survey, system, analysis, benchmark, other]",
            "- tasks: list",
            "- method_family: list",
            "- main_claims: list",
            "",
            f"Title: {paper.title}",
            f"Abstract: {paper.abstract}",
            "Return JSON.",
        ])

        try:
            response = self.llm.generate_json(prompt)
            return PaperSignature(
                paper_type=response.get("paper_type"),
                tasks=response.get("tasks", []),
                domain=domain,
                method_family=response.get("method_family", []),
                main_claims=response.get("main_claims", []),
            )
        except Exception as e:
            logger.warning("Failed: %s", e)
            return PaperSignature(domain=domain)