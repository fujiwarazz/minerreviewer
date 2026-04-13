"""PaperParserV3: 两阶段推断的论文解析器

第一阶段：分析论文的技术领域（关键词、方法、任务）
第二阶段：根据分析结果映射到 primary_area
"""
from __future__ import annotations

import logging
from typing import Any

from clients.llm_client import LLMClient
from common.types import Paper, PaperSignature

logger = logging.getLogger(__name__)


# Primary Area 类别
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

# 简化 domain -> primary_area（反向映射）
DOMAIN_TO_AREA_MAP = {
    "generative": "generative models",
    "representation_learning": "unsupervised, self-supervised, semi-supervised, and supervised representation learning",
    "vision": "applications to computer vision, audio, language, and other modalities",
    "llm": "foundation or frontier models, including LLMs",
    "rl": "reinforcement learning",
    "alignment": "alignment, fairness, safety, privacy, and societal considerations",
    "benchmark": "datasets and benchmarks",
    "science": "applications to physical sciences (physics, chemistry, biology, etc.)",
    "optimization": "optimization",
    "vision_rep": "representation learning for computer vision, audio, language, and other modalities",
    "transfer": "transfer learning, meta learning, and lifelong learning",
    "graphs": "learning on graphs and other geometries & topologies",
    "general_ml": "general machine learning (i.e., none of the above)",
    "other": "other topics in machine learning (i.e., none of the above)",
    "interpretability": "visualization or interpretation of learned representations",  # 语义相同
    "visualization": "visualization or interpretation of learned representations",
    "theory": "learning theory",
    "societal": "societal considerations including fairness, safety, privacy",
    "privacy": "societal considerations including fairness, safety, privacy",
    "security": "societal considerations including fairness, safety, privacy",
    "fairness": "alignment, fairness, safety, privacy, and societal considerations",
    "safety": "alignment, fairness, safety, privacy, and societal considerations",
    "probabilistic": "probabilistic methods (Bayesian methods, variational inference, sampling, UQ, etc.)",
    "robotics": "applications to robotics, autonomy, planning",
    "neuroscience": "applications to neuroscience & cognitive science",
    "time_series": "learning on time series and dynamical systems",
    "causal": "causal reasoning",
    "neurosymbolic": "neurosymbolic & hybrid AI systems (physics-informed, logic & formal reasoning, etc.)",
    "infrastructure": "infrastructure, software libraries, hardware, systems, etc.",
    "metric_learning": "metric learning, kernel learning, and sparse coding",
}


class PaperParserV3:
    """两阶段推断的论文解析器"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.last_raw_response: dict | None = None

    def parse(self, paper: Paper) -> PaperSignature:
        """两阶段解析"""
        # 第一阶段：分析论文技术特征
        analysis = self._analyze_paper(paper)

        # 第二阶段：映射到 primary_area
        primary_area = self._map_to_primary_area(analysis)

        domain = AREA_TO_DOMAIN_MAP.get(primary_area, "other")

        return PaperSignature(
            paper_type=analysis.get("paper_type"),
            tasks=analysis.get("tasks", []),
            domain=domain,
            method_family=analysis.get("method_family", []),
            main_claims=analysis.get("main_claims", []),
        )

    def _analyze_paper(self, paper: Paper) -> dict:
        """第一阶段：分析论文的技术特征"""
        prompt = "\n".join([
            "Analyze this ML paper. Return JSON with:",
            "- paper_type: [empirical, theoretical, survey, system, analysis, benchmark]",
            "- domain: main technical domain from: generative, rl, graphs, llm, vision, vision_rep, optimization, theory, probabilistic, causal, time_series, robotics, neuroscience, interpretability, societal, benchmark, science, transfer, general_ml",
            "- tasks: list of tasks",
            "- method_family: list",
            "- main_claims: list (max 3)",
            "",
            f"Title: {paper.title}",
            f"Abstract: {paper.abstract}",
            "",
            "Return JSON. Choose the most appropriate domain.",
        ])

        try:
            response = self.llm.generate_json(prompt)
            self.last_raw_response = response
            return response
        except Exception as e:
            logger.warning("Analysis failed: %s", e)
            return {}

    def _map_to_primary_area(self, analysis: dict) -> str:
        """根据 domain 映射到 primary_area"""
        domain = analysis.get("domain", "")
        if not domain:
            return "general machine learning (i.e., none of the above)"

        # 直接使用 DOMAIN_TO_AREA_MAP
        if domain in DOMAIN_TO_AREA_MAP:
            return DOMAIN_TO_AREA_MAP[domain]

        # 如果不在列表中，尝试关键词匹配
        domain_lower = domain.lower()
        for key, area in DOMAIN_TO_AREA_MAP.items():
            if key in domain_lower:
                return area

        return "general machine learning (i.e., none of the above)"

    def parse_with_known_area(self, paper: Paper, known_area: str) -> PaperSignature:
        """已知 primary_area 时使用"""
        domain = AREA_TO_DOMAIN_MAP.get(known_area, "other")
        analysis = self._analyze_paper(paper)

        return PaperSignature(
            paper_type=analysis.get("paper_type"),
            tasks=analysis.get("tasks", []),
            domain=domain,
            method_family=analysis.get("method_family", []),
            main_claims=analysis.get("main_claims", []),
        )