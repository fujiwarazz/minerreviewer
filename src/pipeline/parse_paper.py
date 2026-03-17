"""PaperParser: 从论文中提取结构化特征 (PaperSignature)"""
from __future__ import annotations

import logging
from typing import Any

from clients.llm_client import LLMClient
from common.types import Paper, PaperSignature

logger = logging.getLogger(__name__)


class PaperParser:
    """从论文中提取结构化特征"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def parse(self, paper: Paper) -> PaperSignature:
        """解析论文，返回结构化特征"""
        prompt = self._build_prompt(paper)
        try:
            response = self.llm.generate_json(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.warning("Failed to parse paper %s: %s", paper.paper_id, e)
            return PaperSignature()

    def _build_prompt(self, paper: Paper) -> str:
        return "\n".join([
            "Extract structured features from the following paper.",
            "Return a JSON object with the following fields:",
            "- paper_type: one of [empirical, theoretical, survey, system, analysis, benchmark, other]",
            "- tasks: list of tasks (e.g., classification, generation, detection, segmentation)",
            "- domain: one of [vision, nlp, audio, multimodal, reinforcement_learning, graphs, other]",
            "- method_family: list of method families (e.g., transformer, gan, diffusion, gnn, cnn, rnn)",
            "- main_claims: list of main claims/contributions",
            "- claim_strength: one of [strong, moderate, weak]",
            "- datasets: list of datasets used",
            "- evaluation_style: list of evaluation methods (e.g., ablation, human_eval, benchmark, theoretical_analysis)",
            "- baseline_coverage: one of [comprehensive, partial, limited, none]",
            "- risk_profile: list of risks (e.g., reproducibility_risk, ethics_risk, evaluation_risk, scalability_risk)",
            "",
            f"Paper title: {paper.title}",
            f"Abstract: {paper.abstract}",
            "",
            "Return only valid JSON.",
        ])

    def _parse_response(self, response: dict[str, Any]) -> PaperSignature:
        return PaperSignature(
            paper_type=response.get("paper_type"),
            tasks=response.get("tasks", []),
            domain=response.get("domain"),
            method_family=response.get("method_family", []),
            main_claims=response.get("main_claims", []),
            claim_strength=response.get("claim_strength"),
            datasets=response.get("datasets", []),
            evaluation_style=response.get("evaluation_style", []),
            baseline_coverage=response.get("baseline_coverage"),
            risk_profile=response.get("risk_profile", []),
        )