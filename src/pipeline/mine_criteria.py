"""Criteria Miner: 两阶段提取criteria

阶段1：根据domain动态生成themes（数量可控5-8个）
阶段2：基于这些themes提取criteria
"""
from __future__ import annotations

import json
import logging
import uuid

from clients.embedding_client import EmbeddingClient
from clients.llm_client import LLMClient
from common.types import Criterion, Paper, Review, VenuePolicy
from storage.milvus_store import MilvusConfig, MilvusStore

logger = logging.getLogger(__name__)


class CriteriaMiner:
    """两阶段Criteria提取器"""

    def __init__(
        self,
        llm: LLMClient,
        embedding_client: EmbeddingClient | None = None,
        vector_store: dict | None = None,
    ) -> None:
        self.llm = llm
        self.embedding_client = embedding_client
        self.vector_store = vector_store or {}
        # 缓存已生成的themes（同domain可复用）
        self._theme_cache: dict[str, dict] = {}

    def mine_content_criteria(
        self,
        domain: str | None,
        related_papers: list[Paper],
        related_reviews: list[Review],
    ) -> list[Criterion]:
        """两阶段提取：先定themes，再提取criteria"""

        # 阶段1：动态生成themes
        themes = self._generate_themes(domain, related_reviews)

        # 阶段2：基于themes提取criteria
        criteria = self._extract_criteria_with_themes(
            domain, related_papers, related_reviews, themes
        )

        logger.info("Mined %d content criteria for domain: %s (themes: %s)",
                    len(criteria), domain or "general", list(themes.keys()))

        return criteria

    def mine_policy_criteria(
        self,
        venue_policy: VenuePolicy | None,
        random_reviews: list[Review],
    ) -> list[Criterion]:
        """提取review style guidelines"""
        prompt = self._policy_prompt(venue_policy, random_reviews)
        response = self.llm.generate_json(prompt)
        items = response.get("criteria", [])
        criteria = self._parse_criteria(items, kind="policy")
        criteria = self._filter_policy(criteria)
        logger.info("Mined %d policy criteria", len(criteria))
        return criteria

    # === 阶段1：动态生成themes ===

    def _generate_themes(
        self,
        domain: str | None,
        reviews: list[Review],
    ) -> dict[str, str]:
        """根据domain和reviews动态生成themes

        Returns:
            {"theme_name": "description", ...}
        """

        # 检查缓存
        cache_key = domain or "general"
        if cache_key in self._theme_cache:
            logger.info("Using cached themes for %s", cache_key)
            return self._theme_cache[cache_key]

        # 构建prompt
        review_snippets = [r.text[:300] for r in reviews[:10]]

        prompt = "\n".join([
            "You are defining review evaluation themes for a specific domain.",
            "",
            f"## Domain: {domain or 'General Machine Learning'}",
            "",
            "## Sample Reviews (for context)",
            *review_snippets[:8],
            "",
            "## Your Task",
            "Based on the domain and review patterns, define 5-8 KEY EVALUATION THEMES.",
            "",
            "## Guidelines for Good Themes",
            "- Each theme should be a BROAD aspect (not too specific)",
            "- Examples of good themes: 'Methodology', 'Experiments', 'Theoretical Analysis'",
            "- Examples of bad themes (too specific): 'Layer Normalization', 'GAN Training'",
            "- Themes should cover most important evaluation aspects for this domain",
            "- Some universal themes may apply: 'Clarity', 'Significance', 'Originality'",
            "",
            "## Output Format (JSON)",
            "{",
            "  \"themes\": {",
            "    \"ThemeName\": \"Brief description of what to evaluate\",",
            "    ...",
            "  }",
            "}",
            "",
            "CONSTRAINT: Generate exactly 5-8 themes.",
        ])

        try:
            response = self.llm.generate_json(prompt)
            themes = response.get("themes", {})

            # 验证数量
            if len(themes) < 5:
                logger.warning("Generated only %d themes, adding defaults", len(themes))
                themes = self._add_default_themes(themes)
            elif len(themes) > 8:
                # 保留最重要的8个
                themes = dict(list(themes.items())[:8])

            # 缓存
            self._theme_cache[cache_key] = themes

            return themes

        except Exception as e:
            logger.warning("Theme generation failed: %s, using defaults", e)
            return self._default_themes()

    def _default_themes(self) -> dict[str, str]:
        """默认themes（fallback）"""
        return {
            "Quality": "Technical soundness and correctness",
            "Clarity": "Writing quality and presentation",
            "Originality": "Novelty of ideas or methods",
            "Significance": "Impact and importance of contributions",
            "Experiments": "Empirical evaluation quality",
            "Reproducibility": "Details to reproduce results",
        }

    def _add_default_themes(self, existing: dict) -> dict:
        """补充缺失的themes"""
        defaults = self._default_themes()
        for name, desc in defaults.items():
            if len(existing) >= 8:
                break
            if name not in existing:
                existing[name] = desc
        return existing

    # === 阶段2：基于themes提取criteria ===

    def _extract_criteria_with_themes(
        self,
        domain: str | None,
        papers: list[Paper],
        reviews: list[Review],
        themes: dict[str, str],
    ) -> list[Criterion]:

        # 准备context
        paper_snippets = [f"- {p.title}: {p.abstract[:400]}" for p in papers[:3]]
        review_snippets = [f"- {r.text[:400]}" for r in reviews[:6]]

        # 格式化themes
        themes_ref = "\n".join([
            f"- **{name}**: {desc}"
            for name, desc in themes.items()
        ])

        # 获取coverage gap context（如果可用）
        gap_context = self._coverage_gap_context(domain)

        prompt = "\n".join([
            "You are an expert reviewer extracting EVALUATION CRITERIA.",
            "",
            "## Domain",
            domain or "General Machine Learning",
            "",
            "## Predefined Themes (USE THESE, don't create new ones)",
            themes_ref,
            "",
            "## Your Task",
            "For each theme above, extract 1-2 SPECIFIC, VERIFIABLE criteria.",
            "Total criteria: aim for 6-10 across all themes.",
            "",
            "## Good Criteria Examples",
            "- \"Does the paper provide formal theorem statements with proofs?\"",
            "- \"Are baselines compared with identical evaluation protocols?\"",
            "- \"Is there an ablation study for each proposed component?\"",
            "",
            "## Bad Criteria Examples (AVOID)",
            "- \"The paper proposes X method\" (summary, not criterion)",
            "- \"Is the paper well written?\" (too vague)",
            "- \"Does it use dataset Y?\" (too specific to one paper)",
            "",
            "## Related Papers (for domain context)",
            *paper_snippets,
            "",
            "## Related Reviews (for patterns)",
            *review_snippets,
            "",
            gap_context,
            "",
            "## Output Format (JSON)",
            "{",
            "  \"criteria\": [",
            "    {",
            "      \"text\": \"The criterion as a question/checklist item\",",
            "      \"theme\": \"Must be one of the predefined themes above\",",
            "      \"source_ids\": [\"review_1\", \"review_3\"]",
            "    },",
            "    ...",
            "  ]",
            "}",
        ])

        try:
            response = self.llm.generate_json(prompt)
            items = response.get("criteria", [])
            criteria = self._parse_criteria(items, kind="content")

            # 验证theme是否在预定义范围内
            valid_themes = set(themes.keys())
            for c in criteria:
                if c.theme not in valid_themes:
                    # 尝试匹配相近的theme
                    c.theme = self._match_theme(c.theme, valid_themes)

            return criteria

        except Exception as e:
            logger.warning("Criteria extraction failed: %s", e)
            return []

    def _match_theme(self, extracted_theme: str, valid_themes: set) -> str:
        """匹配到最接近的valid theme"""
        extracted_lower = extracted_theme.lower()

        # 简单匹配
        for valid in valid_themes:
            if valid.lower() in extracted_lower or extracted_lower in valid.lower():
                return valid

        # 默认返回第一个
        return list(valid_themes)[0]

    def _coverage_gap_context(self, domain: str | None) -> str:
        """从历史审稿中检索该领域的常见遗漏点"""
        if self.vector_store.get("backend") != "milvus" or self.embedding_client is None:
            return ""
        collection = self.vector_store.get("coverage_gaps_collection")
        if not collection:
            return ""

        try:
            milvus_cfg = MilvusConfig(
                host=self.vector_store.get("host", "localhost"),
                port=int(self.vector_store.get("port", 19530)),
                papers_collection=self.vector_store.get("papers_collection", "papers_iclr"),
                reviews_collection=self.vector_store.get("reviews_collection", "reviews_iclr"),
            )
            milvus = MilvusStore(milvus_cfg)
            query = domain or "machine learning"
            query_vec = self.embedding_client.embed([query])[0].tolist()
            gap_texts = milvus.search_texts(collection, query_vec, top_k=5)

            if gap_texts:
                return "Common missed points in this domain: " + " | ".join(gap_texts)
        except Exception as e:
            logger.warning("Coverage gap search failed: %s", e)

        return ""

    # === Policy criteria (unchanged) ===

    @staticmethod
    def _policy_prompt(venue_policy: VenuePolicy | None, random_reviews: list[Review]) -> str:
        policy_text = json.dumps(venue_policy.model_dump() if venue_policy else {}, ensure_ascii=True)
        review_snippets = [f"- {review.text[:300]}" for review in random_reviews[:6]]

        return "\n".join([
            "You are extracting REVIEW STYLE GUIDELINES.",
            "",
            "## Your Task",
            "Extract 4-6 style/format guidelines about HOW to write reviews.",
            "",
            "## Good Policy Examples",
            "- \"Each weakness should cite specific evidence\"",
            "- \"Include specific suggestions for improvement\"",
            "",
            "## Bad Policy Examples",
            "- \"Mention the dataset\" (content, not style)",
            "- \"Check if novel\" (content criterion)",
            "",
            "## Venue Policy",
            policy_text,
            "",
            "## Sample Reviews",
            *review_snippets,
            "",
            "## Output Format (JSON)",
            "{",
            "  \"criteria\": [",
            "    {\"text\": \"...\", \"theme\": \"Style\", \"source_ids\": [...]},",
            "    ...",
            "  ]",
            "}",
        ])

    def _filter_policy(self, criteria: list[Criterion]) -> list[Criterion]:
        if not criteria:
            return []
        prompt = "\n".join([
            "Filter criteria. Keep only review STYLE guidelines.",
            "Reject content-specific criteria about methods, datasets, or results.",
            f"Criteria: {[c.model_dump() for c in criteria]}",
            "",
            "Output JSON: {\"keep_ids\": [list of criterion_id to keep]}",
        ])
        try:
            response = self.llm.generate_json(prompt)
            keep_ids = response.get("keep_ids", [])
            if isinstance(keep_ids, list) and keep_ids:
                keep_set = {str(item) for item in keep_ids}
                return [c for c in criteria if c.criterion_id in keep_set]
        except Exception as e:
            logger.warning("Policy filter failed: %s", e)
        return criteria

    def _parse_criteria(self, items: list[dict], kind: str) -> list[Criterion]:
        criteria: list[Criterion] = []
        for item in items:
            try:
                source_ids = item.get("source_ids", [])
                if isinstance(source_ids, list):
                    source_ids = [str(v) for v in source_ids]
                elif source_ids is None:
                    source_ids = []
                else:
                    source_ids = [str(source_ids)]

                criteria.append(Criterion(
                    criterion_id=item.get("id") or str(uuid.uuid4()),
                    text=item["text"],
                    theme=item.get("theme", "general"),
                    kind=kind,
                    source_ids=source_ids,
                ))
            except Exception as e:
                logger.warning("Skipping invalid criterion: %s", e)
        return criteria