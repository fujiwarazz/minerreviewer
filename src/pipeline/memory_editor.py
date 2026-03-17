"""MemoryEditor: 决定经验进入 short-term 还是 long-term memory"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from common.types import ExperienceCard, PaperCase
from storage.case_store import CaseStore
from storage.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryEditor:
    """Memory 编辑器

    决定经验进入 short-term 还是 long-term memory
    实现 admission gate (utility/confidence threshold)
    支持 merge/deduplicate/downweight
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        case_store: CaseStore | None = None,
        short_term_utility_threshold: float = 0.3,
        long_term_utility_threshold: float = 0.6,
        confidence_threshold: float = 0.4,
        short_term_ttl_days: int = 30,
        dedup_threshold: float = 0.7,
    ) -> None:
        self.memory_store = memory_store
        self.case_store = case_store
        self.short_term_utility_threshold = short_term_utility_threshold
        self.long_term_utility_threshold = long_term_utility_threshold
        self.confidence_threshold = confidence_threshold
        self.short_term_ttl_days = short_term_ttl_days
        self.dedup_threshold = dedup_threshold

    def admit(self, card: ExperienceCard) -> str:
        """
        决定卡片是否进入 memory

        Returns:
            "admitted_short", "admitted_long", "rejected", "merged"
        """
        # Check confidence threshold
        if card.confidence < self.confidence_threshold:
            logger.debug("Card %s rejected due to low confidence", card.card_id[:8])
            return "rejected"

        # Check for duplicates
        existing = self.memory_store.find_similar(
            venue_id=card.venue_id,
            theme=card.theme,
            content=card.content,
            threshold=self.dedup_threshold,
        )
        if existing:
            # Merge with existing
            return self._merge_card(existing, card)

        # Determine if short-term or long-term
        if card.utility >= self.long_term_utility_threshold:
            result = self.memory_store.add_card(card)
            logger.info("Card %s admitted to long-term memory", card.card_id[:8])
            return "admitted_long"
        elif card.utility >= self.short_term_utility_threshold:
            # Mark as short-term
            card.metadata["memory_tier"] = "short_term"
            card.metadata["expires_at"] = (datetime.utcnow() + timedelta(days=self.short_term_ttl_days)).isoformat()
            result = self.memory_store.add_card(card)
            logger.info("Card %s admitted to short-term memory", card.card_id[:8])
            return "admitted_short"
        else:
            logger.debug("Card %s rejected due to low utility", card.card_id[:8])
            return "rejected"

    def _merge_card(self, existing: ExperienceCard, new: ExperienceCard) -> str:
        """合并新卡片到已存在的卡片"""
        # Update utility with weighted average
        total_use = existing.use_count + 1
        existing.utility = (existing.utility * existing.use_count + new.utility) / total_use
        existing.use_count = total_use
        existing.confidence = max(existing.confidence, new.confidence)

        # Merge source_ids
        existing.source_ids = list(set(existing.source_ids + new.source_ids))

        # Update metadata
        existing.metadata["last_merged"] = datetime.utcnow().isoformat()

        self.memory_store._save()
        logger.info("Merged card %s into existing %s", new.card_id[:8], existing.card_id[:8])
        return "merged"

    def admit_paper_case(self, case: PaperCase) -> bool:
        """将论文案例存入 CaseStore"""
        if not self.case_store:
            logger.warning("No case_store configured")
            return False

        try:
            self.case_store.add_case(case)
            logger.info("Added paper case %s", case.case_id[:8])
            return True
        except Exception as e:
            logger.error("Failed to add paper case: %s", e)
            return False

    def expire_short_term(self) -> int:
        """清理过期的 short-term memory"""
        expired_count = 0
        now = datetime.utcnow()

        for card in self.memory_store.cards:
            if card.metadata.get("memory_tier") == "short_term":
                expires_at_str = card.metadata.get("expires_at")
                if expires_at_str:
                    try:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        if now > expires_at:
                            card.active = False
                            expired_count += 1
                    except ValueError:
                        pass

        if expired_count > 0:
            self.memory_store._save()
            logger.info("Expired %d short-term cards", expired_count)

        return expired_count

    def downweight_unused(self, threshold_days: int = 30, decay_factor: float = 0.9) -> int:
        """降低长时间未使用的卡片的效用"""
        now = datetime.utcnow()
        threshold = timedelta(days=threshold_days)
        downweighted_count = 0

        for card in self.memory_store.cards:
            if not card.active:
                continue
            age = now - card.created_at
            if age > threshold and card.use_count == 0:
                card.utility *= decay_factor
                downweighted_count += 1

        if downweighted_count > 0:
            self.memory_store._save()
            logger.info("Downweighted %d unused cards", downweighted_count)

        return downweighted_count

    def promote_to_long_term(self, min_use_count: int = 3, min_utility: float = 0.5) -> int:
        """将表现良好的 short-term 卡片提升为 long-term"""
        promoted_count = 0

        for card in self.memory_store.cards:
            if not card.active:
                continue
            if card.metadata.get("memory_tier") == "short_term":
                if card.use_count >= min_use_count and card.utility >= min_utility:
                    card.metadata["memory_tier"] = "long_term"
                    if "expires_at" in card.metadata:
                        del card.metadata["expires_at"]
                    promoted_count += 1

        if promoted_count > 0:
            self.memory_store._save()
            logger.info("Promoted %d cards to long-term", promoted_count)

        return promoted_count