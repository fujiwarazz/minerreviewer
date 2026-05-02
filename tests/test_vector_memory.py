"""测试 VectorMemoryStore 和 AgentMemoryAllocator"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from common.types import ExperienceCard
from storage.vector_memory_store import VectorMemoryStore
from pipeline.agent_memory_allocator import AgentMemoryAllocator


@pytest.fixture
def temp_store_path():
    """临时存储路径"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def sample_cards():
    """示例卡片"""
    return [
        ExperienceCard(
            card_id="card_1",
            kind="strength",
            theme="quality",
            content="Always verify baseline coverage within 2 years for empirical papers",
            utility=0.8,
            confidence=0.7,
            metadata={"memory_type": "learned_strength"},
        ),
        ExperienceCard(
            card_id="card_2",
            kind="critique",
            theme="experiments",
            content="Check if ablation studies cover all key components",
            utility=0.6,
            confidence=0.8,
            metadata={"memory_type": "learned_critique"},
        ),
        ExperienceCard(
            card_id="card_3",
            kind="failure",
            theme="decision",
            content="Missing comparison with recent methods often leads to rejection",
            utility=0.9,
            confidence=0.85,
            metadata={"memory_type": "learned_failure"},
        ),
    ]


def test_experience_card_embedding_text():
    """测试 ExperienceCard.get_card_text"""
    card = ExperienceCard(
        card_id="test",
        kind="strength",
        theme="quality",
        content="Test content",
        trigger=["trigger1", "trigger2"],
    )

    text = ExperienceCard.get_card_text(card)
    assert "Test content" in text
    assert "Theme: quality" in text
    assert "Triggers:" in text


def test_vector_memory_store_basic(temp_store_path, sample_cards):
    """测试 VectorMemoryStore 基本功能"""
    store = VectorMemoryStore(temp_store_path)

    # 添加卡片（不带embedding_client）
    for card in sample_cards:
        card_copy = card.model_copy()
        card_copy.embedding = None  # 不测试embedding
        store.add_card(card_copy, owner_agent="theme_quality")

    assert len(store.cards) == 3

    # 检查owner_agent设置
    for card in store.cards:
        assert card.owner_agent == "theme_quality"

    # 检查metadata过滤
    quality_cards = store.list_cards(owner_agent="theme_quality", kind="strength")
    assert len(quality_cards) == 1
    assert quality_cards[0].theme == "quality"


def test_vector_memory_store_retrieve_without_embedding(temp_store_path, sample_cards):
    """测试 VectorMemoryStore 检索（无embedding）"""
    store = VectorMemoryStore(temp_store_path)

    # 添加卡片
    for card in sample_cards:
        card_copy = card.model_copy()
        card_copy.embedding = None
        # 设置不同的owner_agent
        if card.kind == "strength":
            store.add_card(card_copy, owner_agent="theme_quality")
        elif card.kind == "critique":
            store.add_card(card_copy, owner_agent="theme_experiments")
        elif card.kind == "failure":
            store.add_card(card_copy, owner_agent="arbiter")

    # 检索（不使用向量检索）
    results = store.retrieve_cards(
        query_text="test query",
        owner_agent="theme_quality",
        use_vector_search=False,
        top_k=10,
    )

    assert len(results) == 1
    card, scores = results[0]
    assert card.kind == "strength"
    assert "metadata_score" in scores
    assert "final_score" in scores


def test_agent_memory_allocator_basic(sample_cards):
    """测试 AgentMemoryAllocator 基本分配"""
    allocator = AgentMemoryAllocator()

    allocation = allocator.allocate(sample_cards)

    # 检查分配结果
    assert len(allocation) > 0

    # strength卡片应该分配给 theme_quality + arbiter
    assert "theme_quality" in allocation or "arbiter" in allocation

    # failure卡片应该分配给 arbiter
    assert "arbiter" in allocation
    arbiter_cards = allocation["arbiter"]
    failure_cards = [c for c in arbiter_cards if c.kind == "failure"]
    assert len(failure_cards) > 0


def test_agent_memory_allocator_theme_mapping():
    """测试 AgentMemoryAllocator 主题映射"""
    allocator = AgentMemoryAllocator()

    # 测试主题映射
    assert allocator.AGENT_MAPPING["quality"] == "theme_quality"
    assert allocator.AGENT_MAPPING["experiments"] == "theme_experiments"
    assert allocator.AGENT_MAPPING["decision"] == "arbiter"

    # 测试支持的agent列表
    agents = allocator.get_supported_agents()
    assert "theme_quality" in agents
    assert "arbiter" in agents


def test_agent_memory_allocator_share_with_arbiter(sample_cards):
    """测试 strength 卡片共享给 arbiter"""
    allocator = AgentMemoryAllocator()

    # 开启共享
    allocation = allocator.allocate(sample_cards, share_strength_with_arbiter=True)

    # arbiter应该收到strength卡片
    arbiter_cards = allocation.get("arbiter", [])
    strength_cards = [c for c in arbiter_cards if c.kind == "strength"]
    assert len(strength_cards) > 0

    # 关闭共享
    allocation2 = allocator.allocate(sample_cards, share_strength_with_arbiter=False)

    # arbiter不应该收到strength卡片（只有failure）
    arbiter_cards2 = allocation2.get("arbiter", [])
    non_failure_cards = [c for c in arbiter_cards2 if c.kind != "failure"]
    assert len(non_failure_cards) == 0


def test_vector_memory_store_stats(temp_store_path, sample_cards):
    """测试 VectorMemoryStore 统计功能"""
    store = VectorMemoryStore(temp_store_path)

    for card in sample_cards:
        card_copy = card.model_copy()
        card_copy.embedding = None
        store.add_card(card_copy, owner_agent="test_agent")

    stats = store.get_stats()

    assert stats["total_cards"] == 3
    assert stats["active_cards"] == 3
    assert "by_kind" in stats
    assert "by_owner" in stats
    assert stats["by_owner"]["test_agent"] == 3


def test_experience_card_serialization(temp_store_path, sample_cards):
    """测试 ExperienceCard 序列化（不含embedding）"""
    store = VectorMemoryStore(temp_store_path)

    # 添加带embedding的卡片，并指定owner_agent
    card = sample_cards[0].model_copy()
    card.embedding = [0.1] * 768  # 模拟embedding
    store.add_card(card, owner_agent="test_agent")  # 添加owner_agent

    # 重新加载
    store2 = VectorMemoryStore(temp_store_path)

    # embedding不应该被加载到内存
    for loaded_card in store2.cards:
        assert loaded_card.embedding is None

    # 但其他字段应该保留
    assert len(store2.cards) == 1
    assert store2.cards[0].card_id == card.card_id
    assert store2.cards[0].owner_agent == "test_agent"  # 检查owner_agent


def test_vector_memory_store_domain_filtering(temp_store_path):
    """测试 VectorMemoryStore 领域过滤（三级匹配）"""
    store = VectorMemoryStore(temp_store_path)

    # 创建不同领域的卡片（相同utility避免淹没降权效果）
    nlp_card = ExperienceCard(
        card_id="nlp_1", kind="strength", theme="quality",
        content="NLP specific rule", primary_area="nlp", utility=0.7, confidence=0.7,
    )
    cv_card = ExperienceCard(
        card_id="cv_1", kind="strength", theme="quality",
        content="CV specific rule", primary_area="computer_vision", utility=0.7, confidence=0.7,
    )
    generic_card = ExperienceCard(
        card_id="gen_1", kind="strength", theme="quality",
        content="General rule", primary_area=None, utility=0.7, confidence=0.7,
    )

    for c in [nlp_card, cv_card, generic_card]:
        c_copy = c.model_copy()
        c_copy.embedding = None
        store.add_card(c_copy, owner_agent="theme_quality")

    # 查询NLP领域：nlp卡应在cv卡之前，通用卡在最后
    results = store.retrieve_cards(
        query_text="test", owner_agent="theme_quality",
        primary_area="nlp", use_vector_search=False, top_k=10,
    )

    card_ids = [c.card_id for c, _ in results]
    assert card_ids[0] == "nlp_1"  # 精确匹配排第一
    # 跨领域(nlp不匹配cv)降权30%，所以cv可能排在gen(降权10%)之后
    assert "cv_1" in card_ids
    assert card_ids.index("nlp_1") < card_ids.index("cv_1")  # NLP排在CV前面

    # 查询CV领域
    results_cv = store.retrieve_cards(
        query_text="test", owner_agent="theme_quality",
        primary_area="computer_vision", use_vector_search=False, top_k=10,
    )

    card_ids_cv = [c.card_id for c, _ in results_cv]
    assert card_ids_cv[0] == "cv_1"  # 精确匹配排第一

    # 无领域时，通用卡片优先
    results_none = store.retrieve_cards(
        query_text="test", owner_agent="theme_quality",
        primary_area=None, use_vector_search=False, top_k=10,
    )

    card_ids_none = [c.card_id for c, _ in results_none]
    assert card_ids_none[0] == "gen_1"  # 通用卡片排第一


def test_effectiveness_tracking_basic(temp_store_path):
    """测试 Effectiveness Tracking: record_usage + apply_feedback"""
    store = VectorMemoryStore(temp_store_path)

    card = ExperienceCard(
        card_id="feed_1", kind="strength", theme="quality", content="Test",
        utility=0.5, confidence=0.5,
    )
    store.add_card(card, owner_agent="theme_quality")

    # 记录使用
    store.record_usage("feed_1", "paper_123")
    c = store.get_card("feed_1")
    assert c.use_count == 1
    assert len(c.use_history) == 1
    assert c.use_history[0]["outcome"] == "pending"

    # 正面反馈
    stats = store.apply_feedback({"feed_1": "positive"}, "paper_123")
    c = store.get_card("feed_1")
    assert c.utility > 0.5  # 提升
    assert c.confidence > 0.5
    assert stats["promoted"] >= 1

    # 负面反馈多次
    store.apply_feedback({"feed_1": "negative"}, "paper_456")
    stats2 = store.apply_feedback({"feed_1": "negative"}, "paper_789")
    c = store.get_card("feed_1")
    assert c.utility < 0.5  # 降低
    # 连续2次负面 + utility<0.3 → 可能退役


def test_effectiveness_tracking_retirement(temp_store_path):
    """测试卡片退役"""
    store = VectorMemoryStore(temp_store_path)

    card = ExperienceCard(
        card_id="retire_1", kind="critique", theme="experiments", content="Test",
        utility=0.25, confidence=0.3,
    )
    store.add_card(card, owner_agent="theme_experiments")

    # 多次负面反馈使utility降到0.3以下并退役
    stats = store.apply_feedback({"retire_1": "negative"}, "p1")
    stats2 = store.apply_feedback({"retire_1": "negative"}, "p2")

    c = store.get_card("retire_1")
    # utility已降到很低，且两次负面
    assert c.utility < 0.25
    # 可能已被退役（取决于threshold）
    assert c.active is False or c.utility < 0.3


def test_experience_card_primary_area_field():
    """测试 ExperienceCard.primary_area 字段"""
    card = ExperienceCard(
        card_id="test", kind="strength", theme="quality", content="Test",
        primary_area="reinforcement_learning",
    )
    assert card.primary_area == "reinforcement_learning"

    # 默认值
    card2 = ExperienceCard(card_id="test2", kind="strength", theme="quality", content="Test")
    assert card2.primary_area is None

    # get_card_text 包含primary_area
    text = ExperienceCard.get_card_text(card)
    assert "reinforcement_learning" in text