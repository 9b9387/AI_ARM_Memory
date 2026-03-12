from __future__ import annotations

from arm_memory.consolidation import SleepCycleConsolidator
from arm_memory.domain.models import ConsolidationResult
from arm_memory.stores.sqlite_store import SQLiteMemoryStore

from .conftest import StubVectorizer, make_config


def test_apply_extraction_writes_episodic_and_semantic(tmp_path):
    """apply_extraction 接收外部抽取结果并写入情景/语义记忆。"""
    config = make_config(tmp_path)
    consolidator = SleepCycleConsolidator(
        config=config,
        sqlite_store=SQLiteMemoryStore(tmp_path / "arm_memory.db"),
        vectorizer=StubVectorizer(),
        qdrant_store=None,
        neo4j_store=None,
    )
    extraction = {
        "episodic": [{"summary": "一起聊了咖啡", "salience": 0.8}],
        "semantic": [
            {"subject": "user", "predicate": "prefers", "object": "coffee", "summary": "用户偏好咖啡"}
        ],
        "profile": {"preferences": ["coffee"]},
        "profile_items": [{"facet_type": "preferences", "value": "coffee", "confidence": 0.9}],
        "relationship": {"trust_delta": 2.0, "behavior_state": {"boundary_stage": "calm"}},
        "procedures": ["comfort_user"],
        "safety_flags": [],
    }

    result = consolidator.apply_extraction(
        project_id="proj",
        user_id="user",
        extraction=extraction,
        turn_ids=None,
        persona_prompt="",
    )

    assert isinstance(result, ConsolidationResult)
    assert result.episodic_count == 1
    assert result.semantic_count == 1
    assert result.triggered_procedures == ["comfort_user"]
    assert result.user_manual_updated is True


def test_apply_extraction_empty_extraction_normalizes(tmp_path):
    """空 extraction 经 _normalize_extraction 后仍可调用，写入为空。"""
    config = make_config(tmp_path)
    consolidator = SleepCycleConsolidator(
        config=config,
        sqlite_store=SQLiteMemoryStore(tmp_path / "arm_memory.db"),
        vectorizer=StubVectorizer(),
        qdrant_store=None,
        neo4j_store=None,
    )

    result = consolidator.apply_extraction(
        project_id="proj",
        user_id="user",
        extraction={},
        turn_ids=None,
        persona_prompt="",
    )

    assert result.episodic_count == 0
    assert result.semantic_count == 0
    assert result.user_manual_updated is True
