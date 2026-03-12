from __future__ import annotations

from datetime import timedelta

from arm_memory.domain.models import MemoryKind, MemoryTrace
from arm_memory.retrieval import HybridRetrievalEngine
from arm_memory.utils import utcnow

from .conftest import StubVectorizer, make_config


class FakeSQLiteStore:
    def __init__(self, traces: list[MemoryTrace]):
        self._traces = traces

    def list_active_memory_traces(self, project_id, user_id, kinds, limit):
        del project_id, user_id, kinds, limit
        return list(self._traces)

    def list_related_nodes(self, project_id, user_id, query_tokens, limit):
        del project_id, user_id, limit
        if "coffee" in query_tokens:
            return {"coffee": 0.9}
        return {}


def test_score_trace_includes_weighted_signals(tmp_path):
    config = make_config(tmp_path)
    vectorizer = StubVectorizer()
    trace = MemoryTrace(
        project_id="proj",
        user_id="user",
        kind=MemoryKind.SEMANTIC,
        summary="user likes coffee and calm mornings",
        raw_text="coffee calm support",
        entities=["coffee"],
        tags=["preference"],
        salience=0.8,
        vector=vectorizer.embed("coffee calm"),
        access_count=2,
    )
    engine = HybridRetrievalEngine(
        config=config,
        sqlite_store=FakeSQLiteStore([trace]),
        vectorizer=vectorizer,
    )

    breakdown = engine._score_trace(
        trace,
        query_tokens=vectorizer.tokenize("coffee support"),
        query_vector=vectorizer.embed("coffee support"),
        related_nodes={"coffee": 0.9},
        qdrant_scores={trace.trace_id: 0.4},
    )

    assert breakdown["semantic_similarity"] > 0
    assert breakdown["graph_proximity"] == 0.9
    assert breakdown["remote_boost"] == 0.4
    assert breakdown["attentional_weight"] == 0.8
    assert breakdown["total"] > 0


def test_temporal_decay_uses_kind_specific_floor(tmp_path):
    config = make_config(tmp_path)
    vectorizer = StubVectorizer()
    old = utcnow() - timedelta(days=365)
    engine = HybridRetrievalEngine(
        config=config,
        sqlite_store=FakeSQLiteStore([]),
        vectorizer=vectorizer,
    )
    episodic = MemoryTrace(
        project_id="proj",
        user_id="user",
        kind=MemoryKind.EPISODIC,
        summary="old episodic memory",
        last_accessed_at=old,
    )
    semantic = MemoryTrace(
        project_id="proj",
        user_id="user",
        kind=MemoryKind.SEMANTIC,
        summary="old semantic memory",
        last_accessed_at=old,
    )

    episodic_decay = engine._temporal_decay(episodic, utcnow())
    semantic_decay = engine._temporal_decay(semantic, utcnow())

    assert episodic_decay == 0.08
    assert semantic_decay == 0.45
