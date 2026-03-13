from __future__ import annotations

import math
import time
from datetime import datetime

from loguru import logger

from arm_memory.config import ARMConfig
from arm_memory.domain.models import EmotionVector, MemoryKind, MemoryTrace, RetrievalHit
from arm_memory.stores.neo4j_store import Neo4jGraphStore
from arm_memory.stores.qdrant_store import QdrantVectorStore
from arm_memory.stores.sqlite_store import SQLiteMemoryStore
from arm_memory.utils import clamp, cosine_similarity, normalize_text, utcnow
from arm_memory.vectorizer import EmbeddingProvider


class HybridRetrievalEngine:
    def __init__(
        self,
        *,
        config: ARMConfig,
        sqlite_store: SQLiteMemoryStore,
        vectorizer: EmbeddingProvider,
        qdrant_store: QdrantVectorStore | None = None,
        neo4j_store: Neo4jGraphStore | None = None,
    ):
        self.config = config
        self.sqlite_store = sqlite_store
        self.vectorizer = vectorizer
        self.qdrant_store = qdrant_store
        self.neo4j_store = neo4j_store

    _RETRIEVAL_MODE_WEIGHTS: dict[str, dict[str, float]] = {
        "fact": {
            "retrieval_weight_semantic": 0.30,
            "retrieval_weight_lexical": 0.25,
            "retrieval_weight_graph": 0.30,
            "retrieval_weight_salience": 0.10,
            "retrieval_weight_remote": 0.05,
            "retrieval_weight_emotion": 0.00,
        },
        "emotional": {
            "retrieval_weight_semantic": 0.30,
            "retrieval_weight_lexical": 0.10,
            "retrieval_weight_graph": 0.10,
            "retrieval_weight_salience": 0.20,
            "retrieval_weight_remote": 0.05,
            "retrieval_weight_emotion": 0.25,
        },
    }

    def retrieve(
        self,
        *,
        project_id: str,
        user_id: str,
        query: str,
        top_k: int | None = None,
        query_emotion_hint: EmotionVector | None = None,
        retrieval_mode: str = "default",
    ) -> list[RetrievalHit]:
        t0 = time.monotonic()
        limit = top_k or self.config.retrieval_top_k
        candidate_limit = self.config.retrieval_candidate_limit

        query_vector = self.vectorizer.embed(query)
        query_tokens = self.vectorizer.tokenize(query)

        # 1. Query graph_store (and sqlite_store) to get related entities/nodes
        # Note: Currently these return dict[str, float] mapping *entity* to score.
        # If your graph store is modified to return trace_ids, it will be included in graph_boosts.
        related_nodes = self.sqlite_store.list_related_nodes(
            project_id,
            user_id,
            query_tokens,
            limit=32,
        )
        graph_boosts: dict[str, float] = {}
        if self.neo4j_store and self.neo4j_store.enabled:
            graph_boosts = self.neo4j_store.search_related(
                project_id=project_id,
                user_id=user_id,
                entities=query_tokens,
                limit=16,
            )
            if self.config.neo4j_max_hops >= 2:
                multihop = self.neo4j_store.search_related_multihop(
                    project_id=project_id,
                    user_id=user_id,
                    entities=query_tokens,
                    max_hops=self.config.neo4j_max_hops,
                    limit=20,
                )
                for key, value in multihop.items():
                    graph_boosts[key] = max(graph_boosts.get(key, 0.0), value)
            for key, value in graph_boosts.items():
                related_nodes[key] = max(related_nodes.get(key, 0.0), value)

        # 2. Query vector_store to get vector_boosts (dict of trace_id -> score)
        vector_boosts: dict[str, float] = {}
        if self.qdrant_store and self.qdrant_store.enabled:
            vector_boosts = self.qdrant_store.search(
                project_id=project_id,
                user_id=user_id,
                query_vector=query_vector,
                limit=max(limit * self.config.retrieval_qdrant_prefetch_factor, 8),
            )

        # 3. Combine IDs (from graph and vector) 
        # (Assuming graph_boosts might contain trace_ids as per requirements, 
        # though currently it returns entity strings)
        combined_ids = set(vector_boosts.keys()) | set(graph_boosts.keys())
        
        # 4. Filter and fetch MemoryTrace objects
        traces: list[MemoryTrace] = []
        if combined_ids:
            # Pass these IDs to get_traces_by_ids (top N combined limit if needed)
            id_list = list(combined_ids)[:candidate_limit]
            traces = self.sqlite_store.get_traces_by_ids(id_list)
        
        # Fallback if no vector/graph stores provided or no hits: generic recent/salient fetch
        if not traces:
            traces = self.sqlite_store.list_active_memory_traces(
                project_id,
                user_id,
                kinds=[MemoryKind.EPISODIC, MemoryKind.SEMANTIC],
                limit=candidate_limit,
            )

        if not traces:
            return []

        # 5. Score and filter the traces
        weight_overrides = self._RETRIEVAL_MODE_WEIGHTS.get(retrieval_mode)
        scored: list[RetrievalHit] = []
        qdrant_candidate_ids = set(vector_boosts.keys()) if vector_boosts else set()
        
        for trace in traces:
            # Fast-skip traces not in vector results if vector hits exist and no entity overlap
            if qdrant_candidate_ids and trace.trace_id not in qdrant_candidate_ids:
                trace_entity_set = {normalize_text(e) for e in trace.entities}
                query_token_set = {normalize_text(t) for t in query_tokens}
                if not (trace_entity_set & query_token_set):
                    has_graph_hit = any(
                        normalize_text(e) in related_nodes for e in trace.entities
                    )
                    if not has_graph_hit:
                        continue

            breakdown = self._score_trace(trace, query_tokens, query_vector, related_nodes, vector_boosts, query_emotion_hint, weight_overrides)
            total = breakdown.pop("total")
            if total <= 0:
                continue
            scored.append(RetrievalHit(trace=trace, score=total, score_breakdown=breakdown))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        results = scored[:limit]
        elapsed_ms = (time.monotonic() - t0) * 1000
        if results:
            for hit in results[:3]:
                logger.debug(
                    "retrieval_hit trace_id={} score={:.4f} breakdown={}",
                    hit.trace.trace_id,
                    hit.score,
                    hit.score_breakdown,
                )
        logger.info(
            "retrieval_complete candidates={} scored={} returned={} elapsed_ms={:.1f} top_score={:.4f}",
            len(traces),
            len(scored),
            len(results),
            elapsed_ms,
            results[0].score if results else 0.0,
        )
        return results

    def _score_trace(
        self,
        trace: MemoryTrace,
        query_tokens: list[str],
        query_vector: list[float],
        related_nodes: dict[str, float],
        qdrant_scores: dict[str, float],
        query_emotion_hint: EmotionVector | None = None,
        weight_overrides: dict[str, float] | None = None,
    ) -> dict[str, float]:
        trace_tokens = self.vectorizer.tokenize(" ".join([trace.summary, trace.raw_text, *trace.entities, *trace.tags]))
        lexical = self._token_overlap(query_tokens, trace_tokens)
        vector = cosine_similarity(query_vector, trace.vector) if trace.vector else 0.0
        semantic_similarity = max(lexical, vector)

        entity_hits = [normalize_text(entity) for entity in trace.entities]
        graph_proximity = 0.0
        for entity in entity_hits:
            graph_proximity = max(graph_proximity, related_nodes.get(entity, 0.0))
        if not graph_proximity:
            overlap = set(entity_hits) & {normalize_text(token) for token in query_tokens}
            graph_proximity = 1.0 if overlap else 0.0
        graph_proximity = clamp(graph_proximity, 0.0, 1.0)

        attentional_weight = clamp(trace.salience, 0.0, 1.0)
        temporal_decay = self._temporal_decay(trace, utcnow())
        access_boost = 1.0 + math.log1p(max(trace.access_count, 0)) * max(
            self.config.reconsolidation_boost - 1.0,
            0.01,
        )
        remote_boost = qdrant_scores.get(trace.trace_id, 0.0)

        emotion_similarity = self._emotion_similarity(trace.emotion, query_emotion_hint)

        def _w(key: str) -> float:
            if weight_overrides and key in weight_overrides:
                return weight_overrides[key]
            return getattr(self.config, key)

        total = (
            semantic_similarity * _w("retrieval_weight_semantic")
            + lexical * _w("retrieval_weight_lexical")
            + graph_proximity * _w("retrieval_weight_graph")
            + attentional_weight * _w("retrieval_weight_salience")
            + remote_boost * _w("retrieval_weight_remote")
            + emotion_similarity * _w("retrieval_weight_emotion")
        ) * temporal_decay * access_boost
        total = clamp(total, 0.0, 10.0)
        return {
            "semantic_similarity": semantic_similarity,
            "lexical_overlap": lexical,
            "graph_proximity": graph_proximity,
            "attentional_weight": attentional_weight,
            "temporal_decay": temporal_decay,
            "access_boost": access_boost,
            "remote_boost": remote_boost,
            "emotion_similarity": emotion_similarity,
            "total": total,
        }

    def _temporal_decay(self, trace: MemoryTrace, now: datetime) -> float:
        age_seconds = max((now - trace.last_accessed_at).total_seconds(), 0.0)
        age_hours = age_seconds / 3600.0
        base_half_life = (
            self.config.semantic_half_life_hours
            if trace.kind == MemoryKind.SEMANTIC
            else self.config.episodic_half_life_hours
        )
        if base_half_life <= 0:
            return 1.0
        half_life = base_half_life * (
            1.0 + self.config.half_life_access_boost_factor * math.log1p(max(0, trace.access_count))
        )
        decay = math.exp(-(math.log(2) / half_life) * age_hours)
        floor = 0.45 if trace.kind == MemoryKind.SEMANTIC else 0.08
        return max(decay, floor)

    @staticmethod
    def _emotion_similarity(trace_emotion: EmotionVector, hint: EmotionVector | None) -> float:
        if hint is None:
            return 0.5
        val_diff = abs(trace_emotion.valence - hint.valence)
        aro_diff = abs(trace_emotion.arousal - hint.arousal)
        return clamp(1.0 - 0.5 * (val_diff + aro_diff), 0.0, 1.0)

    @staticmethod
    def _token_overlap(query_tokens: list[str], trace_tokens: list[str]) -> float:
        if not query_tokens or not trace_tokens:
            return 0.0
        left = set(query_tokens)
        right = set(trace_tokens)
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
