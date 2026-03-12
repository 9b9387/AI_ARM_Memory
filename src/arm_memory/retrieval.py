from __future__ import annotations

import logging
import math
import time
from datetime import datetime

from arm_memory.config import ARMConfig
from arm_memory.domain.models import MemoryKind, MemoryTrace, RetrievalHit
from arm_memory.stores.neo4j_store import Neo4jGraphStore
from arm_memory.stores.qdrant_store import QdrantVectorStore
from arm_memory.stores.sqlite_store import SQLiteMemoryStore
from arm_memory.utils import clamp, cosine_similarity, normalize_text, utcnow
from arm_memory.vectorizer import EmbeddingProvider

logger = logging.getLogger(__name__)


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

    def retrieve(
        self,
        *,
        project_id: str,
        user_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        t0 = time.monotonic()
        limit = top_k or self.config.retrieval_top_k
        candidate_limit = self.config.retrieval_candidate_limit
        traces = self.sqlite_store.list_active_memory_traces(
            project_id,
            user_id,
            kinds=[MemoryKind.EPISODIC, MemoryKind.SEMANTIC],
            limit=candidate_limit,
        )
        if not traces:
            return []

        query_vector = self.vectorizer.embed(query)
        query_tokens = self.vectorizer.tokenize(query)
        related_nodes = self.sqlite_store.list_related_nodes(
            project_id,
            user_id,
            query_tokens,
            limit=32,
        )
        if self.neo4j_store and self.neo4j_store.enabled:
            for key, value in self.neo4j_store.search_related(
                project_id=project_id,
                user_id=user_id,
                entities=query_tokens,
                limit=16,
            ).items():
                related_nodes[key] = max(related_nodes.get(key, 0.0), value)

        qdrant_scores: dict[str, float] = {}
        if self.qdrant_store and self.qdrant_store.enabled:
            qdrant_scores = self.qdrant_store.search(
                project_id=project_id,
                user_id=user_id,
                query_vector=query_vector,
                limit=max(limit * self.config.retrieval_qdrant_prefetch_factor, 8),
            )

        scored: list[RetrievalHit] = []
        qdrant_candidate_ids = set(qdrant_scores.keys()) if qdrant_scores else set()
        for trace in traces:
            # Pre-filter: if Qdrant returned candidates, fast-skip traces not in
            # Qdrant results and with no entity overlap with query tokens.
            if qdrant_candidate_ids and trace.trace_id not in qdrant_candidate_ids:
                trace_entity_set = {normalize_text(e) for e in trace.entities}
                query_token_set = {normalize_text(t) for t in query_tokens}
                if not (trace_entity_set & query_token_set):
                    # Check related_nodes overlap before skipping
                    has_graph_hit = any(
                        normalize_text(e) in related_nodes for e in trace.entities
                    )
                    if not has_graph_hit:
                        continue

            breakdown = self._score_trace(trace, query_tokens, query_vector, related_nodes, qdrant_scores)
            total = breakdown.pop("total")
            if total <= 0:
                continue
            scored.append(RetrievalHit(trace=trace, score=total, score_breakdown=breakdown))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        results = scored[:limit]
        elapsed_ms = (time.monotonic() - t0) * 1000
        if results and logger.isEnabledFor(logging.DEBUG):
            for hit in results[:3]:
                logger.debug(
                    "retrieval_hit trace_id=%s score=%.4f breakdown=%s",
                    hit.trace.trace_id,
                    hit.score,
                    hit.score_breakdown,
                )
        logger.info(
            "retrieval_complete candidates=%d scored=%d returned=%d elapsed_ms=%.1f top_score=%.4f",
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

        total = (
            semantic_similarity * self.config.retrieval_weight_semantic
            + lexical * self.config.retrieval_weight_lexical
            + graph_proximity * self.config.retrieval_weight_graph
            + attentional_weight * self.config.retrieval_weight_salience
            + remote_boost * self.config.retrieval_weight_remote
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
            "total": total,
        }

    def _temporal_decay(self, trace: MemoryTrace, now: datetime) -> float:
        age_seconds = max((now - trace.last_accessed_at).total_seconds(), 0.0)
        age_hours = age_seconds / 3600.0
        half_life = (
            self.config.semantic_half_life_hours
            if trace.kind == MemoryKind.SEMANTIC
            else self.config.episodic_half_life_hours
        )
        if half_life <= 0:
            return 1.0
        decay = math.exp(-(math.log(2) / half_life) * age_hours)
        floor = 0.45 if trace.kind == MemoryKind.SEMANTIC else 0.08
        return max(decay, floor)

    @staticmethod
    def _token_overlap(query_tokens: list[str], trace_tokens: list[str]) -> float:
        if not query_tokens or not trace_tokens:
            return 0.0
        left = set(query_tokens)
        right = set(trace_tokens)
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
