from __future__ import annotations

import base64
from typing import Any

import httpx
from loguru import logger

from arm_memory.domain.models import GraphEdge, RelationshipState, SemanticFact


class Neo4jGraphStore:
    def __init__(
        self,
        *,
        url: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ):
        self.url = url.rstrip("/")
        self.user = user
        self.password = password
        self.database = database

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.user and self.password)

    def _headers(self) -> dict[str, str]:
        token = base64.b64encode(f"{self.user}:{self.password}".encode("utf-8")).decode("ascii")
        return {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        }

    def ping(self) -> bool:
        if not self.enabled:
            return False
        try:
            self._execute("RETURN 1 AS ok", {}, strict=True)
            return True
        except Exception:
            logger.exception("Failed to reach Neo4j")
            return False

    def _execute(
        self,
        statement: str,
        parameters: dict[str, Any],
        *,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        payload = {
            "statements": [
                {
                    "statement": statement,
                    "parameters": parameters,
                }
            ]
        }
        try:
            with httpx.Client(timeout=10, headers=self._headers()) as client:
                response = client.post(
                    f"{self.url}/db/{self.database}/tx/commit",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("results", [])
        except Exception:
            if strict:
                raise
            logger.exception("Neo4j request failed")
            return []

    def sync_fact(self, fact: SemanticFact) -> None:
        if not self.enabled:
            return
        statement = """
        MERGE (subject:Entity {project_id: $project_id, user_id: $user_id, key: $subject})
        SET subject.label = $subject
        MERGE (target:Entity {project_id: $project_id, user_id: $user_id, key: $object_text})
        SET target.label = $object_text
        MERGE (subject)-[rel:RELATED {project_id: $project_id, user_id: $user_id, predicate: $predicate}]->(target)
        SET rel.summary = $summary,
            rel.salience = $salience,
            rel.confidence = $confidence,
            rel.updated_at = $updated_at
        """
        self._execute(
            statement,
            {
                "project_id": fact.project_id,
                "user_id": fact.user_id,
                "subject": fact.subject,
                "object_text": fact.object_text,
                "predicate": fact.predicate,
                "summary": fact.summary or f"{fact.subject} {fact.predicate} {fact.object_text}",
                "salience": fact.salience,
                "confidence": fact.confidence,
                "updated_at": fact.updated_at.isoformat(),
            },
        )

    def sync_edge(self, edge: GraphEdge) -> None:
        if not self.enabled:
            return
        statement = """
        MERGE (source:Entity {project_id: $project_id, user_id: $user_id, key: $source_node})
        SET source.label = $source_node
        MERGE (target:Entity {project_id: $project_id, user_id: $user_id, key: $target_node})
        SET target.label = $target_node
        MERGE (source)-[rel:RELATED {project_id: $project_id, user_id: $user_id, predicate: $edge_type}]->(target)
        SET rel.weight = $weight,
            rel.updated_at = $updated_at
        """
        self._execute(
            statement,
            {
                "project_id": edge.project_id,
                "user_id": edge.user_id,
                "source_node": edge.source_node,
                "target_node": edge.target_node,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
                "updated_at": edge.updated_at.isoformat(),
            },
        )

    def sync_relationship_state(
        self,
        *,
        project_id: str,
        user_id: str,
        companion_node: str,
        state: RelationshipState,
    ) -> None:
        if not self.enabled:
            return
        statement = """
        MERGE (user:User {project_id: $project_id, user_id: $user_id, key: $user_node})
        SET user.label = $user_node
        MERGE (companion:Companion {project_id: $project_id, user_id: $user_id, key: $companion_node})
        SET companion.label = $companion_node
        MERGE (user)-[rel:RELATIONSHIP_STATE {project_id: $project_id, user_id: $user_id}]->(companion)
        SET rel.intimacy_level = $intimacy_level,
            rel.trust_score = $trust_score,
            rel.current_stage = $current_stage,
            rel.recent_conflict_level = $recent_conflict_level,
            rel.updated_at = $updated_at
        """
        self._execute(
            statement,
            {
                "project_id": project_id,
                "user_id": user_id,
                "user_node": f"user:{user_id}",
                "companion_node": companion_node,
                "intimacy_level": state.intimacy_level,
                "trust_score": state.trust_score,
                "current_stage": state.current_stage.value,
                "recent_conflict_level": state.recent_conflict_level,
                "updated_at": state.updated_at.isoformat(),
            },
        )

    def search_related(
        self,
        *,
        project_id: str,
        user_id: str,
        entities: list[str],
        limit: int = 10,
    ) -> dict[str, float]:
        if not self.enabled or not entities:
            return {}
        statement = """
        MATCH (n:Entity {project_id: $project_id, user_id: $user_id})
        WHERE n.key IN $entities
        MATCH (n)-[rel:RELATED]-(m)
        RETURN m.key AS key, MAX(COALESCE(rel.weight, 1.0)) AS score
        ORDER BY score DESC
        LIMIT $limit
        """
        results = self._execute(
            statement,
            {
                "project_id": project_id,
                "user_id": user_id,
                "entities": entities,
                "limit": limit,
            },
        )
        if not results:
            return {}
        rows = results[0].get("data") or []
        matches: dict[str, float] = {}
        for row in rows:
            values = row.get("row") or []
            if len(values) >= 2:
                matches[str(values[0])] = float(values[1])
        return matches
