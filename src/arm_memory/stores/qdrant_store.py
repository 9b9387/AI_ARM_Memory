from __future__ import annotations

import logging
from typing import Any

import httpx

from arm_memory.domain.models import MemoryTrace

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(self, *, url: str, collection: str, vector_size: int):
        self.url = url.rstrip("/")
        self.collection = collection
        self.vector_size = vector_size

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def ping(self) -> bool:
        if not self.enabled:
            return False
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.url}/collections")
                response.raise_for_status()
            return True
        except Exception:
            logger.warning("Failed to reach Qdrant", exc_info=True)
            return False

    def ensure_collection(self, *, strict: bool = False) -> None:
        if not self.enabled:
            return
        payload = {
            "vectors": {
                "size": self.vector_size,
                "distance": "Cosine",
            }
        }
        try:
            with httpx.Client(timeout=10) as client:
                response = client.put(
                    f"{self.url}/collections/{self.collection}",
                    json=payload,
                )
                if response.status_code == 409:
                    self._handle_existing_collection(client, strict=strict)
                    return
                response.raise_for_status()
        except Exception:
            if strict:
                raise
            logger.warning("Failed to ensure Qdrant collection", exc_info=True)

    def _handle_existing_collection(self, client: httpx.Client, *, strict: bool = False) -> None:
        try:
            response = client.get(f"{self.url}/collections/{self.collection}")
            response.raise_for_status()
            config = (
                response.json()
                .get("result", {})
                .get("config", {})
                .get("params", {})
                .get("vectors", {})
            )
        except Exception:
            logger.info(
                "Qdrant collection '%s' already exists; skipped creation.",
                self.collection,
            )
            return

        actual_size = config.get("size")
        actual_distance = str(config.get("distance", "")).lower()
        expected_distance = "cosine"
        if actual_size == self.vector_size and actual_distance == expected_distance:
            logger.info(
                (
                    "Qdrant collection '%s' already exists with expected config "
                    "(size=%s, distance=%s)."
                ),
                self.collection,
                actual_size,
                config.get("distance"),
            )
            return

        logger.warning(
            (
                "Qdrant collection '%s' already exists but config differs: "
                "expected size=%s distance=%s, got size=%s distance=%s"
            ),
            self.collection,
            self.vector_size,
            "Cosine",
            actual_size,
            config.get("distance"),
        )
        if strict:
            raise RuntimeError(
                "Qdrant collection configuration mismatch for "
                f"{self.collection}: expected size={self.vector_size} distance=Cosine"
            )

    def upsert_trace(self, trace: MemoryTrace) -> None:
        if not self.enabled or not trace.vector:
            return
        payload = {
            "points": [
                {
                    "id": trace.trace_id,
                    "vector": trace.vector,
                    "payload": {
                        "project_id": trace.project_id,
                        "user_id": trace.user_id,
                        "kind": trace.kind.value,
                        "summary": trace.summary,
                        "salience": trace.salience,
                        "entities": trace.entities,
                        "tags": trace.tags,
                    },
                }
            ]
        }
        try:
            with httpx.Client(timeout=10) as client:
                response = client.put(
                    f"{self.url}/collections/{self.collection}/points?wait=true",
                    json=payload,
                )
                response.raise_for_status()
        except Exception:
            logger.warning("Failed to upsert trace to Qdrant", exc_info=True)

    def search(
        self,
        *,
        project_id: str,
        user_id: str,
        query_vector: list[float],
        limit: int = 10,
    ) -> dict[str, float]:
        if not self.enabled or not query_vector:
            return {}
        payload: dict[str, Any] = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": False,
            "filter": {
                "must": [
                    {"key": "project_id", "match": {"value": project_id}},
                    {"key": "user_id", "match": {"value": user_id}},
                ]
            },
        }
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(
                    f"{self.url}/collections/{self.collection}/points/search",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json().get("result") or []
        except Exception:
            logger.warning("Failed to search Qdrant", exc_info=True)
            return {}

        hits: dict[str, float] = {}
        for item in result:
            point_id = str(item.get("id", ""))
            score = float(item.get("score", 0.0))
            if point_id:
                hits[point_id] = score
        return hits
