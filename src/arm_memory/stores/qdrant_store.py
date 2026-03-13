from __future__ import annotations

from typing import Any

from loguru import logger
from qdrant_client import QdrantClient, models

from arm_memory.domain.models import MemoryTrace


class QdrantVectorStore:
    def __init__(self, *, url: str, collection: str, vector_size: int):
        self.url = url.rstrip("/")
        self.collection = collection
        self.vector_size = vector_size
        self.client = QdrantClient(url=self.url) if self.enabled else None

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def ping(self) -> bool:
        if not self.enabled or not self.client:
            return False
        try:
            self.client.get_collections()
            return True
        except Exception:
            logger.exception("Failed to reach Qdrant")
            return False

    def ensure_collection(self, *, strict: bool = False) -> None:
        if not self.enabled or not self.client:
            return
        try:
            exists = self.client.collection_exists(self.collection)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            else:
                self._handle_existing_collection(strict=strict)
        except Exception:
            if strict:
                raise
            logger.exception("Failed to ensure Qdrant collection")

    def _handle_existing_collection(self, *, strict: bool = False) -> None:
        try:
            if not self.client:
                return
            config = self.client.get_collection(self.collection).config
            if config.params and config.params.vectors:
                vectors_config = config.params.vectors
                # Handle single default vector config
                if isinstance(vectors_config, models.VectorParams):
                    actual_size = vectors_config.size
                    actual_distance = vectors_config.distance
                elif hasattr(vectors_config, "size") and hasattr(vectors_config, "distance"):
                    actual_size = getattr(vectors_config, "size")
                    actual_distance = getattr(vectors_config, "distance")
                else:
                    return

                if actual_size == self.vector_size and actual_distance == models.Distance.COSINE:
                    logger.info(
                        (
                            "Qdrant collection '{}' already exists with expected config "
                            "(size={}, distance={})."
                        ),
                        self.collection,
                        actual_size,
                        actual_distance.name if hasattr(actual_distance, "name") else str(actual_distance),
                    )
                    return

                logger.warning(
                    (
                        "Qdrant collection '{}' already exists but config differs: "
                        "expected size={} distance={}, got size={} distance={}"
                    ),
                    self.collection,
                    self.vector_size,
                    "Cosine",
                    actual_size,
                    actual_distance.name if hasattr(actual_distance, "name") else str(actual_distance),
                )
                if strict:
                    raise RuntimeError(
                        "Qdrant collection configuration mismatch for "
                        f"{self.collection}: expected size={self.vector_size} distance=Cosine"
                    )
        except Exception as e:
            if strict and isinstance(e, RuntimeError):
                raise
            logger.info(
                "Could not verify existing Qdrant collection '{}'.",
                self.collection,
            )

    def upsert_trace(self, trace: MemoryTrace) -> None:
        if not self.enabled or not self.client or not trace.vector:
            return
        try:
            point = models.PointStruct(
                id=str(trace.trace_id),
                vector=trace.vector,
                payload={
                    "project_id": str(trace.project_id),
                    "user_id": str(trace.user_id),
                    "kind": trace.kind.value,
                    "summary": trace.summary,
                    "salience": trace.salience,
                    "entities": trace.entities,
                    "tags": trace.tags,
                },
            )
            self.client.upsert(
                collection_name=self.collection,
                points=[point],
                wait=True,
            )
        except Exception:
            logger.exception("Failed to upsert trace to Qdrant")

    def search(
        self,
        *,
        project_id: str,
        user_id: str,
        query_vector: list[float],
        limit: int = 10,
    ) -> dict[str, float]:
        if not self.enabled or not self.client or not query_vector:
            return {}
        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=limit,
                with_payload=False,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="project_id", match=models.MatchValue(value=str(project_id))
                        ),
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=str(user_id))
                        ),
                    ]
                ),
            )

            hits = {str(point.id): point.score for point in response.points}
            return hits
        except Exception:
            logger.exception("Failed to search Qdrant")
            return {}

