from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    return Path(raw).expanduser()


def _mask_secret(value: str, *, keep: int = 4) -> str:
    if not value:
        return "(empty)"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}...{value[-keep:]}"


@dataclass(slots=True)
class ARMConfig:
    base_dir: Path
    data_dir: Path
    log_dir: Path
    log_path: Path
    sqlite_path: Path
    policies_dir: Path
    companion_profile_path: Path
    log_level: str = "INFO"
    qdrant_url: str = ""
    qdrant_collection: str = "arm_memory"
    neo4j_url: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    retrieval_top_k: int = 6
    working_memory_turns: int = 12
    user_manual_memory_limit: int = 12
    vector_dimensions: int = 1024
    embedding_provider: str = "mlx"
    embedding_model: str = "mlx-community/Qwen3-Embedding-0.6B-mxfp8"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_timeout: int = 120
    embedding_batch_size: int = 16
    embedding_max_length: int = 512
    embedding_model_path: str = ""
    embedding_cache_dir: str = ""
    profile_item_limit_per_facet: int = 24
    profile_item_stale_days: int = 180
    episodic_half_life_hours: int = 168
    semantic_half_life_hours: int = 2160
    reconsolidation_boost: float = 1.15
    retrieval_weight_semantic: float = 0.40
    retrieval_weight_lexical: float = 0.15
    retrieval_weight_graph: float = 0.20
    retrieval_weight_salience: float = 0.15
    retrieval_weight_remote: float = 0.10
    retrieval_candidate_limit: int = 300
    retrieval_qdrant_prefetch_factor: int = 2
    memory_archive_min_age_days: int = 90
    memory_archive_min_salience: float = 0.20
    memory_merge_similarity_threshold: float = 0.85
    memory_merge_skip_emotion_contradiction: bool = True
    salience_boost_per_access: float = 0.02
    salience_cap: float = 1.0
    companion_node_name: str = "companion:self"
    service_host: str = "127.0.0.1"
    service_port: int = 8788
    service_ws_path: str = "/ws"
    outbox_max_attempts: int = 5
    outbox_retry_base_seconds: int = 5
    outbox_retry_max_seconds: int = 300
    outbox_reclaim_in_progress_after_seconds: int = 300
    outbox_replay_interval_seconds: int = 60
    outbox_replay_limit: int = 50
    require_qdrant: bool = True
    require_neo4j: bool = True

    @classmethod
    def from_env(cls, *, base_dir: Path | None = None) -> "ARMConfig":
        # 项目根目录（pyproject.toml 所在）；package_dir = src/arm_memory
        _here = Path(__file__).resolve().parent
        package_dir = _here
        project_root = (base_dir or _here.parent.parent).resolve()
        data_dir = _env_path("ARM_DATA_DIR", project_root / "data" / "arm_memory")
        log_dir = _env_path("ARM_LOG_DIR", data_dir / "logs")
        log_path = _env_path("ARM_LOG_PATH", log_dir / "arm_memory.log")
        sqlite_path = _env_path("ARM_SQLITE_PATH", data_dir / "arm_memory.db")
        policies_dir = _env_path("ARM_POLICIES_DIR", project_root / "policies")
        companion_profile_path = _env_path(
            "COMPANION_PROFILE_PATH",
            project_root / "personas" / "companion_profile.md",
        )
        return cls(
            base_dir=project_root,
            data_dir=data_dir,
            log_dir=log_dir,
            log_path=log_path,
            sqlite_path=sqlite_path,
            policies_dir=policies_dir,
            companion_profile_path=companion_profile_path,
            log_level=os.getenv("ARM_LOG_LEVEL", "INFO"),
            qdrant_url=os.getenv("ARM_QDRANT_URL", ""),
            qdrant_collection=os.getenv("ARM_QDRANT_COLLECTION", "arm_memory"),
            neo4j_url=os.getenv("ARM_NEO4J_URL", ""),
            neo4j_user=os.getenv("ARM_NEO4J_USER", ""),
            neo4j_password=os.getenv("ARM_NEO4J_PASSWORD", ""),
            neo4j_database=os.getenv("ARM_NEO4J_DATABASE", "neo4j"),
            retrieval_top_k=int(os.getenv("ARM_RETRIEVAL_TOP_K", "6")),
            working_memory_turns=int(os.getenv("ARM_WORKING_MEMORY_TURNS", "12")),
            user_manual_memory_limit=int(os.getenv("ARM_USER_MANUAL_MEMORY_LIMIT", "12")),
            vector_dimensions=int(os.getenv("ARM_VECTOR_DIMENSIONS", "1024")),
            embedding_provider=os.getenv("ARM_EMBEDDING_PROVIDER", "mlx"),
            embedding_model=os.getenv("ARM_EMBEDDING_MODEL", "mlx-community/Qwen3-Embedding-0.6B-mxfp8"),
            embedding_base_url=(os.getenv("ARM_EMBEDDING_BASE_URL") or "").strip(),
            embedding_api_key=os.getenv("ARM_EMBEDDING_API_KEY", ""),
            embedding_timeout=int(os.getenv("ARM_EMBEDDING_TIMEOUT", "120")),
            embedding_batch_size=int(os.getenv("ARM_EMBEDDING_BATCH_SIZE", "16")),
            embedding_max_length=int(os.getenv("ARM_EMBEDDING_MAX_LENGTH", "512")),
            embedding_model_path=(os.getenv("ARM_EMBEDDING_MODEL_PATH") or "").strip(),
            embedding_cache_dir=(
                (os.getenv("ARM_EMBEDDING_CACHE_DIR") or "").strip()
                or str((data_dir / "embedding_cache").resolve())
            ),
            profile_item_limit_per_facet=int(os.getenv("ARM_PROFILE_ITEM_LIMIT_PER_FACET", "24")),
            profile_item_stale_days=int(os.getenv("ARM_PROFILE_ITEM_STALE_DAYS", "180")),
            episodic_half_life_hours=int(os.getenv("ARM_EPISODIC_HALF_LIFE_HOURS", "168")),
            semantic_half_life_hours=int(os.getenv("ARM_SEMANTIC_HALF_LIFE_HOURS", "2160")),
            reconsolidation_boost=float(os.getenv("ARM_RECONSOLIDATION_BOOST", "1.15")),
            retrieval_weight_semantic=float(os.getenv("ARM_RETRIEVAL_WEIGHT_SEMANTIC", "0.40")),
            retrieval_weight_lexical=float(os.getenv("ARM_RETRIEVAL_WEIGHT_LEXICAL", "0.15")),
            retrieval_weight_graph=float(os.getenv("ARM_RETRIEVAL_WEIGHT_GRAPH", "0.20")),
            retrieval_weight_salience=float(os.getenv("ARM_RETRIEVAL_WEIGHT_SALIENCE", "0.15")),
            retrieval_weight_remote=float(os.getenv("ARM_RETRIEVAL_WEIGHT_REMOTE", "0.10")),
            retrieval_candidate_limit=int(os.getenv("ARM_RETRIEVAL_CANDIDATE_LIMIT", "300")),
            retrieval_qdrant_prefetch_factor=int(os.getenv("ARM_RETRIEVAL_QDRANT_PREFETCH_FACTOR", "2")),
            memory_archive_min_age_days=int(os.getenv("ARM_MEMORY_ARCHIVE_MIN_AGE_DAYS", "90")),
            memory_archive_min_salience=float(os.getenv("ARM_MEMORY_ARCHIVE_MIN_SALIENCE", "0.20")),
            memory_merge_similarity_threshold=float(os.getenv("ARM_MEMORY_MERGE_SIMILARITY_THRESHOLD", "0.85")),
            memory_merge_skip_emotion_contradiction=_env_bool("ARM_MEMORY_MERGE_SKIP_EMOTION_CONTRADICTION", True),
            salience_boost_per_access=float(os.getenv("ARM_SALIENCE_BOOST_PER_ACCESS", "0.02")),
            salience_cap=float(os.getenv("ARM_SALIENCE_CAP", "1.0")),
            companion_node_name=os.getenv("ARM_COMPANION_NODE_NAME", "companion:self"),
            service_host=os.getenv("ARM_SERVICE_HOST", "127.0.0.1"),
            service_port=int(os.getenv("ARM_SERVICE_PORT", "8788")),
            service_ws_path=os.getenv("ARM_SERVICE_WS_PATH", "/ws"),
            outbox_max_attempts=int(os.getenv("ARM_OUTBOX_MAX_ATTEMPTS", "5")),
            outbox_retry_base_seconds=int(os.getenv("ARM_OUTBOX_RETRY_BASE_SECONDS", "5")),
            outbox_retry_max_seconds=int(os.getenv("ARM_OUTBOX_RETRY_MAX_SECONDS", "300")),
            outbox_reclaim_in_progress_after_seconds=int(
                os.getenv("ARM_OUTBOX_RECLAIM_IN_PROGRESS_AFTER_SECONDS", "300")
            ),
            outbox_replay_interval_seconds=int(os.getenv("ARM_OUTBOX_REPLAY_INTERVAL_SECONDS", "60")),
            outbox_replay_limit=int(os.getenv("ARM_OUTBOX_REPLAY_LIMIT", "50")),
            require_qdrant=True,
            require_neo4j=True,
        )

    @property
    def remote_backends_enabled(self) -> bool:
        return bool(self.qdrant_url or self.neo4j_url)

    @property
    def qdrant_enabled(self) -> bool:
        return bool(self.qdrant_url)

    @property
    def neo4j_enabled(self) -> bool:
        return bool(self.neo4j_url and self.neo4j_user and self.neo4j_password)

    def summary_lines(self) -> list[str]:
        return [
            f"[ARM] base_dir={self.base_dir} data_dir={self.data_dir} sqlite={self.sqlite_path}",
            (
                "[Files] "
                f"profile={self.companion_profile_path} policies={self.policies_dir} "
                f"companion_node={self.companion_node_name}"
            ),
            (
                "[Logging] "
                f"level={self.log_level} log_dir={self.log_dir} log_path={self.log_path}"
            ),
            (
                "[Retrieval] "
                f"top_k={self.retrieval_top_k} working_turns={self.working_memory_turns} "
                f"user_manual_limit={self.user_manual_memory_limit} vector_dims={self.vector_dimensions} "
                f"candidate_limit={self.retrieval_candidate_limit} "
                f"weights=sem:{self.retrieval_weight_semantic}/lex:{self.retrieval_weight_lexical}/"
                f"graph:{self.retrieval_weight_graph}/sal:{self.retrieval_weight_salience}/"
                f"remote:{self.retrieval_weight_remote}"
            ),
            (
                "[Embedding] "
                f"provider={self.embedding_provider} model={self.embedding_model} "
                f"model_path={self.embedding_model_path or '(default)'} cache_dir={self.embedding_cache_dir} "
                f"base_url={self.embedding_base_url} batch_size={self.embedding_batch_size} "
                f"max_length={self.embedding_max_length}"
            ),
            (
                "[ProfileItems] "
                f"limit_per_facet={self.profile_item_limit_per_facet} "
                f"stale_days={self.profile_item_stale_days}"
            ),
            (
                "[Decay] "
                f"episodic_half_life_h={self.episodic_half_life_hours} "
                f"semantic_half_life_h={self.semantic_half_life_hours} "
                f"reconsolidation_boost={self.reconsolidation_boost}"
            ),
            (
                "[Service] "
                f"host={self.service_host} port={self.service_port} ws_path={self.service_ws_path} "
                f"require_qdrant={self.require_qdrant} require_neo4j={self.require_neo4j}"
            ),
            (
                "[Outbox] "
                f"max_attempts={self.outbox_max_attempts} "
                f"retry_base_s={self.outbox_retry_base_seconds} "
                f"retry_max_s={self.outbox_retry_max_seconds} "
                f"reclaim_in_progress_after_s={self.outbox_reclaim_in_progress_after_seconds} "
                f"replay_interval_s={self.outbox_replay_interval_seconds} "
                f"replay_limit={self.outbox_replay_limit}"
            ),
            f"[Qdrant] enabled={self.qdrant_enabled} url={self.qdrant_url or '(disabled)'} collection={self.qdrant_collection}",
            (
                "[Neo4j] "
                f"enabled={self.neo4j_enabled} url={self.neo4j_url or '(disabled)'} "
                f"user={self.neo4j_user or '(empty)'} password={_mask_secret(self.neo4j_password)} "
                f"database={self.neo4j_database}"
            ),
        ]
