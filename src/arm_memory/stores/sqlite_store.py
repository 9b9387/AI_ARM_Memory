from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

from loguru import logger

from arm_memory.domain.models import (
    ConversationTurn,
    EmotionVector,
    GraphEdge,
    MemoryKind,
    MemoryOperation,
    MemoryTrace,
    PersonaDefinition,
    ProfileFacetType,
    ProfileItemStatus,
    SemanticFact,
    SensitivityLevel,
    UserProfile,
    UserProfileItem,
    UserManualSnapshot,
    persona_from_dict,
    profile_facet_type_from_value,
    profile_from_dict,
    profile_item_from_dict,
    relationship_from_dict,
    trace_from_dict,
    RelationshipState,
)
from arm_memory.utils import dumps_json, loads_json, normalize_text, parse_iso_datetime, to_iso, utcnow


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[override]
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.close()


class SQLiteMemoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            factory=_ClosingConnection,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS session_turns (
                    turn_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    consolidated INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_session_turns_lookup
                    ON session_turns(project_id, user_id, created_at);

                CREATE TABLE IF NOT EXISTS memory_traces (
                    trace_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    raw_text TEXT NOT NULL DEFAULT '',
                    normalized_summary TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    salience REAL NOT NULL DEFAULT 0.5,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    source_turn_ids_json TEXT NOT NULL DEFAULT '[]',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    entities_json TEXT NOT NULL DEFAULT '[]',
                    vector_json TEXT NOT NULL DEFAULT '[]',
                    emotion_json TEXT NOT NULL DEFAULT '{}',
                    sensitivity TEXT NOT NULL DEFAULT 'private',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    happened_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_memory_trace_lookup
                    ON memory_traces(project_id, user_id, kind, status, created_at);

                CREATE TABLE IF NOT EXISTS semantic_facts (
                    fact_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_text TEXT NOT NULL,
                    normalized_key TEXT NOT NULL,
                    normalized_object TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.75,
                    salience REAL NOT NULL DEFAULT 0.65,
                    source_trace_id TEXT,
                    source_turn_ids_json TEXT NOT NULL DEFAULT '[]',
                    entities_json TEXT NOT NULL DEFAULT '[]',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    emotion_json TEXT NOT NULL DEFAULT '{}',
                    sensitivity TEXT NOT NULL DEFAULT 'private',
                    status TEXT NOT NULL DEFAULT 'active',
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_semantic_fact_lookup
                    ON semantic_facts(project_id, user_id, normalized_key, status, updated_at);

                CREATE TABLE IF NOT EXISTS relationship_states (
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    intimacy_level REAL NOT NULL,
                    trust_score REAL NOT NULL,
                    attachment_style_guess TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    recent_conflict_level REAL NOT NULL,
                    boundary_flags_json TEXT NOT NULL DEFAULT '[]',
                    last_emotional_peak_at TEXT,
                    notes_json TEXT NOT NULL DEFAULT '[]',
                    behavior_state_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS user_profiles (
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    identity_notes_json TEXT NOT NULL DEFAULT '[]',
                    preferences_json TEXT NOT NULL DEFAULT '[]',
                    dislikes_json TEXT NOT NULL DEFAULT '[]',
                    boundaries_json TEXT NOT NULL DEFAULT '[]',
                    vulnerabilities_json TEXT NOT NULL DEFAULT '[]',
                    support_preferences_json TEXT NOT NULL DEFAULT '[]',
                    goals_json TEXT NOT NULL DEFAULT '[]',
                    important_people_json TEXT NOT NULL DEFAULT '[]',
                    important_places_json TEXT NOT NULL DEFAULT '[]',
                    manual_notes_json TEXT NOT NULL DEFAULT '[]',
                    personality_traits_json TEXT NOT NULL DEFAULT '[]',
                    hobbies_json TEXT NOT NULL DEFAULT '[]',
                    habits_json TEXT NOT NULL DEFAULT '[]',
                    communication_style_json TEXT NOT NULL DEFAULT '[]',
                    attachment_style_json TEXT NOT NULL DEFAULT '[]',
                    life_routines_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS profile_items (
                    item_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    facet_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    source TEXT NOT NULL DEFAULT 'unknown',
                    source_trace_id TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    evidence_count INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'active',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_profile_items_lookup
                    ON profile_items(project_id, user_id, facet_type, status, last_seen_at);

                CREATE TABLE IF NOT EXISTS persona_registry (
                    persona_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    policy_pack TEXT NOT NULL DEFAULT 'default',
                    description TEXT NOT NULL DEFAULT '',
                    is_enabled INTEGER NOT NULL DEFAULT 1,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    source_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_persona_registry_lookup
                    ON persona_registry(project_id, user_id, is_active, is_enabled, updated_at);

                CREATE TABLE IF NOT EXISTS user_manual_snapshots (
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS graph_nodes (
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    node_key TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, user_id, node_key)
                );

                CREATE TABLE IF NOT EXISTS graph_edges (
                    edge_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    source_node TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    target_node TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    emotion_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'active',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_graph_edges_lookup
                    ON graph_edges(project_id, user_id, source_node, target_node, edge_type, status);

                CREATE TABLE IF NOT EXISTS remote_sync_outbox (
                    task_id TEXT PRIMARY KEY,
                    backend TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    error_message TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    next_attempt_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_remote_sync_outbox_lookup
                    ON remote_sync_outbox(status, backend, next_attempt_at, updated_at);
                """
            )
            self._ensure_column(
                conn,
                table="relationship_states",
                column="behavior_state_json",
                definition="TEXT NOT NULL DEFAULT '{}'",
            )
            for column in (
                ("personality_traits_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("hobbies_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("habits_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("communication_style_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("attachment_style_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("life_routines_json", "TEXT NOT NULL DEFAULT '[]'"),
            ):
                self._ensure_column(conn, table="user_profiles", column=column[0], definition=column[1])
            self._ensure_column(
                conn,
                table="remote_sync_outbox",
                column="next_attempt_at",
                definition="TEXT NOT NULL DEFAULT ''",
            )

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        *,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column in columns:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def transaction(self):
        return self._connect()

    def store_turn(self, turn: ConversationTurn, *, conn: sqlite3.Connection | None = None) -> None:
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            active_conn.execute(
                """
                INSERT OR REPLACE INTO session_turns (
                    turn_id, project_id, user_id, session_id, role, content,
                    created_at, consolidated, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_id,
                    turn.project_id,
                    turn.user_id,
                    turn.session_id,
                    turn.role,
                    turn.content,
                    to_iso(turn.created_at),
                    0,
                    dumps_json(turn.metadata),
                ),
            )

    def list_recent_turns(
        self,
        project_id: str,
        user_id: str,
        *,
        limit: int = 20,
        session_id: str | None = None,
        unconsolidated_only: bool = False,
    ) -> list[ConversationTurn]:
        clauses = ["project_id = ?", "user_id = ?"]
        params: list[object] = [project_id, user_id]
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if unconsolidated_only:
            clauses.append("consolidated = 0")
        sql = (
            "SELECT * FROM session_turns WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC LIMIT ?"
        )
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        turns = [self._row_to_turn(row) for row in rows]
        turns.reverse()
        return turns

    def mark_turns_consolidated(
        self,
        turn_ids: Sequence[str],
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not turn_ids:
            return
        placeholders = ", ".join("?" for _ in turn_ids)
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            active_conn.execute(
                f"UPDATE session_turns SET consolidated = 1 WHERE turn_id IN ({placeholders})",
                list(turn_ids),
            )

    def add_memory_trace(
        self,
        trace: MemoryTrace,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> MemoryOperation:
        normalized_summary = normalize_text(trace.summary)
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            existing = active_conn.execute(
                """
                SELECT trace_id FROM memory_traces
                WHERE project_id = ? AND user_id = ? AND kind = ?
                  AND status = 'active' AND normalized_summary = ?
                LIMIT 1
                """,
                (trace.project_id, trace.user_id, trace.kind.value, normalized_summary),
            ).fetchone()
            if existing:
                active_conn.execute(
                    """
                    UPDATE memory_traces
                    SET salience = MAX(salience, ?),
                        updated_at = ?
                    WHERE trace_id = ?
                    """,
                    (trace.salience, to_iso(trace.updated_at), existing["trace_id"]),
                )
                return MemoryOperation.NOOP

            active_conn.execute(
                """
                INSERT INTO memory_traces (
                    trace_id, project_id, user_id, kind, summary, raw_text,
                    normalized_summary, status, salience, access_count,
                    source_turn_ids_json, tags_json, entities_json, vector_json,
                    emotion_json, sensitivity, created_at, updated_at,
                    happened_at, last_accessed_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.project_id,
                    trace.user_id,
                    trace.kind.value,
                    trace.summary,
                    trace.raw_text,
                    normalized_summary,
                    trace.status,
                    trace.salience,
                    trace.access_count,
                    dumps_json(trace.source_turn_ids),
                    dumps_json(trace.tags),
                    dumps_json(trace.entities),
                    dumps_json(trace.vector),
                    dumps_json(trace.emotion.to_dict()),
                    trace.sensitivity.value,
                    to_iso(trace.created_at),
                    to_iso(trace.updated_at),
                    to_iso(trace.happened_at),
                    to_iso(trace.last_accessed_at),
                    dumps_json(trace.metadata),
                ),
            )
        return MemoryOperation.ADD

    def get_trace(self, trace_id: str) -> MemoryTrace | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        return self._row_to_trace(row) if row else None

    def get_traces_by_ids(self, trace_ids: Sequence[str]) -> list[MemoryTrace]:
        if not trace_ids:
            return []
        placeholders = ", ".join("?" for _ in trace_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_traces WHERE trace_id IN ({placeholders})",
                list(trace_ids),
            ).fetchall()
        traces = [self._row_to_trace(row) for row in rows]
        index = {trace.trace_id: trace for trace in traces}
        return [index[trace_id] for trace_id in trace_ids if trace_id in index]

    def list_active_memory_traces(
        self,
        project_id: str,
        user_id: str,
        *,
        kinds: Sequence[MemoryKind] | None = None,
        limit: int = 200,
        conn: sqlite3.Connection | None = None,
    ) -> list[MemoryTrace]:
        params: list[object] = [project_id, user_id]
        clauses = ["project_id = ?", "user_id = ?", "status = 'active'"]
        if kinds:
            placeholders = ", ".join("?" for _ in kinds)
            clauses.append(f"kind IN ({placeholders})")
            params.extend(kind.value for kind in kinds)
        sql = (
            "SELECT * FROM memory_traces WHERE "
            + " AND ".join(clauses)
            + " ORDER BY updated_at DESC LIMIT ?"
        )
        params.append(limit)
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            rows = active_conn.execute(sql, params).fetchall()
        return [self._row_to_trace(row) for row in rows]

    def record_memory_access(
        self,
        trace_id: str,
        *,
        salience_boost: float = 0.0,
        salience_cap: float = 1.0,
    ) -> None:
        now = to_iso(utcnow())
        with self._connect() as conn:
            if salience_boost > 0:
                conn.execute(
                    """
                    UPDATE memory_traces
                    SET access_count = access_count + 1,
                        last_accessed_at = ?,
                        updated_at = ?,
                        salience = MIN(?, salience + ?)
                    WHERE trace_id = ?
                    """,
                    (now, now, salience_cap, salience_boost, trace_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE memory_traces
                    SET access_count = access_count + 1,
                        last_accessed_at = ?,
                        updated_at = ?
                    WHERE trace_id = ?
                    """,
                    (now, now, trace_id),
                )

    def apply_semantic_fact(
        self,
        fact: SemanticFact,
        *,
        operation: MemoryOperation = MemoryOperation.ADD,
        conn: sqlite3.Connection | None = None,
    ) -> MemoryOperation:
        normalized_key = fact.normalized_key
        normalized_object = normalize_text(fact.object_text)
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            active_rows = active_conn.execute(
                """
                SELECT * FROM semantic_facts
                WHERE project_id = ? AND user_id = ? AND normalized_key = ? AND status = 'active'
                ORDER BY updated_at DESC
                """,
                (fact.project_id, fact.user_id, normalized_key),
            ).fetchall()

            if operation == MemoryOperation.DELETE:
                if not active_rows:
                    return MemoryOperation.NOOP
                now = to_iso(utcnow())
                active_conn.execute(
                    """
                    UPDATE semantic_facts
                    SET status = 'deleted', valid_to = ?, updated_at = ?
                    WHERE project_id = ? AND user_id = ? AND normalized_key = ? AND status = 'active'
                    """,
                    (now, now, fact.project_id, fact.user_id, normalized_key),
                )
                return MemoryOperation.DELETE

            for row in active_rows:
                if row["normalized_object"] == normalized_object:
                    active_conn.execute(
                        """
                        UPDATE semantic_facts
                        SET confidence = MAX(confidence, ?),
                            salience = MAX(salience, ?),
                            updated_at = ?
                        WHERE fact_id = ?
                        """,
                        (fact.confidence, fact.salience, to_iso(fact.updated_at), row["fact_id"]),
                    )
                    return MemoryOperation.NOOP

            result = MemoryOperation.ADD
            if active_rows:
                result = MemoryOperation.UPDATE
                now = to_iso(utcnow())
                active_conn.execute(
                    """
                    UPDATE semantic_facts
                    SET status = 'superseded', valid_to = ?, updated_at = ?
                    WHERE project_id = ? AND user_id = ? AND normalized_key = ? AND status = 'active'
                    """,
                    (now, now, fact.project_id, fact.user_id, normalized_key),
                )

            active_conn.execute(
                """
                INSERT INTO semantic_facts (
                    fact_id, project_id, user_id, subject, predicate, object_text,
                    normalized_key, normalized_object, summary, confidence, salience,
                    source_trace_id, source_turn_ids_json, entities_json, tags_json,
                    emotion_json, sensitivity, status, valid_from, valid_to,
                    created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.fact_id,
                    fact.project_id,
                    fact.user_id,
                    fact.subject,
                    fact.predicate,
                    fact.object_text,
                    normalized_key,
                    normalized_object,
                    fact.summary or f"{fact.subject} {fact.predicate} {fact.object_text}",
                    fact.confidence,
                    fact.salience,
                    fact.source_trace_id,
                    dumps_json(fact.source_turn_ids),
                    dumps_json(fact.entities),
                    dumps_json(fact.tags),
                    dumps_json(fact.emotion.to_dict()),
                    fact.sensitivity.value,
                    fact.status,
                    to_iso(fact.valid_from),
                    to_iso(fact.valid_to) if fact.valid_to else None,
                    to_iso(fact.created_at),
                    to_iso(fact.updated_at),
                    dumps_json(fact.metadata),
                ),
            )
        return result

    def list_semantic_facts(
        self,
        project_id: str,
        user_id: str,
        *,
        limit: int = 100,
        active_only: bool = True,
        conn: sqlite3.Connection | None = None,
    ) -> list[SemanticFact]:
        clauses = ["project_id = ?", "user_id = ?"]
        params: list[object] = [project_id, user_id]
        if active_only:
            clauses.append("status = 'active'")
        sql = (
            "SELECT * FROM semantic_facts WHERE "
            + " AND ".join(clauses)
            + " ORDER BY updated_at DESC LIMIT ?"
        )
        params.append(limit)
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            rows = active_conn.execute(sql, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def save_profile(self, profile: UserProfile, *, conn: sqlite3.Connection | None = None) -> None:
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            active_conn.execute(
                """
                INSERT OR REPLACE INTO user_profiles (
                    project_id, user_id, display_name, identity_notes_json,
                    preferences_json, dislikes_json, boundaries_json,
                    vulnerabilities_json, support_preferences_json, goals_json,
                    important_people_json, important_places_json, manual_notes_json,
                    personality_traits_json, hobbies_json, habits_json,
                    communication_style_json, attachment_style_json, life_routines_json,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.project_id,
                    profile.user_id,
                    profile.display_name,
                    dumps_json(profile.identity_notes),
                    dumps_json(profile.preferences),
                    dumps_json(profile.dislikes),
                    dumps_json(profile.boundaries),
                    dumps_json(profile.vulnerabilities),
                    dumps_json(profile.support_preferences),
                    dumps_json(profile.goals),
                    dumps_json(profile.important_people),
                    dumps_json(profile.important_places),
                    dumps_json(profile.manual_notes),
                    dumps_json(profile.personality_traits),
                    dumps_json(profile.hobbies),
                    dumps_json(profile.habits),
                    dumps_json(profile.communication_style),
                    dumps_json(profile.attachment_style),
                    dumps_json(profile.life_routines),
                    to_iso(profile.updated_at),
                ),
            )

    def load_profile(self, project_id: str, user_id: str) -> UserProfile:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            ).fetchone()
        if not row:
            return UserProfile(project_id=project_id, user_id=user_id)
        payload = {
            "project_id": row["project_id"],
            "user_id": row["user_id"],
            "display_name": row["display_name"],
            "identity_notes": loads_json(row["identity_notes_json"], default=[]),
            "preferences": loads_json(row["preferences_json"], default=[]),
            "dislikes": loads_json(row["dislikes_json"], default=[]),
            "boundaries": loads_json(row["boundaries_json"], default=[]),
            "vulnerabilities": loads_json(row["vulnerabilities_json"], default=[]),
            "support_preferences": loads_json(row["support_preferences_json"], default=[]),
            "goals": loads_json(row["goals_json"], default=[]),
            "important_people": loads_json(row["important_people_json"], default=[]),
            "important_places": loads_json(row["important_places_json"], default=[]),
            "manual_notes": loads_json(row["manual_notes_json"], default=[]),
            "personality_traits": loads_json(row["personality_traits_json"], default=[]),
            "hobbies": loads_json(row["hobbies_json"], default=[]),
            "habits": loads_json(row["habits_json"], default=[]),
            "communication_style": loads_json(row["communication_style_json"], default=[]),
            "attachment_style": loads_json(row["attachment_style_json"], default=[]),
            "life_routines": loads_json(row["life_routines_json"], default=[]),
            "updated_at": row["updated_at"],
        }
        return profile_from_dict(payload)

    def save_profile_item(
        self,
        item: UserProfileItem,
        *,
        conn: sqlite3.Connection | None = None,
        facet_limit: int | None = None,
    ) -> None:
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            existing = active_conn.execute(
                """
                SELECT item_id, confidence, evidence_count
                FROM profile_items
                WHERE project_id = ? AND user_id = ? AND facet_type = ?
                  AND normalized_value = ? AND status != ?
                LIMIT 1
                """,
                (
                    item.project_id,
                    item.user_id,
                    item.facet_type.value,
                    item.normalized_value,
                    ProfileItemStatus.DELETED.value,
                ),
            ).fetchone()
            if existing:
                existing_confidence = float(existing["confidence"])
                existing_evidence = max(1, int(existing["evidence_count"]))
                next_evidence = max(existing_evidence + 1, item.evidence_count)
                blended_confidence = (
                    (existing_confidence * existing_evidence) + max(0.0, min(1.0, item.confidence))
                ) / next_evidence
                active_conn.execute(
                    """
                    UPDATE profile_items
                    SET confidence = ?,
                        source = ?,
                        source_trace_id = COALESCE(?, source_trace_id),
                        last_seen_at = ?,
                        evidence_count = ?,
                        status = ?,
                        metadata_json = ?
                    WHERE item_id = ?
                    """,
                    (
                        blended_confidence,
                        item.source,
                        item.source_trace_id,
                        to_iso(item.last_seen_at),
                        next_evidence,
                        item.status.value,
                        dumps_json(item.metadata),
                        existing["item_id"],
                    ),
                )
                if facet_limit is not None:
                    self._archive_excess_profile_items(
                        active_conn,
                        project_id=item.project_id,
                        user_id=item.user_id,
                        facet_type=item.facet_type,
                        keep_limit=facet_limit,
                    )
                return
            active_conn.execute(
                """
                INSERT INTO profile_items (
                    item_id, project_id, user_id, facet_type, value, normalized_value,
                    confidence, source, source_trace_id, first_seen_at, last_seen_at,
                    evidence_count, status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.item_id,
                    item.project_id,
                    item.user_id,
                    item.facet_type.value,
                    item.value,
                    item.normalized_value,
                    item.confidence,
                    item.source,
                    item.source_trace_id,
                    to_iso(item.first_seen_at),
                    to_iso(item.last_seen_at),
                    item.evidence_count,
                    item.status.value,
                    dumps_json(item.metadata),
                ),
            )
            if facet_limit is not None:
                self._archive_excess_profile_items(
                    active_conn,
                    project_id=item.project_id,
                    user_id=item.user_id,
                    facet_type=item.facet_type,
                    keep_limit=facet_limit,
                )

    def list_profile_items(
        self,
        project_id: str,
        user_id: str,
        *,
        facet_types: Sequence[ProfileFacetType | str] | None = None,
        status: ProfileItemStatus | str | None = ProfileItemStatus.ACTIVE,
        max_age_days: int | None = None,
        limit: int = 200,
        conn: sqlite3.Connection | None = None,
    ) -> list[UserProfileItem]:
        clauses = ["project_id = ?", "user_id = ?"]
        params: list[object] = [project_id, user_id]
        if facet_types:
            normalized = [
                item.value if isinstance(item, ProfileFacetType) else profile_facet_type_from_value(str(item)).value
                for item in facet_types
            ]
            placeholders = ", ".join("?" for _ in normalized)
            clauses.append(f"facet_type IN ({placeholders})")
            params.extend(normalized)
        if status is not None:
            status_value = status.value if isinstance(status, ProfileItemStatus) else str(status)
            clauses.append("status = ?")
            params.append(status_value)
        if max_age_days is not None and max_age_days > 0:
            clauses.append("last_seen_at >= ?")
            params.append(to_iso(utcnow() - timedelta(days=max_age_days)))
        sql = (
            "SELECT * FROM profile_items WHERE "
            + " AND ".join(clauses)
            + " ORDER BY last_seen_at DESC LIMIT ?"
        )
        params.append(limit)
        
        if conn:
            rows = conn.execute(sql, params).fetchall()
        else:
            with self._connect() as temp_conn:
                rows = temp_conn.execute(sql, params).fetchall()
                
        return [self._row_to_profile_item(row) for row in rows]

    def _archive_excess_profile_items(
        self,
        conn: sqlite3.Connection,
        *,
        project_id: str,
        user_id: str,
        facet_type: ProfileFacetType,
        keep_limit: int,
    ) -> None:
        if keep_limit <= 0:
            return
        rows = conn.execute(
            """
            SELECT item_id
            FROM profile_items
            WHERE project_id = ? AND user_id = ? AND facet_type = ? AND status = ?
            ORDER BY confidence DESC, evidence_count DESC, last_seen_at DESC
            LIMIT -1 OFFSET ?
            """,
            (project_id, user_id, facet_type.value, ProfileItemStatus.ACTIVE.value, keep_limit),
        ).fetchall()
        archived_ids = [str(row["item_id"]) for row in rows]
        if not archived_ids:
            return
        placeholders = ", ".join("?" for _ in archived_ids)
        conn.execute(
            f"UPDATE profile_items SET status = ? WHERE item_id IN ({placeholders})",
            [ProfileItemStatus.ARCHIVED.value, *archived_ids],
        )

    def save_persona(
        self,
        persona: PersonaDefinition,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            if persona.is_active:
                active_conn.execute(
                    "UPDATE persona_registry SET is_active = 0, updated_at = ? WHERE project_id = ? AND user_id = ?",
                    (to_iso(utcnow()), persona.project_id, persona.user_id),
                )
            active_conn.execute(
                """
                INSERT OR REPLACE INTO persona_registry (
                    persona_id, project_id, user_id, name, prompt, policy_pack,
                    description, is_enabled, is_active, source_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persona.persona_id,
                    persona.project_id,
                    persona.user_id,
                    persona.name,
                    persona.prompt,
                    persona.policy_pack,
                    persona.description,
                    1 if persona.is_enabled else 0,
                    1 if persona.is_active else 0,
                    persona.source_path,
                    to_iso(persona.created_at),
                    to_iso(persona.updated_at),
                ),
            )

    def list_personas(
        self,
        project_id: str,
        user_id: str,
        *,
        include_disabled: bool = True,
    ) -> list[PersonaDefinition]:
        clauses = ["project_id = ?", "user_id = ?"]
        params: list[object] = [project_id, user_id]
        if not include_disabled:
            clauses.append("is_enabled = 1")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM persona_registry WHERE " + " AND ".join(clauses) + " ORDER BY is_active DESC, updated_at DESC",
                params,
            ).fetchall()
        return [self._row_to_persona(row) for row in rows]

    def get_active_persona(self, project_id: str, user_id: str) -> PersonaDefinition | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM persona_registry
                WHERE project_id = ? AND user_id = ? AND is_active = 1 AND is_enabled = 1
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (project_id, user_id),
            ).fetchone()
        if not row:
            return None
        return self._row_to_persona(row)

    def activate_persona(self, project_id: str, user_id: str, persona_id: str) -> None:
        now = to_iso(utcnow())
        with self._connect() as conn:
            conn.execute(
                "UPDATE persona_registry SET is_active = 0, updated_at = ? WHERE project_id = ? AND user_id = ?",
                (now, project_id, user_id),
            )
            conn.execute(
                "UPDATE persona_registry SET is_active = 1, updated_at = ? WHERE project_id = ? AND user_id = ? AND persona_id = ?",
                (now, project_id, user_id, persona_id),
            )

    def save_relationship_state(
        self,
        state: RelationshipState,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            active_conn.execute(
                """
                INSERT OR REPLACE INTO relationship_states (
                    project_id, user_id, intimacy_level, trust_score,
                    attachment_style_guess, current_stage, recent_conflict_level,
                    boundary_flags_json, last_emotional_peak_at, notes_json,
                    behavior_state_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.project_id,
                    state.user_id,
                    state.intimacy_level,
                    state.trust_score,
                    state.attachment_style_guess,
                    state.current_stage.value,
                    state.recent_conflict_level,
                    dumps_json(state.boundary_flags),
                    to_iso(state.last_emotional_peak_at) if state.last_emotional_peak_at else None,
                    dumps_json(state.notes),
                    dumps_json(state.behavior_state.to_dict()),
                    to_iso(state.updated_at),
                ),
            )

    def load_relationship_state(self, project_id: str, user_id: str) -> RelationshipState:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM relationship_states WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            ).fetchone()
        if not row:
            return RelationshipState(project_id=project_id, user_id=user_id)
        payload = {
            "project_id": row["project_id"],
            "user_id": row["user_id"],
            "intimacy_level": row["intimacy_level"],
            "trust_score": row["trust_score"],
            "attachment_style_guess": row["attachment_style_guess"],
            "current_stage": row["current_stage"],
            "recent_conflict_level": row["recent_conflict_level"],
            "boundary_flags": loads_json(row["boundary_flags_json"], default=[]),
            "last_emotional_peak_at": row["last_emotional_peak_at"],
            "notes": loads_json(row["notes_json"], default=[]),
            "behavior_state": loads_json(row["behavior_state_json"], default={}),
            "updated_at": row["updated_at"],
        }
        return relationship_from_dict(payload)

    def save_user_manual(
        self,
        snapshot: UserManualSnapshot,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            active_conn.execute(
                """
                INSERT OR REPLACE INTO user_manual_snapshots (
                    project_id, user_id, content, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    snapshot.project_id,
                    snapshot.user_id,
                    snapshot.content,
                    to_iso(snapshot.updated_at),
                ),
            )

    def load_user_manual(self, project_id: str, user_id: str) -> UserManualSnapshot | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_manual_snapshots WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            ).fetchone()
        if not row:
            return None
        return UserManualSnapshot(
            project_id=row["project_id"],
            user_id=row["user_id"],
            content=row["content"],
            updated_at=parse_iso_datetime(row["updated_at"]),
        )

    def upsert_graph_edge(
        self,
        edge: GraphEdge,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        now = to_iso(edge.updated_at)
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            self._upsert_graph_node(
                active_conn,
                edge.project_id,
                edge.user_id,
                edge.source_node,
                "entity",
                edge.source_node,
            )
            self._upsert_graph_node(
                active_conn,
                edge.project_id,
                edge.user_id,
                edge.target_node,
                "entity",
                edge.target_node,
            )
            existing = active_conn.execute(
                """
                SELECT edge_id FROM graph_edges
                WHERE project_id = ? AND user_id = ? AND source_node = ?
                  AND edge_type = ? AND target_node = ? AND status = 'active'
                LIMIT 1
                """,
                (
                    edge.project_id,
                    edge.user_id,
                    edge.source_node,
                    edge.edge_type,
                    edge.target_node,
                ),
            ).fetchone()
            if existing:
                active_conn.execute(
                    """
                    UPDATE graph_edges
                    SET weight = ?, emotion_json = ?, metadata_json = ?, updated_at = ?
                    WHERE edge_id = ?
                    """,
                    (
                        edge.weight,
                        dumps_json(edge.emotion.to_dict()),
                        dumps_json(edge.metadata),
                        now,
                        existing["edge_id"],
                    ),
                )
            else:
                active_conn.execute(
                    """
                    INSERT INTO graph_edges (
                        edge_id, project_id, user_id, source_node, edge_type,
                        target_node, weight, emotion_json, status, metadata_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge.edge_id,
                        edge.project_id,
                        edge.user_id,
                        edge.source_node,
                        edge.edge_type,
                        edge.target_node,
                        edge.weight,
                        dumps_json(edge.emotion.to_dict()),
                        edge.status,
                        dumps_json(edge.metadata),
                        now,
                    ),
                )

    def enqueue_remote_sync_task(
        self,
        *,
        backend: str,
        operation: str,
        project_id: str,
        user_id: str,
        payload: dict,
        error_message: str = "",
        conn: sqlite3.Connection | None = None,
    ) -> str:
        now = to_iso(utcnow())
        task_id = uuid4().hex
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            active_conn.execute(
                """
                INSERT INTO remote_sync_outbox (
                    task_id, backend, operation, project_id, user_id,
                    payload_json, error_message, attempts, status, next_attempt_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    backend,
                    operation,
                    project_id,
                    user_id,
                    dumps_json(payload),
                    error_message,
                    0,
                    "pending",
                    now,
                    now,
                    now,
                ),
            )
        return task_id

    def claim_remote_sync_tasks(
        self,
        *,
        limit: int = 50,
        max_attempts: int = 5,
        reclaim_in_progress_after_seconds: int = 300,
    ) -> list[dict[str, object]]:
        now_dt = utcnow()
        now = to_iso(now_dt)
        reclaim_before = to_iso(now_dt - timedelta(seconds=max(1, reclaim_in_progress_after_seconds)))
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE remote_sync_outbox
                SET status = 'failed',
                    updated_at = ?,
                    next_attempt_at = ?,
                    error_message = CASE
                        WHEN error_message = '' THEN 'task claim timeout, reset for retry'
                        ELSE error_message
                    END
                WHERE status = 'in_progress'
                  AND updated_at < ?
                """,
                (now, now, reclaim_before),
            )
            rows = conn.execute(
                """
                SELECT *
                FROM remote_sync_outbox
                WHERE status IN ('pending', 'failed')
                  AND attempts < ?
                  AND (next_attempt_at = '' OR next_attempt_at <= ?)
                ORDER BY next_attempt_at ASC, created_at ASC
                LIMIT ?
                """,
                (max_attempts, now, limit),
            ).fetchall()
            task_ids = [str(row["task_id"]) for row in rows]
            if task_ids:
                placeholders = ", ".join("?" for _ in task_ids)
                conn.execute(
                    f"UPDATE remote_sync_outbox SET status = 'in_progress', updated_at = ? WHERE task_id IN ({placeholders})",
                    [now, *task_ids],
                )
        return [self._row_to_remote_sync_task(row) for row in rows]

    def mark_remote_sync_task_done(self, task_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE remote_sync_outbox
                SET status = 'done', updated_at = ?
                WHERE task_id = ?
                """,
                (to_iso(utcnow()), task_id),
            )

    def mark_remote_sync_task_failed(
        self,
        task_id: str,
        *,
        error_message: str,
        max_attempts: int = 5,
        retry_base_seconds: int = 5,
        retry_max_seconds: int = 300,
    ) -> None:
        now_dt = utcnow()
        now = to_iso(now_dt)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT attempts FROM remote_sync_outbox WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return
            attempts = int(row["attempts"]) + 1
            should_dead = attempts >= max(1, max_attempts)
            delay_seconds = min(
                max(0, retry_max_seconds),
                max(0, retry_base_seconds) * (2 ** max(0, attempts - 1)),
            )
            next_attempt_at = to_iso(now_dt + timedelta(seconds=delay_seconds))
            conn.execute(
                """
                UPDATE remote_sync_outbox
                SET status = ?,
                    attempts = ?,
                    error_message = ?,
                    next_attempt_at = ?,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (
                    "dead" if should_dead else "failed",
                    attempts,
                    error_message,
                    next_attempt_at,
                    now,
                    task_id,
                ),
            )

    def get_remote_sync_outbox_summary(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM remote_sync_outbox
                GROUP BY status
                """
            ).fetchall()
        summary = {str(row["status"]): int(row["count"]) for row in rows}
        return {
            "pending": summary.get("pending", 0),
            "failed": summary.get("failed", 0),
            "in_progress": summary.get("in_progress", 0),
            "done": summary.get("done", 0),
            "dead": summary.get("dead", 0),
        }

    def _row_to_remote_sync_task(self, row: sqlite3.Row) -> dict[str, object]:
        return {
            "task_id": row["task_id"],
            "backend": row["backend"],
            "operation": row["operation"],
            "project_id": row["project_id"],
            "user_id": row["user_id"],
            "payload": loads_json(row["payload_json"], default={}),
            "error_message": row["error_message"],
            "attempts": int(row["attempts"]),
            "status": row["status"],
            "next_attempt_at": row["next_attempt_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_related_nodes(
        self,
        project_id: str,
        user_id: str,
        entities: Sequence[str],
        *,
        limit: int = 20,
    ) -> dict[str, float]:
        if not entities:
            return {}
        normalized = {normalize_text(entity) for entity in entities if entity.strip()}
        if not normalized:
            return {}

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_node, target_node, weight
                FROM graph_edges
                WHERE project_id = ? AND user_id = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 500
                """,
                (project_id, user_id),
            ).fetchall()

        scores: dict[str, float] = {}
        for row in rows:
            source = row["source_node"]
            target = row["target_node"]
            weight = float(row["weight"])
            if normalize_text(source) in normalized:
                scores[target] = max(scores.get(target, 0.0), weight)
            if normalize_text(target) in normalized:
                scores[source] = max(scores.get(source, 0.0), weight)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return dict(ordered)

    def clear_user_data(self, project_id: str, user_id: str) -> None:
        tables = [
            "session_turns",
            "memory_traces",
            "semantic_facts",
            "relationship_states",
            "user_profiles",
            "profile_items",
            "persona_registry",
            "user_manual_snapshots",
            "graph_nodes",
            "graph_edges",
        ]
        with self._connect() as conn:
            for table in tables:
                conn.execute(
                    f"DELETE FROM {table} WHERE project_id = ? AND user_id = ?",
                    (project_id, user_id),
                )

    def stats(self, project_id: str, user_id: str) -> dict[str, int]:
        with self._connect() as conn:
            episodic = conn.execute(
                """
                SELECT COUNT(*) AS count FROM memory_traces
                WHERE project_id = ? AND user_id = ? AND kind = ? AND status = 'active'
                """,
                (project_id, user_id, MemoryKind.EPISODIC.value),
            ).fetchone()["count"]
            semantic = conn.execute(
                """
                SELECT COUNT(*) AS count FROM semantic_facts
                WHERE project_id = ? AND user_id = ? AND status = 'active'
                """,
                (project_id, user_id),
            ).fetchone()["count"]
            turns = conn.execute(
                """
                SELECT COUNT(*) AS count FROM session_turns
                WHERE project_id = ? AND user_id = ?
                """,
                (project_id, user_id),
            ).fetchone()["count"]
            profile_items = conn.execute(
                """
                SELECT COUNT(*) AS count FROM profile_items
                WHERE project_id = ? AND user_id = ? AND status = ?
                """,
                (project_id, user_id, ProfileItemStatus.ACTIVE.value),
            ).fetchone()["count"]
            personas = conn.execute(
                """
                SELECT COUNT(*) AS count FROM persona_registry
                WHERE project_id = ? AND user_id = ?
                """,
                (project_id, user_id),
            ).fetchone()["count"]
        return {
            "episodic": episodic,
            "semantic": semantic,
            "turns": turns,
            "profile_items": profile_items,
            "personas": personas,
        }

    # ------------------------------------------------------------------
    # Memory archiving & merging
    # ------------------------------------------------------------------

    def archive_stale_memory_traces(
        self,
        project_id: str,
        user_id: str,
        *,
        min_age_days: int = 90,
        max_salience: float = 0.20,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Archive traces older than *min_age_days* whose salience <= *max_salience*."""
        cutoff = to_iso(utcnow() - timedelta(days=min_age_days))
        now = to_iso(utcnow())
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            cur = active_conn.execute(
                """
                UPDATE memory_traces
                SET status = 'archived', updated_at = ?
                WHERE project_id = ? AND user_id = ?
                  AND status = 'active'
                  AND salience <= ?
                  AND updated_at < ?
                """,
                (now, project_id, user_id, max_salience, cutoff),
            )
            archived = cur.rowcount
        if archived:
            logger.info(
                "archive_stale_traces project={} user={} archived={} min_age_days={} max_salience={:.2f}",
                project_id, user_id, archived, min_age_days, max_salience,
            )
        return archived

    def merge_trace_pair(
        self,
        keep: MemoryTrace,
        discard: MemoryTrace,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Merge *discard* into *keep*: combine entities/tags, sum access_count, archive discard."""
        merged_entities = list(dict.fromkeys(keep.entities + discard.entities))
        merged_tags = list(dict.fromkeys(keep.tags + discard.tags))
        merged_source_turn_ids = list(dict.fromkeys(keep.source_turn_ids + discard.source_turn_ids))
        new_salience = max(keep.salience, discard.salience)
        new_access_count = keep.access_count + discard.access_count
        now = to_iso(utcnow())
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            active_conn.execute(
                """
                UPDATE memory_traces
                SET entities_json = ?, tags_json = ?, source_turn_ids_json = ?,
                    salience = ?, access_count = ?, updated_at = ?
                WHERE trace_id = ?
                """,
                (
                    dumps_json(merged_entities),
                    dumps_json(merged_tags),
                    dumps_json(merged_source_turn_ids),
                    new_salience,
                    new_access_count,
                    now,
                    keep.trace_id,
                ),
            )
            active_conn.execute(
                """
                UPDATE memory_traces
                SET status = 'merged', updated_at = ?,
                    metadata_json = json_set(COALESCE(metadata_json, '{}'), '$.merged_into', ?)
                WHERE trace_id = ?
                """,
                (now, keep.trace_id, discard.trace_id),
            )
        logger.info(
            "merge_trace keep={} discard={} salience={:.2f} access_count={}",
            keep.trace_id, discard.trace_id, new_salience, new_access_count,
        )

    def archive_redundant_episodic_traces(
        self,
        project_id: str,
        user_id: str,
        *,
        min_age_days: int = 60,
        semantic_confidence_min: float = 0.8,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Archive episodic traces whose entities are covered by high-confidence semantic facts."""
        cutoff = to_iso(utcnow() - timedelta(days=min_age_days))
        context = nullcontext(conn) if conn is not None else self._connect()
        with context as active_conn:
            traces = active_conn.execute(
                """
                SELECT trace_id, entities_json FROM memory_traces
                WHERE project_id = ? AND user_id = ? AND status = 'active'
                  AND kind = 'episodic' AND updated_at < ?
                """,
                (project_id, user_id, cutoff),
            ).fetchall()
            if not traces:
                return 0

            facts = active_conn.execute(
                """
                SELECT subject, object_text FROM semantic_facts
                WHERE project_id = ? AND user_id = ? AND status = 'active'
                  AND confidence >= ?
                """,
                (project_id, user_id, semantic_confidence_min),
            ).fetchall()
            if not facts:
                return 0

            from arm_memory.utils import normalize_text
            fact_terms: set[str] = set()
            for f in facts:
                fact_terms.add(normalize_text(f["subject"]))
                fact_terms.add(normalize_text(f["object_text"]))

            now = to_iso(utcnow())
            archived = 0
            for row in traces:
                entities = loads_json(row["entities_json"], default=[])
                if not entities:
                    continue
                entity_set = {normalize_text(e) for e in entities}
                if entity_set & fact_terms:
                    active_conn.execute(
                        """
                        UPDATE memory_traces
                        SET status = 'archived', updated_at = ?,
                            metadata_json = json_set(COALESCE(metadata_json, '{}'), '$.archived_reason', 'semantic_covered')
                        WHERE trace_id = ?
                        """,
                        (now, row["trace_id"]),
                    )
                    archived += 1
            if archived:
                logger.info(
                    "archive_redundant_episodic project={} user={} archived={}",
                    project_id, user_id, archived,
                )
            return archived

    def _row_to_profile_item(self, row: sqlite3.Row) -> UserProfileItem:
        return profile_item_from_dict(
            {
                "item_id": row["item_id"],
                "project_id": row["project_id"],
                "user_id": row["user_id"],
                "facet_type": row["facet_type"],
                "value": row["value"],
                "confidence": row["confidence"],
                "source": row["source"],
                "source_trace_id": row["source_trace_id"],
                "first_seen_at": row["first_seen_at"],
                "last_seen_at": row["last_seen_at"],
                "evidence_count": row["evidence_count"],
                "status": row["status"],
                "metadata": loads_json(row["metadata_json"], default={}),
            }
        )

    def _row_to_persona(self, row: sqlite3.Row) -> PersonaDefinition:
        return persona_from_dict(
            {
                "persona_id": row["persona_id"],
                "project_id": row["project_id"],
                "user_id": row["user_id"],
                "name": row["name"],
                "prompt": row["prompt"],
                "policy_pack": row["policy_pack"],
                "description": row["description"],
                "is_enabled": bool(row["is_enabled"]),
                "is_active": bool(row["is_active"]),
                "source_path": row["source_path"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    def _upsert_graph_node(
        self,
        conn: sqlite3.Connection,
        project_id: str,
        user_id: str,
        node_key: str,
        node_type: str,
        label: str,
    ) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO graph_nodes (
                project_id, user_id, node_key, node_type, label, metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                user_id,
                node_key,
                node_type,
                label,
                "{}",
                to_iso(utcnow()),
            ),
        )

    def _row_to_turn(self, row: sqlite3.Row) -> ConversationTurn:
        return ConversationTurn(
            turn_id=row["turn_id"],
            project_id=row["project_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=parse_iso_datetime(row["created_at"]),
            metadata=loads_json(row["metadata_json"], default={}),
        )

    def _row_to_trace(self, row: sqlite3.Row) -> MemoryTrace:
        return trace_from_dict(
            {
                "trace_id": row["trace_id"],
                "project_id": row["project_id"],
                "user_id": row["user_id"],
                "kind": row["kind"],
                "summary": row["summary"],
                "raw_text": row["raw_text"],
                "status": row["status"],
                "salience": row["salience"],
                "access_count": row["access_count"],
                "source_turn_ids": loads_json(row["source_turn_ids_json"], default=[]),
                "tags": loads_json(row["tags_json"], default=[]),
                "entities": loads_json(row["entities_json"], default=[]),
                "vector": loads_json(row["vector_json"], default=[]),
                "emotion": loads_json(row["emotion_json"], default={}),
                "sensitivity": row["sensitivity"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "happened_at": row["happened_at"],
                "last_accessed_at": row["last_accessed_at"],
                "metadata": loads_json(row["metadata_json"], default={}),
            }
        )

    def _row_to_fact(self, row: sqlite3.Row) -> SemanticFact:
        return SemanticFact(
            fact_id=row["fact_id"],
            project_id=row["project_id"],
            user_id=row["user_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object_text=row["object_text"],
            summary=row["summary"],
            confidence=float(row["confidence"]),
            salience=float(row["salience"]),
            source_trace_id=row["source_trace_id"],
            source_turn_ids=loads_json(row["source_turn_ids_json"], default=[]),
            entities=loads_json(row["entities_json"], default=[]),
            tags=loads_json(row["tags_json"], default=[]),
            emotion=EmotionVector.from_dict(loads_json(row["emotion_json"], default={})),
            sensitivity=SensitivityLevel(row["sensitivity"]),
            status=row["status"],
            valid_from=parse_iso_datetime(row["valid_from"]),
            valid_to=parse_iso_datetime(row["valid_to"]) if row["valid_to"] else None,
            created_at=parse_iso_datetime(row["created_at"]),
            updated_at=parse_iso_datetime(row["updated_at"]),
            metadata=loads_json(row["metadata_json"], default={}),
        )
