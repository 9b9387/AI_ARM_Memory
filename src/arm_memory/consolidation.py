from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from arm_memory.config import ARMConfig
from arm_memory.domain.models import (
    BoundaryStage,
    ConsolidationResult,
    ConversationTurn,
    EmotionLabel,
    EmotionVector,
    GraphEdge,
    MemoryKind,
    MemoryOperation,
    MemoryTrace,
    ProfileFacetType,
    ProcedureRule,
    RelationshipStage,
    RelationshipState,
    SemanticFact,
    SensitivityLevel,
    UserManualSnapshot,
    UserProfile,
    UserProfileItem,
    relationship_stage_from_value,
)
from arm_memory.stores.neo4j_store import Neo4jGraphStore
from arm_memory.stores.qdrant_store import QdrantVectorStore
from arm_memory.stores.sqlite_store import SQLiteMemoryStore
from arm_memory.utils import clamp, normalize_text, utcnow
from arm_memory.vectorizer import EmbeddingProvider

logger = logging.getLogger(__name__)


class _ConsolidationBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ConsolidationEmotionPayload(_ConsolidationBaseModel):
    label: str = EmotionLabel.NEUTRAL.value
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    intensity: float = 0.0
    confidence: float = 0.0


class ConsolidationMemoryPayload(_ConsolidationBaseModel):
    summary: str = ""
    raw_text: str = ""
    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    salience: float = 0.6
    sensitivity: str = SensitivityLevel.PRIVATE.value
    emotion: ConsolidationEmotionPayload = Field(default_factory=ConsolidationEmotionPayload)


class ConsolidationSemanticPayload(_ConsolidationBaseModel):
    subject: str = "user"
    predicate: str = "notes"
    object: str = ""
    summary: str = ""
    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    salience: float = 0.65
    confidence: float = 0.75
    sensitivity: str = SensitivityLevel.PRIVATE.value
    operation: str = MemoryOperation.ADD.value
    emotion: ConsolidationEmotionPayload = Field(default_factory=ConsolidationEmotionPayload)


class ConsolidationProfilePayload(_ConsolidationBaseModel):
    display_name: str = ""
    identity_notes: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    vulnerabilities: list[str] = Field(default_factory=list)
    support_preferences: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    important_people: list[str] = Field(default_factory=list)
    important_places: list[str] = Field(default_factory=list)
    manual_notes: list[str] = Field(default_factory=list)
    personality_traits: list[str] = Field(default_factory=list)
    hobbies: list[str] = Field(default_factory=list)
    habits: list[str] = Field(default_factory=list)
    communication_style: list[str] = Field(default_factory=list)
    attachment_style: list[str] = Field(default_factory=list)
    life_routines: list[str] = Field(default_factory=list)


class ConsolidationProfileItemPayload(_ConsolidationBaseModel):
    facet_type: str = ProfileFacetType.MANUAL_NOTES.value
    value: str = ""
    confidence: float = 0.7
    source: str = "llm_profile"
    status: str = "active"
    source_trace_id: str | None = None


class ConsolidationBehaviorStatePayload(_ConsolidationBaseModel):
    boundary_stage: str = BoundaryStage.CALM.value
    violation_count: int = 0
    cooldown_until: str | None = None
    last_violation_at: str | None = None
    last_repair_at: str | None = None
    last_boundary_reason: str = ""
    recent_flags: list[str] = Field(default_factory=list)


class ConsolidationRelationshipPayload(_ConsolidationBaseModel):
    intimacy_delta: float = 0.0
    trust_delta: float = 0.0
    recent_conflict_level: float = 0.0
    current_stage: str = RelationshipStage.INITIAL_CONTACT.value
    boundary_flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    behavior_state: ConsolidationBehaviorStatePayload = Field(
        default_factory=ConsolidationBehaviorStatePayload
    )


class SleepCycleConsolidator:
    """仅负责将外部提供的结构化抽取结果写入存储，不执行对话→抽取。"""

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
        if self.qdrant_store:
            self.qdrant_store.ensure_collection()

    def apply_extraction(
        self,
        *,
        project_id: str,
        user_id: str,
        extraction: dict[str, Any],
        turn_ids: list[str] | None = None,
        persona_prompt: str = "",
    ) -> ConsolidationResult:
        """接收外部已生成的结构化抽取结果，写入记忆/画像/关系并更新用户说明书。"""
        t0 = time.monotonic()
        extraction = self._normalize_extraction(extraction)
        relationship_state = self.sqlite_store.load_relationship_state(project_id, user_id)
        profile = self.sqlite_store.load_profile(project_id, user_id)
        source_turns: list[ConversationTurn] = []  # 外部抽取无本地 turns，留空

        result = ConsolidationResult()
        qdrant_traces: list[MemoryTrace] = []
        neo4j_facts: list[SemanticFact] = []
        neo4j_edges: list[GraphEdge] = []
        with self.sqlite_store.transaction() as conn:
            for item in extraction.get("episodic", []):
                trace = self._trace_from_extracted_item(
                    item,
                    project_id=project_id,
                    user_id=user_id,
                    kind=MemoryKind.EPISODIC,
                    source_turns=source_turns,
                )
                op = self.sqlite_store.add_memory_trace(trace, conn=conn)
                result.bump(op)
                if op != MemoryOperation.NOOP:
                    qdrant_traces.append(trace)
                result.episodic_count += 1

            for item in extraction.get("semantic", []):
                trace = self._trace_from_extracted_item(
                    item,
                    project_id=project_id,
                    user_id=user_id,
                    kind=MemoryKind.SEMANTIC,
                    source_turns=source_turns,
                )
                trace_op = self.sqlite_store.add_memory_trace(trace, conn=conn)
                if trace_op != MemoryOperation.NOOP:
                    qdrant_traces.append(trace)

                fact = self._fact_from_extracted_item(
                    item,
                    project_id=project_id,
                    user_id=user_id,
                    source_trace_id=trace.trace_id,
                    source_turns=source_turns,
                )
                item["_source_trace_id"] = trace.trace_id
                fact_op = self.sqlite_store.apply_semantic_fact(
                    fact,
                    operation=self._operation_from_value(item.get("operation")),
                    conn=conn,
                )
                result.bump(fact_op)
                edge = GraphEdge(
                    project_id=project_id,
                    user_id=user_id,
                    source_node=self._entity_key(fact.subject),
                    edge_type=fact.predicate,
                    target_node=self._entity_key(fact.object_text),
                    weight=clamp(fact.salience, 0.2, 1.0),
                    emotion=fact.emotion,
                    metadata={"fact_id": fact.fact_id, "summary": fact.summary},
                )
                self.sqlite_store.upsert_graph_edge(edge, conn=conn)
                neo4j_facts.append(fact)
                neo4j_edges.append(edge)
                result.semantic_count += 1

            profile_items = self._apply_profile_updates(
                profile,
                extraction.get("profile") or {},
                extraction.get("semantic") or [],
                extraction.get("profile_items") or [],
                project_id=project_id,
                user_id=user_id,
            )
            self.sqlite_store.save_profile(profile, conn=conn)
            for item in profile_items:
                self.sqlite_store.save_profile_item(
                    item,
                    conn=conn,
                    facet_limit=self.config.profile_item_limit_per_facet,
                )

            self._apply_relationship_updates(relationship_state, extraction.get("relationship") or {})
            self.sqlite_store.save_relationship_state(relationship_state, conn=conn)

            snapshot = self.build_user_manual(
                project_id=project_id,
                user_id=user_id,
                persona_prompt=persona_prompt,
                conn=conn,
                profile_override=profile,
                relationship_override=relationship_state,
            )
            self.sqlite_store.save_user_manual(snapshot, conn=conn)
            if turn_ids:
                self.sqlite_store.mark_turns_consolidated(turn_ids, conn=conn)

        result.user_manual_updated = True
        result.triggered_procedures = list(extraction.get("procedures") or [])
        if extraction.get("safety_flags"):
            result.notes.append("Safety flags: " + ", ".join(extraction["safety_flags"]))

        for trace in qdrant_traces:
            if not self.qdrant_store:
                break
            try:
                self.qdrant_store.upsert_trace(trace)
            except Exception as exc:
                self.sqlite_store.enqueue_remote_sync_task(
                    backend="qdrant",
                    operation="upsert_trace",
                    project_id=project_id,
                    user_id=user_id,
                    payload=trace.to_dict(),
                    error_message=str(exc),
                )

        if self.neo4j_store:
            for fact in neo4j_facts:
                try:
                    self.neo4j_store.sync_fact(fact)
                except Exception as exc:
                    self.sqlite_store.enqueue_remote_sync_task(
                        backend="neo4j",
                        operation="sync_fact",
                        project_id=project_id,
                        user_id=user_id,
                        payload=fact.to_dict(),
                        error_message=str(exc),
                    )
            for edge in neo4j_edges:
                try:
                    self.neo4j_store.sync_edge(edge)
                except Exception as exc:
                    self.sqlite_store.enqueue_remote_sync_task(
                        backend="neo4j",
                        operation="sync_edge",
                        project_id=project_id,
                        user_id=user_id,
                        payload=edge.to_dict(),
                        error_message=str(exc),
                    )
            try:
                self.neo4j_store.sync_relationship_state(
                    project_id=project_id,
                    user_id=user_id,
                    companion_node=self.config.companion_node_name,
                    state=relationship_state,
                )
            except Exception as exc:
                self.sqlite_store.enqueue_remote_sync_task(
                    backend="neo4j",
                    operation="sync_relationship_state",
                    project_id=project_id,
                    user_id=user_id,
                    payload=relationship_state.to_dict(),
                    error_message=str(exc),
                )

        # ------------------------------------------------------------------
        # Post-consolidation: archive stale traces & merge similar ones
        # ------------------------------------------------------------------
        try:
            archived = self.sqlite_store.archive_stale_memory_traces(
                project_id,
                user_id,
                min_age_days=self.config.memory_archive_min_age_days,
                max_salience=self.config.memory_archive_min_salience,
            )
            result.notes.append(f"archived_traces={archived}")
        except Exception:
            logger.warning("archive_stale_traces failed", exc_info=True)

        try:
            pairs = self.sqlite_store.find_similar_trace_pairs(
                project_id,
                user_id,
                similarity_threshold=self.config.memory_merge_similarity_threshold,
                limit=self.config.retrieval_candidate_limit,
            )
            merged_ids: set[str] = set()
            merge_count = 0
            for keep, discard, sim in pairs:
                if keep.trace_id in merged_ids or discard.trace_id in merged_ids:
                    continue
                self.sqlite_store.merge_trace_pair(keep, discard)
                merged_ids.add(discard.trace_id)
                merge_count += 1
            if merge_count:
                result.notes.append(f"merged_traces={merge_count}")
        except Exception:
            logger.warning("merge_similar_traces failed", exc_info=True)

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "apply_extraction_complete project=%s user=%s turn_ids=%d "
            "episodic=%d semantic=%d added=%d updated=%d elapsed_ms=%.1f notes=%s",
            project_id,
            user_id,
            len(turn_ids) if turn_ids else 0,
            result.episodic_count,
            result.semantic_count,
            result.operations.get(MemoryOperation.ADD, 0),
            result.operations.get(MemoryOperation.UPDATE, 0),
            elapsed_ms,
            result.notes,
        )
        return result

    def build_user_manual(
        self,
        *,
        project_id: str,
        user_id: str,
        persona_prompt: str,
        conn=None,
        profile_override: UserProfile | None = None,
        profile_items_override: list[UserProfileItem] | None = None,
        relationship_override: RelationshipState | None = None,
    ) -> UserManualSnapshot:
        profile = profile_override or self.sqlite_store.load_profile(project_id, user_id)
        if profile_items_override is not None:
            profile_items = profile_items_override
        else:
            profile_items = self.sqlite_store.list_profile_items(
                project_id, user_id, 
                max_age_days=self.config.profile_item_stale_days,
                conn=conn
            )
        relationship = relationship_override or self.sqlite_store.load_relationship_state(project_id, user_id)
        facts = self.sqlite_store.list_semantic_facts(
            project_id,
            user_id,
            limit=self.config.user_manual_memory_limit,
            conn=conn,
        )
        episodes = self.sqlite_store.list_active_memory_traces(
            project_id,
            user_id,
            kinds=[MemoryKind.EPISODIC],
            limit=5,
            conn=conn,
        )

        lines: list[str] = []
        lines.append("## 用户画像")
        if profile.display_name:
            lines.append(f"- 称呼或身份：{profile.display_name}")
            
        from collections import defaultdict
        
        grouped_items: dict[str, list[str]] = defaultdict(list)
        for item in profile_items:
            grouped_items[item.facet_type.value].append(item.value)
            
        def _get_facet_lines(facet_type: str, legacy_list: list[str]) -> list[str]:
            items = grouped_items.get(facet_type, [])
            # Merge and deduplicate, preferring profile_items
            seen = set(items)
            for old_item in legacy_list:
                if old_item not in seen:
                    items.append(old_item)
                    seen.add(old_item)
            return items

        identities = _get_facet_lines("identity_notes", profile.identity_notes)
        if identities:
            lines.append("- 身份信息：" + "；".join(identities[:4]))
            
        prefs = _get_facet_lines("preferences", profile.preferences)
        if prefs:
            lines.append("- 偏好：" + "；".join(prefs[:6]))
            
        dislikes = _get_facet_lines("dislikes", profile.dislikes)
        if dislikes:
            lines.append("- 反感或规避：" + "；".join(dislikes[:6]))
            
        boundaries = _get_facet_lines("boundaries", profile.boundaries)
        if boundaries:
            lines.append("- 明确边界：" + "；".join(boundaries[:6]))
            
        vulns = _get_facet_lines("vulnerabilities", profile.vulnerabilities)
        if vulns:
            lines.append("- 脆弱点：" + "；".join(vulns[:4]))
            
        supports = _get_facet_lines("support_preferences", profile.support_preferences)
        if supports:
            lines.append("- 支持方式：" + "；".join(supports[:4]))
            
        goals = _get_facet_lines("goals", profile.goals)
        if goals:
            lines.append("- 当前目标：" + "；".join(goals[:4]))
            
        traits = _get_facet_lines("personality_traits", profile.personality_traits)
        if traits:
            lines.append("- 性格特征：" + "；".join(traits[:6]))
            
        hobbies = _get_facet_lines("hobbies", profile.hobbies)
        if hobbies:
            lines.append("- 爱好：" + "；".join(hobbies[:6]))
            
        habits = _get_facet_lines("habits", profile.habits)
        if habits:
            lines.append("- 习惯：" + "；".join(habits[:6]))
            
        styles = _get_facet_lines("communication_style", profile.communication_style)
        if styles:
            lines.append("- 沟通风格：" + "；".join(styles[:4]))
            
        attach = _get_facet_lines("attachment_style", profile.attachment_style)
        if attach:
            lines.append("- 依恋风格：" + "；".join(attach[:4]))
            
        routines = _get_facet_lines("life_routines", profile.life_routines)
        if routines:
            lines.append("- 生活节律：" + "；".join(routines[:4]))
            
        manual = _get_facet_lines("manual_notes", profile.manual_notes)
        if manual:
            lines.append("- 人工备注：" + "；".join(manual[:6]))

        lines.append("\n## 关系状态")
        lines.append(f"- 阶段：{relationship.current_stage.value}")
        lines.append(f"- 亲密度：{relationship.intimacy_level:.1f}/100")
        lines.append(f"- 信任度：{relationship.trust_score:.1f}/100")
        lines.append(f"- 近期冲突：{relationship.recent_conflict_level:.1f}/100")
        lines.append(f"- 边界阶段：{relationship.behavior_state.boundary_stage.value}")
        if relationship.boundary_flags:
            lines.append("- 风险边界：" + "；".join(relationship.boundary_flags[:6]))
        if relationship.notes:
            lines.append("- 关系注记：" + "；".join(relationship.notes[:4]))
        if relationship.behavior_state.violation_count:
            lines.append(f"- 越界累计：{relationship.behavior_state.violation_count}")

        if facts:
            lines.append("\n## 长期事实")
            for fact in facts[: self.config.user_manual_memory_limit]:
                lines.append(f"- {fact.summary or fact.subject + ' ' + fact.predicate + ' ' + fact.object_text}")

        if episodes:
            lines.append("\n## 近期情景")
            for trace in episodes[:5]:
                lines.append(f"- {trace.summary}")

        lines.append("\n## 使用准则")
        lines.append("- 优先遵守人格基线和关系边界，再结合用户长期事实与近期情景。")
        lines.append("- 遇到高风险、强依赖、操控性请求时，保持温柔但明确的边界。")

        return UserManualSnapshot(
            project_id=project_id,
            user_id=user_id,
            content="\n".join(lines).strip(),
        )

    def _normalize_extraction(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "episodic": list(payload.get("episodic") or []),
            "semantic": list(payload.get("semantic") or []),
            "profile": dict(payload.get("profile") or {}),
            "profile_items": list(payload.get("profile_items") or []),
            "relationship": dict(payload.get("relationship") or {}),
            "procedures": list(payload.get("procedures") or []),
            "safety_flags": list(payload.get("safety_flags") or []),
        }

    def _trace_from_extracted_item(
        self,
        item: dict[str, Any],
        *,
        project_id: str,
        user_id: str,
        kind: MemoryKind,
        source_turns: list[ConversationTurn],
    ) -> MemoryTrace:
        now = utcnow()
        summary = item.get("summary") or item.get("object") or item.get("raw_text") or ""
        raw_text = item.get("raw_text") or summary
        entities = [str(entity).strip() for entity in (item.get("entities") or []) if str(entity).strip()]
        if not entities:
            entities = self.vectorizer.tokenize(summary)[:6]
        text_for_vector = " ".join([summary, raw_text, *entities])
        return MemoryTrace(
            project_id=project_id,
            user_id=user_id,
            kind=kind,
            summary=summary,
            raw_text=raw_text,
            salience=clamp(float(item.get("salience", 0.6)), 0.0, 1.0),
            source_turn_ids=[turn.turn_id for turn in source_turns],
            tags=[str(tag) for tag in (item.get("tags") or [])],
            entities=entities,
            vector=self.vectorizer.embed(text_for_vector),
            emotion=EmotionVector.from_dict(item.get("emotion")),
            sensitivity=self._sensitivity_from_value(item.get("sensitivity")),
            created_at=now,
            updated_at=now,
            happened_at=now,
            last_accessed_at=now,
            metadata={"operation": item.get("operation", "ADD")},
        )

    def _fact_from_extracted_item(
        self,
        item: dict[str, Any],
        *,
        project_id: str,
        user_id: str,
        source_trace_id: str,
        source_turns: list[ConversationTurn],
    ) -> SemanticFact:
        subject = str(item.get("subject") or "user").strip()
        predicate = str(item.get("predicate") or "notes").strip()
        object_text = str(item.get("object") or item.get("object_text") or item.get("summary") or "").strip()
        now = utcnow()
        return SemanticFact(
            project_id=project_id,
            user_id=user_id,
            subject=subject,
            predicate=predicate,
            object_text=object_text,
            summary=str(item.get("summary") or f"{subject} {predicate} {object_text}").strip(),
            confidence=clamp(float(item.get("confidence", 0.75)), 0.0, 1.0),
            salience=clamp(float(item.get("salience", 0.65)), 0.0, 1.0),
            source_trace_id=source_trace_id,
            source_turn_ids=[turn.turn_id for turn in source_turns],
            entities=[str(entity) for entity in (item.get("entities") or [])],
            tags=[str(tag) for tag in (item.get("tags") or [])],
            emotion=EmotionVector.from_dict(item.get("emotion")),
            sensitivity=self._sensitivity_from_value(item.get("sensitivity")),
            valid_from=now,
            created_at=now,
            updated_at=now,
            metadata={"operation": item.get("operation", "ADD")},
        )

    def _apply_profile_updates(
        self,
        profile: UserProfile,
        profile_update: dict[str, Any],
        semantic_items: list[dict[str, Any]],
        extracted_profile_items: list[dict[str, Any]],
        *,
        project_id: str,
        user_id: str,
    ) -> list[UserProfileItem]:
        items: list[UserProfileItem] = []
        if profile_update.get("display_name") and not profile.display_name:
            profile.display_name = str(profile_update["display_name"]).strip()

        profile.merge_lists(
            identity_notes=[str(item) for item in profile_update.get("identity_notes", [])],
            preferences=[str(item) for item in profile_update.get("preferences", [])],
            dislikes=[str(item) for item in profile_update.get("dislikes", [])],
            boundaries=[str(item) for item in profile_update.get("boundaries", [])],
            vulnerabilities=[str(item) for item in profile_update.get("vulnerabilities", [])],
            support_preferences=[str(item) for item in profile_update.get("support_preferences", [])],
            goals=[str(item) for item in profile_update.get("goals", [])],
            important_people=[str(item) for item in profile_update.get("important_people", [])],
            important_places=[str(item) for item in profile_update.get("important_places", [])],
            manual_notes=[str(item) for item in profile_update.get("manual_notes", [])],
            personality_traits=[str(item) for item in profile_update.get("personality_traits", [])],
            hobbies=[str(item) for item in profile_update.get("hobbies", [])],
            habits=[str(item) for item in profile_update.get("habits", [])],
            communication_style=[str(item) for item in profile_update.get("communication_style", [])],
            attachment_style=[str(item) for item in profile_update.get("attachment_style", [])],
            life_routines=[str(item) for item in profile_update.get("life_routines", [])],
        )

        for facet_type, values in {
            ProfileFacetType.IDENTITY_NOTES: profile_update.get("identity_notes", []),
            ProfileFacetType.PREFERENCES: profile_update.get("preferences", []),
            ProfileFacetType.DISLIKES: profile_update.get("dislikes", []),
            ProfileFacetType.BOUNDARIES: profile_update.get("boundaries", []),
            ProfileFacetType.VULNERABILITIES: profile_update.get("vulnerabilities", []),
            ProfileFacetType.SUPPORT_PREFERENCES: profile_update.get("support_preferences", []),
            ProfileFacetType.GOALS: profile_update.get("goals", []),
            ProfileFacetType.IMPORTANT_PEOPLE: profile_update.get("important_people", []),
            ProfileFacetType.IMPORTANT_PLACES: profile_update.get("important_places", []),
            ProfileFacetType.MANUAL_NOTES: profile_update.get("manual_notes", []),
            ProfileFacetType.PERSONALITY_TRAITS: profile_update.get("personality_traits", []),
            ProfileFacetType.HOBBIES: profile_update.get("hobbies", []),
            ProfileFacetType.HABITS: profile_update.get("habits", []),
            ProfileFacetType.COMMUNICATION_STYLE: profile_update.get("communication_style", []),
            ProfileFacetType.ATTACHMENT_STYLE: profile_update.get("attachment_style", []),
            ProfileFacetType.LIFE_ROUTINES: profile_update.get("life_routines", []),
        }.items():
            items.extend(
                self._profile_items_from_values(
                    project_id=project_id,
                    user_id=user_id,
                    facet_type=facet_type,
                    values=values,
                    source="consolidation_profile",
                )
            )

        for payload in extracted_profile_items:
            profile_item = self._profile_item_from_payload(
                project_id=project_id,
                user_id=user_id,
                payload=payload,
            )
            if profile_item is not None:
                items.append(profile_item)

        for item in semantic_items:
            predicate = str(item.get("predicate", "")).lower()
            obj = str(item.get("object") or item.get("object_text") or "").strip()
            if not obj:
                continue
            if predicate == "prefers":
                profile.merge_lists(preferences=[obj])
                items.extend(self._profile_items_from_values(project_id=project_id, user_id=user_id, facet_type=ProfileFacetType.PREFERENCES, values=[obj], source="consolidation_semantic", confidence=float(item.get("confidence", 0.78)), source_trace_id=item.get("_source_trace_id")))
            elif predicate == "avoids":
                profile.merge_lists(dislikes=[obj])
                items.extend(self._profile_items_from_values(project_id=project_id, user_id=user_id, facet_type=ProfileFacetType.DISLIKES, values=[obj], source="consolidation_semantic", confidence=float(item.get("confidence", 0.8)), source_trace_id=item.get("_source_trace_id")))
            elif predicate == "identity":
                if not profile.display_name:
                    profile.display_name = obj
                profile.merge_lists(identity_notes=[obj])
                items.extend(self._profile_items_from_values(project_id=project_id, user_id=user_id, facet_type=ProfileFacetType.IDENTITY_NOTES, values=[obj], source="consolidation_semantic", confidence=float(item.get("confidence", 0.85)), source_trace_id=item.get("_source_trace_id")))
            elif predicate == "boundary":
                profile.merge_lists(boundaries=[obj])
                items.extend(self._profile_items_from_values(project_id=project_id, user_id=user_id, facet_type=ProfileFacetType.BOUNDARIES, values=[obj], source="consolidation_semantic", confidence=float(item.get("confidence", 0.9)), source_trace_id=item.get("_source_trace_id")))
            elif predicate == "hobby":
                profile.merge_lists(hobbies=[obj])
                items.extend(self._profile_items_from_values(project_id=project_id, user_id=user_id, facet_type=ProfileFacetType.HOBBIES, values=[obj], source="consolidation_semantic", confidence=float(item.get("confidence", 0.8)), source_trace_id=item.get("_source_trace_id")))
            elif predicate == "habit":
                profile.merge_lists(habits=[obj])
                items.extend(self._profile_items_from_values(project_id=project_id, user_id=user_id, facet_type=ProfileFacetType.HABITS, values=[obj], source="consolidation_semantic", confidence=float(item.get("confidence", 0.76)), source_trace_id=item.get("_source_trace_id")))
            elif predicate == "personality_trait":
                profile.merge_lists(personality_traits=[obj])
                items.extend(self._profile_items_from_values(project_id=project_id, user_id=user_id, facet_type=ProfileFacetType.PERSONALITY_TRAITS, values=[obj], source="consolidation_semantic", confidence=float(item.get("confidence", 0.74)), source_trace_id=item.get("_source_trace_id")))
            elif predicate == "communication_style":
                profile.merge_lists(communication_style=[obj])
                items.extend(self._profile_items_from_values(project_id=project_id, user_id=user_id, facet_type=ProfileFacetType.COMMUNICATION_STYLE, values=[obj], source="consolidation_semantic", confidence=float(item.get("confidence", 0.76)), source_trace_id=item.get("_source_trace_id")))
            elif predicate == "attachment_style":
                profile.merge_lists(attachment_style=[obj])
                items.extend(self._profile_items_from_values(project_id=project_id, user_id=user_id, facet_type=ProfileFacetType.ATTACHMENT_STYLE, values=[obj], source="consolidation_semantic", confidence=float(item.get("confidence", 0.72)), source_trace_id=item.get("_source_trace_id")))
            elif predicate == "life_routine":
                profile.merge_lists(life_routines=[obj])
                items.extend(self._profile_items_from_values(project_id=project_id, user_id=user_id, facet_type=ProfileFacetType.LIFE_ROUTINES, values=[obj], source="consolidation_semantic", confidence=float(item.get("confidence", 0.68)), source_trace_id=item.get("_source_trace_id")))
        return items

    def _profile_items_from_values(
        self,
        *,
        project_id: str,
        user_id: str,
        facet_type: ProfileFacetType,
        values: list[Any],
        source: str,
        confidence: float = 0.7,
        source_trace_id: str | None = None,
    ) -> list[UserProfileItem]:
        items: list[UserProfileItem] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            items.append(
                UserProfileItem(
                    project_id=project_id,
                    user_id=user_id,
                    facet_type=facet_type,
                    value=text,
                    confidence=confidence,
                    source=source,
                    source_trace_id=source_trace_id,
                )
            )
        return items

    def _profile_item_from_payload(
        self,
        *,
        project_id: str,
        user_id: str,
        payload: dict[str, Any],
    ) -> UserProfileItem | None:
        facet_type = str(payload.get("facet_type") or "").strip()
        value = str(payload.get("value") or "").strip()
        if not facet_type or not value:
            return None
        try:
            normalized_facet = ProfileFacetType(facet_type)
        except ValueError:
            return None
        return UserProfileItem(
            project_id=project_id,
            user_id=user_id,
            facet_type=normalized_facet,
            value=value,
            confidence=clamp(float(payload.get("confidence", 0.7)), 0.0, 1.0),
            source=str(payload.get("source") or "llm_profile"),
            source_trace_id=payload.get("source_trace_id") or None,
            metadata={"status": str(payload.get("status") or "active")},
        )

    def _apply_relationship_updates(
        self,
        state: RelationshipState,
        update: dict[str, Any],
    ) -> None:
        stage = relationship_stage_from_value(update.get("current_stage"))
        conflict = update.get("recent_conflict_level")
        state.apply_delta(
            intimacy_delta=float(update.get("intimacy_delta", 0.0)),
            trust_delta=float(update.get("trust_delta", 0.0)),
            conflict_target=float(conflict) if conflict is not None else None,
            current_stage=stage,
            boundary_flags=[str(flag) for flag in update.get("boundary_flags", [])],
            notes=[str(note) for note in update.get("notes", [])],
            behavior_state=update.get("behavior_state"),
        )
        if state.recent_conflict_level >= 50:
            state.current_stage = RelationshipStage.REPAIR

    def _operation_from_value(self, value: str | None) -> MemoryOperation:
        if not value:
            return MemoryOperation.ADD
        try:
            return MemoryOperation(value.upper())
        except ValueError:
            return MemoryOperation.ADD

    def _sensitivity_from_value(self, value: str | None) -> SensitivityLevel:
        if not value:
            return SensitivityLevel.PRIVATE
        lowered = str(value).lower()
        for candidate in SensitivityLevel:
            if lowered == candidate.value:
                return candidate
        return SensitivityLevel.PRIVATE

    def _entity_key(self, value: str) -> str:
        return normalize_text(value)
