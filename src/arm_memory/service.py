from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from arm_memory.config import ARMConfig
from arm_memory.consolidation import SleepCycleConsolidator
from arm_memory.domain.models import (
    BoundaryStage,
    BuildContextResult,
    ConsolidationResult,
    ConversationTurn,
    EmotionLabel,
    EmotionVector,
    GraphEdge,
    MemoryKind,
    MemoryOperation,
    MemoryTrace,
    PersonaDefinition,
    ProfileFacetType,
    ProcedureRule,
    RelationshipState,
    SemanticFact,
    SensitivityLevel,
    UserManualSnapshot,
    UserProfileItem,
    relationship_from_dict,
    sensitivity_from_value,
    trace_from_dict,
)
from arm_memory.retrieval import HybridRetrievalEngine
from arm_memory.stores import Neo4jGraphStore, QdrantVectorStore, SQLiteMemoryStore
from arm_memory.utils import parse_markdown_document, read_text_if_exists, utcnow
from arm_memory.vectorizer import EmbeddingProvider, build_embedding_provider


class ARMMemoryService:
    def __init__(
        self,
        *,
        config: ARMConfig,
        sqlite_store: SQLiteMemoryStore | None = None,
        qdrant_store: QdrantVectorStore | None = None,
        neo4j_store: Neo4jGraphStore | None = None,
        vectorizer: EmbeddingProvider | None = None,
    ):
        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = vectorizer or build_embedding_provider(config)
        self.sqlite_store = sqlite_store or SQLiteMemoryStore(config.sqlite_path)
        self.qdrant_store = qdrant_store or QdrantVectorStore(
            url=config.qdrant_url,
            collection=config.qdrant_collection,
            vector_size=config.vector_dimensions,
        )
        self.neo4j_store = neo4j_store or Neo4jGraphStore(
            url=config.neo4j_url,
            user=config.neo4j_user,
            password=config.neo4j_password,
            database=config.neo4j_database,
        )
        self._validate_remote_dependencies()
        self.retrieval_engine = HybridRetrievalEngine(
            config=config,
            sqlite_store=self.sqlite_store,
            vectorizer=self.vectorizer,
            qdrant_store=self.qdrant_store,
            neo4j_store=self.neo4j_store,
        )
        self.consolidator = SleepCycleConsolidator(
            config=config,
            sqlite_store=self.sqlite_store,
            vectorizer=self.vectorizer,
            qdrant_store=self.qdrant_store,
            neo4j_store=self.neo4j_store,
        )
        self._ensure_default_persona_seed(project_id="system", user_id="system")
        logger.info(
            "ARM memory service initialized: sqlite={} qdrant_enabled={} neo4j_enabled={} log_path={}",
            self.config.sqlite_path,
            self.qdrant_store.enabled,
            self.neo4j_store.enabled,
            self.config.log_path,
        )

    @classmethod
    def from_env(cls, *, base_dir: Path | None = None) -> "ARMMemoryService":
        return cls(config=ARMConfig.from_env(base_dir=base_dir))

    def build_context(
        self,
        *,
        project_id: str,
        user_id: str,
        message: str,
        user_emotion_hint: EmotionVector | dict | None = None,
        max_context_tokens: int | None = None,
        focus_facets: list[str] | None = None,
        retrieval_mode: str = "default",
    ) -> BuildContextResult:
        logger.info(
            "Building context: project_id={} user_id={} message_length={}",
            project_id,
            user_id,
            len(message),
        )
        emotion_hint: EmotionVector | None = None
        if isinstance(user_emotion_hint, dict):
            emotion_hint = EmotionVector.from_dict(user_emotion_hint)
        elif isinstance(user_emotion_hint, EmotionVector):
            emotion_hint = user_emotion_hint

        parsed_facets: list[ProfileFacetType] | None = None
        if focus_facets:
            parsed_facets = []
            for f in focus_facets:
                try:
                    parsed_facets.append(ProfileFacetType(f))
                except ValueError:
                    pass

        persona_prompt = self.load_persona_prompt(project_id=project_id, user_id=user_id)
        relationship = self.sqlite_store.load_relationship_state(project_id, user_id)

        if parsed_facets:
            snapshot = self.consolidator.build_user_manual(
                project_id=project_id,
                user_id=user_id,
                persona_prompt=persona_prompt,
                focus_facets=parsed_facets,
            )
        else:
            snapshot = self.sqlite_store.load_user_manual(project_id, user_id)
            if snapshot is None:
                snapshot = self.consolidator.build_user_manual(
                    project_id=project_id,
                    user_id=user_id,
                    persona_prompt=persona_prompt,
                )
                self.sqlite_store.save_user_manual(snapshot)

        retrieval_query = message
        expand_n = self.config.retrieval_query_expand_turns
        if expand_n > 0:
            recent = self.sqlite_store.list_recent_turns(
                project_id, user_id, limit=expand_n
            )
            if recent:
                context_text = " ".join(t.content for t in recent[-expand_n:])
                retrieval_query = message + " " + context_text
                max_chars = self.config.embedding_max_length * 3
                if len(retrieval_query) > max_chars:
                    retrieval_query = retrieval_query[-max_chars:]

        hits = self.retrieval_engine.retrieve(
            project_id=project_id,
            user_id=user_id,
            query=retrieval_query,
            query_emotion_hint=emotion_hint,
            retrieval_mode=retrieval_mode,
        )
        for hit in hits:
            self.sqlite_store.record_memory_access(
                hit.trace.trace_id,
                salience_boost=self.config.salience_boost_per_access,
                salience_cap=self.config.salience_cap,
            )

        procedures = self.select_procedures(
            message=message,
            relationship=relationship,
            hits=hits,
        )
        result = BuildContextResult(
            persona_prompt=persona_prompt,
            relationship_prompt=self._format_relationship_prompt(relationship),
            user_manual_summary=snapshot.content,
            top_memories=hits,
            triggered_procedures=procedures,
        )
        if max_context_tokens is not None and max_context_tokens > 0:
            result.prompt_sections_capped = result.prompt_sections_with_budget(max_context_tokens)
        logger.info(
            "Context built: project_id={} user_id={} hits={} procedures={}",
            project_id,
            user_id,
            len(result.top_memories),
            len(result.triggered_procedures),
        )
        return result

    def get_relationship_state(self, *, project_id: str, user_id: str) -> RelationshipState:
        return self.sqlite_store.load_relationship_state(project_id, user_id)

    def save_relationship_state(self, state: RelationshipState) -> None:
        state.behavior_state.normalize()
        self.sqlite_store.save_relationship_state(state)
        if self.neo4j_store:
            self.neo4j_store.sync_relationship_state(
                project_id=state.project_id,
                user_id=state.user_id,
                companion_node=self.config.companion_node_name,
                state=state,
            )

    def get_user_manual_snapshot(
        self,
        *,
        project_id: str,
        user_id: str,
    ) -> UserManualSnapshot | None:
        return self.sqlite_store.load_user_manual(project_id, user_id)

    def get_recent_turns(
        self,
        *,
        project_id: str,
        user_id: str,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[ConversationTurn]:
        return self.sqlite_store.list_recent_turns(
            project_id,
            user_id,
            limit=limit,
            session_id=session_id,
            unconsolidated_only=False,
        )

    def get_recent_traces(
        self,
        *,
        project_id: str,
        user_id: str,
        limit: int = 20,
    ) -> list[MemoryTrace]:
        return self.sqlite_store.list_active_memory_traces(
            project_id,
            user_id,
            limit=limit,
        )

    def get_autonomy_snapshot(
        self,
        *,
        project_id: str,
        user_id: str,
        query: str = "",
        turns_limit: int = 20,
        traces_limit: int = 12,
    ) -> dict[str, Any]:
        build_context = self.build_context(
            project_id=project_id,
            user_id=user_id,
            message=query or "系统心跳：评估是否需要主动行动",
        )
        user_manual = self.get_user_manual_snapshot(project_id=project_id, user_id=user_id)
        return {
            "project_id": project_id,
            "user_id": user_id,
            "persona_prompt": self.load_persona_prompt(project_id=project_id, user_id=user_id),
            "active_persona": self.get_active_persona(project_id=project_id, user_id=user_id).to_dict(),
            "relationship": self.get_relationship_state(project_id=project_id, user_id=user_id).to_dict(),
            "user_manual": user_manual.to_dict() if user_manual else None,
            "policy_rules": [rule.to_dict() for rule in self.load_policy_rules()],
            "recent_turns": [
                turn.to_dict()
                for turn in self.get_recent_turns(
                    project_id=project_id,
                    user_id=user_id,
                    limit=turns_limit,
                )
            ],
            "recent_traces": [
                trace.to_dict()
                for trace in self.get_recent_traces(
                    project_id=project_id,
                    user_id=user_id,
                    limit=traces_limit,
                )
            ],
            "build_context": build_context.to_dict(),
        }

    def ingest_turn(
        self,
        *,
        project_id: str,
        user_id: str,
        role: str,
        content: str,
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        turn = ConversationTurn(
            project_id=project_id,
            user_id=user_id,
            role=role,
            content=content,
            session_id=session_id,
            metadata=metadata or {},
        )
        self.sqlite_store.store_turn(turn)
        logger.info(
            "Turn ingested: project_id={} user_id={} session_id={} role={} turn_id={} content_length={}",
            project_id,
            user_id,
            session_id,
            role,
            turn.turn_id,
            len(content),
        )
        return turn.turn_id

    def apply_extraction(
        self,
        *,
        project_id: str,
        user_id: str,
        extraction: dict,
        turn_ids: list[str] | None = None,
        persona_prompt: str = "",
    ):
        """接收外部已生成的结构化抽取结果并写入存储。对话→抽取由调用方在项目外完成。"""
        if not extraction:
            logger.info(
                "Apply extraction skipped: project_id={} user_id={} reason=empty_extraction",
                project_id,
                user_id,
            )
            return ConsolidationResult(notes=["extraction 为空，未执行写入。"])
        prompt = persona_prompt or self.load_persona_prompt(project_id=project_id, user_id=user_id)
        result = self.consolidator.apply_extraction(
            project_id=project_id,
            user_id=user_id,
            extraction=extraction,
            turn_ids=turn_ids,
            persona_prompt=prompt,
        )
        logger.info(
            "Extraction applied: project_id={} user_id={} episodic={} semantic={} procedures={} operations={}",
            project_id,
            user_id,
            result.episodic_count,
            result.semantic_count,
            len(result.triggered_procedures),
            result.operations,
        )
        return result

    def manual_remember(
        self,
        *,
        project_id: str,
        user_id: str,
        text: str,
        tags: list[str] | None = None,
        sensitivity: SensitivityLevel = SensitivityLevel.PRIVATE,
    ) -> MemoryOperation:
        now = utcnow()
        normalized_tags = tags or ["manual", "operator_note"]
        trace = MemoryTrace(
            project_id=project_id,
            user_id=user_id,
            kind=MemoryKind.SEMANTIC,
            summary=text.strip(),
            raw_text=text.strip(),
            salience=0.95,
            tags=normalized_tags,
            entities=self.vectorizer.tokenize(text)[:8],
            vector=self.vectorizer.embed(text),
            sensitivity=sensitivity,
            created_at=now,
            updated_at=now,
            happened_at=now,
            last_accessed_at=now,
            metadata={"source": "manual"},
        )
        self.sqlite_store.add_memory_trace(trace)
        fact = SemanticFact(
            project_id=project_id,
            user_id=user_id,
            subject="user",
            predicate="manual_note",
            object_text=text.strip(),
            summary=f"人工记忆：{text.strip()}",
            confidence=1.0,
            salience=0.95,
            source_trace_id=trace.trace_id,
            tags=normalized_tags,
            entities=trace.entities,
            sensitivity=sensitivity,
            created_at=now,
            updated_at=now,
            valid_from=now,
            metadata={"source": "manual"},
        )
        operation = self.sqlite_store.apply_semantic_fact(fact)
        self.sqlite_store.upsert_graph_edge(
            GraphEdge(
                project_id=project_id,
                user_id=user_id,
                source_node="user",
                edge_type="manual_note",
                target_node=self._entity_key(text),
                weight=0.95,
                metadata={"summary": fact.summary},
            )
        )
        if self.qdrant_store:
            self.qdrant_store.upsert_trace(trace)
        if self.neo4j_store:
            self.neo4j_store.sync_fact(fact)
        profile = self.sqlite_store.load_profile(project_id, user_id)
        profile.merge_lists(manual_notes=[text.strip()])
        self.sqlite_store.save_profile(profile)
        self.sqlite_store.save_profile_item(
            UserProfileItem(
                project_id=project_id,
                user_id=user_id,
                facet_type=ProfileFacetType.MANUAL_NOTES,
                value=text.strip(),
                confidence=1.0,
                source="manual_remember",
                source_trace_id=trace.trace_id,
            )
            ,
            facet_limit=self.config.profile_item_limit_per_facet,
        )
        snapshot = self.consolidator.build_user_manual(
            project_id=project_id,
            user_id=user_id,
            persona_prompt=self.load_persona_prompt(project_id=project_id, user_id=user_id),
        )
        self.sqlite_store.save_user_manual(snapshot)
        logger.info(
            "Manual memory saved: project_id={} user_id={} trace_id={} operation={} tags={}",
            project_id,
            user_id,
            trace.trace_id,
            operation.value,
            normalized_tags,
        )
        return operation

    def get_summary(self, *, project_id: str, user_id: str) -> str:
        stats = self.sqlite_store.stats(project_id, user_id)
        relationship = self.sqlite_store.load_relationship_state(project_id, user_id)
        snapshot = self.sqlite_store.load_user_manual(project_id, user_id)
        facts = self.sqlite_store.list_semantic_facts(project_id, user_id, limit=6)
        active_persona = self.get_active_persona(project_id=project_id, user_id=user_id)

        lines = [
            "ARM 记忆摘要",
            f"- 情景记忆：{stats['episodic']}",
            f"- 语义事实：{stats['semantic']}",
            f"- 对话轮次：{stats['turns']}",
            f"- 画像条目：{stats['profile_items']}",
            f"- Persona 数量：{stats['personas']}",
            f"- 当前 Persona：{active_persona.name}",
            f"- 关系阶段：{relationship.current_stage.value}",
            f"- 亲密度：{relationship.intimacy_level:.1f}/100",
            f"- 信任度：{relationship.trust_score:.1f}/100",
            f"- 冲突水平：{relationship.recent_conflict_level:.1f}/100",
            f"- 边界阶段：{relationship.behavior_state.boundary_stage.value}",
        ]
        if relationship.boundary_flags:
            lines.append("- 风险边界：" + "；".join(relationship.boundary_flags[:5]))
        if relationship.behavior_state.violation_count:
            lines.append(f"- 越界累计：{relationship.behavior_state.violation_count}")
        if relationship.behavior_state.in_cooldown():
            lines.append(
                "- 冷却中直到：" + str(relationship.behavior_state.cooldown_until)
            )
        if facts:
            lines.append("")
            lines.append("近期长期事实：")
            for fact in facts:
                lines.append(f"- {fact.summary}")
        if snapshot:
            lines.append("")
            lines.append("用户说明书预览：")
            preview = snapshot.content[:600]
            if len(snapshot.content) > 600:
                preview += "\n..."
            lines.append(preview)
        return "\n".join(lines)

    def clear_user_data(self, *, project_id: str, user_id: str) -> None:
        self.sqlite_store.clear_user_data(project_id, user_id)
        logger.warning("User data cleared: project_id={} user_id={}", project_id, user_id)

    def replay_remote_sync_outbox(self, *, limit: int = 50) -> dict[str, int]:
        tasks = self.sqlite_store.claim_remote_sync_tasks(
            limit=limit,
            max_attempts=self.config.outbox_max_attempts,
            reclaim_in_progress_after_seconds=self.config.outbox_reclaim_in_progress_after_seconds,
        )
        result = {"processed": 0, "succeeded": 0, "failed": 0}
        for task in tasks:
            result["processed"] += 1
            task_id = str(task["task_id"])
            try:
                self._replay_remote_sync_task(task)
            except Exception as exc:
                self.sqlite_store.mark_remote_sync_task_failed(
                    task_id,
                    error_message=str(exc),
                    max_attempts=self.config.outbox_max_attempts,
                    retry_base_seconds=self.config.outbox_retry_base_seconds,
                    retry_max_seconds=self.config.outbox_retry_max_seconds,
                )
                result["failed"] += 1
                logger.warning(
                    "Remote sync replay failed: task_id={} backend={} operation={} error={}",
                    task_id,
                    task.get("backend"),
                    task.get("operation"),
                    exc,
                )
            else:
                self.sqlite_store.mark_remote_sync_task_done(task_id)
                result["succeeded"] += 1
        logger.info("Remote sync replay summary: {}", result)
        return result

    def get_remote_sync_outbox_summary(self) -> dict[str, int]:
        return self.sqlite_store.get_remote_sync_outbox_summary()

    def list_profile_items(
        self,
        *,
        project_id: str,
        user_id: str,
        facet_types: list[str] | None = None,
        include_stale: bool = False,
        limit: int = 200,
    ) -> list[UserProfileItem]:
        return self.sqlite_store.list_profile_items(
            project_id,
            user_id,
            facet_types=facet_types,
            max_age_days=None if include_stale else self.config.profile_item_stale_days,
            limit=limit,
        )

    def save_profile_item(
        self,
        *,
        project_id: str,
        user_id: str,
        facet_type: str,
        value: str,
        confidence: float = 0.75,
        source: str = "manual",
        source_trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UserProfileItem:
        item = UserProfileItem(
            project_id=project_id,
            user_id=user_id,
            facet_type=ProfileFacetType(facet_type),
            value=value.strip(),
            confidence=confidence,
            source=source,
            source_trace_id=source_trace_id,
            metadata=metadata or {},
        )
        self.sqlite_store.save_profile_item(item, facet_limit=self.config.profile_item_limit_per_facet)
        logger.info(
            "Profile item saved: project_id={} user_id={} facet_type={} source={}",
            project_id,
            user_id,
            item.facet_type.value,
            source,
        )
        return item

    def list_personas(
        self,
        *,
        project_id: str,
        user_id: str,
        include_disabled: bool = True,
    ) -> list[PersonaDefinition]:
        personas = self.sqlite_store.list_personas(
            project_id=project_id,
            user_id=user_id,
            include_disabled=include_disabled,
        )
        if personas:
            return personas
        return [self._ensure_default_persona_seed(project_id=project_id, user_id=user_id)]

    def get_active_persona(self, *, project_id: str, user_id: str) -> PersonaDefinition:
        persona = self.sqlite_store.get_active_persona(project_id, user_id)
        if persona is not None:
            return persona
        return self._ensure_default_persona_seed(project_id=project_id, user_id=user_id)

    def save_persona(self, persona: PersonaDefinition) -> PersonaDefinition:
        self.sqlite_store.save_persona(persona)
        logger.info(
            "Persona saved: project_id={} user_id={} persona_id={} name={} active={}",
            persona.project_id,
            persona.user_id,
            persona.persona_id,
            persona.name,
            persona.is_active,
        )
        return persona

    def activate_persona(self, *, project_id: str, user_id: str, persona_id: str) -> PersonaDefinition:
        self.sqlite_store.activate_persona(project_id, user_id, persona_id)
        persona = self.get_active_persona(project_id=project_id, user_id=user_id)
        logger.info(
            "Persona activated: project_id={} user_id={} persona_id={} name={}",
            project_id,
            user_id,
            persona.persona_id,
            persona.name,
        )
        return persona

    def _validate_remote_dependencies(self) -> None:
        """当 config 要求时，校验 Qdrant / Neo4j 已配置且可达，否则退出并输出启动方法。"""
        _qdrant_help = (
            "启动 Qdrant: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant\n"
            "配置: ARM_QDRANT_URL=http://127.0.0.1:6333"
        )
        _neo4j_help = (
            "启动 Neo4j: docker run -d --name neo4j -p 7474:7474 -p 7687:7687 "
            "-e NEO4J_AUTH=neo4j/your-password neo4j:latest\n"
            "配置: ARM_NEO4J_URL=http://127.0.0.1:7474 ARM_NEO4J_USER=neo4j ARM_NEO4J_PASSWORD=your-password"
        )
        if self.config.require_qdrant and not self.qdrant_store.enabled:
            raise ValueError(
                "Qdrant 为必选依赖，ARM_QDRANT_URL 未配置。\n" + _qdrant_help
            )
        if self.config.require_neo4j and not self.neo4j_store.enabled:
            raise ValueError(
                "Neo4j 为必选依赖，ARM_NEO4J_* 未配置完整。\n" + _neo4j_help
            )
        if self.config.require_qdrant:
            self.qdrant_store.ensure_collection(strict=True)
            if not self.qdrant_store.ping():
                raise RuntimeError("Qdrant 不可达，请先启动 Qdrant 并配置 ARM_QDRANT_URL。\n" + _qdrant_help)
        if self.config.require_neo4j and not self.neo4j_store.ping():
            raise RuntimeError("Neo4j 不可达，请先启动 Neo4j 并配置 ARM_NEO4J_*。\n" + _neo4j_help)

    def load_persona_prompt(self, *, project_id: str = "system", user_id: str = "system") -> str:
        return self.get_active_persona(project_id=project_id, user_id=user_id).prompt

    def load_policy_rules(self) -> list[ProcedureRule]:
        rules: list[ProcedureRule] = []
        if not self.config.policies_dir.exists():
            return rules
        for path in sorted(self.config.policies_dir.glob("*.md")):
            metadata, body = parse_markdown_document(path)
            if not body.strip():
                continue
            rules.append(
                ProcedureRule(
                    name=str(metadata.get("name") or path.stem),
                    prompt=body.strip(),
                    priority=int(metadata.get("priority", 50)),
                    triggers=[str(item) for item in metadata.get("triggers", [])],
                    tags=[str(item) for item in metadata.get("tags", [])],
                    path=str(path),
                    enabled=bool(metadata.get("enabled", True)),
                )
            )
        rules.sort(key=lambda item: item.priority, reverse=True)
        return rules

    def select_procedures(
        self,
        *,
        message: str,
        relationship: RelationshipState,
        hits,
    ) -> list[ProcedureRule]:
        rules = {rule.name: rule for rule in self.load_policy_rules() if rule.enabled}
        lowered = message.lower()
        selected: list[ProcedureRule] = []
        behavior = relationship.behavior_state
        behavior.normalize()

        if relationship.recent_conflict_level >= 45 or relationship.trust_score <= 45:
            rule = rules.get("rebuild_trust")
            if rule:
                selected.append(rule)
        if any("boundary" in flag for flag in relationship.boundary_flags):
            rule = rules.get("set_boundary")
            if rule and rule not in selected:
                selected.append(rule)
        if behavior.boundary_stage == BoundaryStage.WARNING:
            rule = rules.get("warn_disrespect")
            if rule and rule not in selected:
                selected.append(rule)
        elif behavior.boundary_stage == BoundaryStage.REFUSAL:
            rule = rules.get("angry_refusal")
            if rule and rule not in selected:
                selected.append(rule)
        elif behavior.boundary_stage == BoundaryStage.COOLDOWN and behavior.in_cooldown():
            rule = rules.get("cooldown_mode")
            if rule and rule not in selected:
                selected.append(rule)
        if not selected:
            for hit in hits:
                if hit.trace.emotion.label in {EmotionLabel.SADNESS, EmotionLabel.FEAR}:
                    rule = rules.get("comfort_user")
                    if rule:
                        selected.append(rule)
                        break
        unique: list[ProcedureRule] = []
        seen: set[str] = set()
        for rule in selected:
            if rule.name in seen:
                continue
            seen.add(rule.name)
            unique.append(rule)
        unique.sort(key=lambda item: item.priority, reverse=True)
        return unique

    def _format_relationship_prompt(self, relationship: RelationshipState) -> str:
        lines = [
            f"- 当前关系阶段：{relationship.current_stage.value}",
            f"- 当前亲密度：{relationship.intimacy_level:.1f}/100",
            f"- 当前信任度：{relationship.trust_score:.1f}/100",
            f"- 近期冲突水平：{relationship.recent_conflict_level:.1f}/100",
            f"- 当前边界阶段：{relationship.behavior_state.boundary_stage.value}",
            "- 回复时要保持连续性、温柔度和清晰边界，不要因讨好而失去人格一致性。",
        ]
        if relationship.boundary_flags:
            lines.append("- 当前边界关注：" + "；".join(relationship.boundary_flags[:5]))
        if relationship.notes:
            lines.append("- 关系注记：" + "；".join(relationship.notes[:4]))
        if relationship.behavior_state.violation_count:
            lines.append(f"- 近期越界累计：{relationship.behavior_state.violation_count}")
        if relationship.behavior_state.in_cooldown():
            lines.append("- 当前仍在冷处理期，除非用户明确修复，否则只允许一句边界回应或不回应。")
        elif relationship.behavior_state.boundary_stage == BoundaryStage.REFUSAL:
            lines.append("- 目前要直接拒绝冒犯、操控或越界内容，不转移话题，不假装没事。")
        elif relationship.behavior_state.boundary_stage == BoundaryStage.WARNING:
            lines.append("- 目前要保留一点不满和距离感，提醒对方注意说话方式。")
        if relationship.current_stage == relationship.current_stage.REPAIR:
            lines.append("- 优先修复信任，先确认感受，再给解释和行动承诺。")
        return "\n".join(lines)

    def _replay_remote_sync_task(self, task: dict[str, object]) -> None:
        backend = str(task["backend"])
        operation = str(task["operation"])
        payload = dict(task["payload"] or {})
        project_id = str(task["project_id"])
        user_id = str(task["user_id"])

        if backend == "qdrant":
            if not self.qdrant_store.enabled:
                raise RuntimeError("Qdrant backend is not configured")
            if operation != "upsert_trace":
                raise ValueError(f"Unsupported Qdrant outbox operation: {operation}")
            self.qdrant_store.upsert_trace(trace_from_dict(payload))
            return

        if backend == "neo4j":
            if not self.neo4j_store.enabled:
                raise RuntimeError("Neo4j backend is not configured")
            if operation == "sync_fact":
                self.neo4j_store.sync_fact(self._semantic_fact_from_payload(payload))
                return
            if operation == "sync_edge":
                self.neo4j_store.sync_edge(self._graph_edge_from_payload(payload))
                return
            if operation == "sync_relationship_state":
                self.neo4j_store.sync_relationship_state(
                    project_id=project_id,
                    user_id=user_id,
                    companion_node=self.config.companion_node_name,
                    state=relationship_from_dict(payload),
                )
                return
            raise ValueError(f"Unsupported Neo4j outbox operation: {operation}")

        raise ValueError(f"Unsupported outbox backend: {backend}")

    def _ensure_default_persona_seed(self, *, project_id: str, user_id: str) -> PersonaDefinition:
        existing = self.sqlite_store.get_active_persona(project_id, user_id)
        if existing is not None:
            return existing
        text = read_text_if_exists(self.config.companion_profile_path)
        prompt = text or "# Companion Profile\n\n你是一个稳定、真诚、富有边界感的虚拟恋人。"
        persona = PersonaDefinition(
            project_id=project_id,
            user_id=user_id,
            name="companion-default",
            prompt=prompt,
            policy_pack="default",
            description="Seeded from companion_profile.md",
            is_enabled=True,
            is_active=True,
            source_path=str(self.config.companion_profile_path),
        )
        self.sqlite_store.save_persona(persona)
        return persona

    def _semantic_fact_from_payload(self, payload: dict[str, Any]) -> SemanticFact:
        return SemanticFact(
            fact_id=str(payload.get("fact_id") or ""),
            project_id=str(payload["project_id"]),
            user_id=str(payload["user_id"]),
            subject=str(payload.get("subject") or ""),
            predicate=str(payload.get("predicate") or ""),
            object_text=str(payload.get("object_text") or ""),
            summary=str(payload.get("summary") or ""),
            confidence=float(payload.get("confidence", 0.75)),
            salience=float(payload.get("salience", 0.65)),
            source_trace_id=payload.get("source_trace_id") or None,
            source_turn_ids=list(payload.get("source_turn_ids") or []),
            entities=list(payload.get("entities") or []),
            tags=list(payload.get("tags") or []),
            emotion=EmotionVector.from_dict(payload.get("emotion")),
            sensitivity=sensitivity_from_value(payload.get("sensitivity")),
            status=str(payload.get("status") or "active"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def _graph_edge_from_payload(self, payload: dict[str, Any]) -> GraphEdge:
        return GraphEdge(
            edge_id=str(payload.get("edge_id") or ""),
            project_id=str(payload["project_id"]),
            user_id=str(payload["user_id"]),
            source_node=str(payload.get("source_node") or ""),
            edge_type=str(payload.get("edge_type") or ""),
            target_node=str(payload.get("target_node") or ""),
            weight=float(payload.get("weight", 1.0)),
            emotion=EmotionVector.from_dict(payload.get("emotion")),
            status=str(payload.get("status") or "active"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def _entity_key(self, value: str) -> str:
        return " ".join(self.vectorizer.tokenize(value)) or value.strip().lower()

    def observability_snapshot(
        self,
        *,
        project_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Return a structured dict of key metrics for monitoring/debugging."""
        stats = self.sqlite_store.stats(project_id, user_id)
        outbox = self.sqlite_store.get_remote_sync_outbox_summary()
        relationship = self.sqlite_store.load_relationship_state(project_id, user_id)
        return {
            "project_id": project_id,
            "user_id": user_id,
            "memory_counts": stats,
            "sync_outbox": outbox,
            "relationship": {
                "stage": relationship.current_stage.value,
                "intimacy": relationship.intimacy_level,
                "trust": relationship.trust_score,
                "conflict": relationship.recent_conflict_level,
                "boundary_stage": relationship.behavior_state.boundary_stage.value,
                "violation_count": relationship.behavior_state.violation_count,
                "in_cooldown": relationship.behavior_state.in_cooldown(),
            },
            "config_snapshot": {
                "retrieval_candidate_limit": self.config.retrieval_candidate_limit,
                "memory_archive_min_age_days": self.config.memory_archive_min_age_days,
                "memory_archive_min_salience": self.config.memory_archive_min_salience,
                "memory_merge_similarity_threshold": self.config.memory_merge_similarity_threshold,
            },
        }
