from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from arm_memory.utils import ensure_utc, to_iso, utcnow


class MemoryKind(StrEnum):
    """记忆类型：工作记忆、情景记忆、语义记忆、程序记忆。"""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryOperation(StrEnum):
    """记忆操作类型：新增、更新、删除、无操作。"""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class SensitivityLevel(StrEnum):
    """敏感度级别：公开、私密、敏感、受限。"""

    PUBLIC = "public"
    PRIVATE = "private"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class EmotionLabel(StrEnum):
    """情绪标签（基于 Plutchik 等）：中性、喜悦、悲伤、恐惧、愤怒、厌恶、惊讶、信任、期待。"""

    NEUTRAL = "neutral"
    JOY = "joy"
    SADNESS = "sadness"
    FEAR = "fear"
    ANGER = "anger"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


class RelationshipStage(StrEnum):
    """关系阶段：初次接触、短期互动、关系加深、长期纽带、修复、疏远。"""

    INITIAL_CONTACT = "InitialContact"
    SHORT_TERM_INTERACTION = "ShortTermInteraction"
    GROWING_BOND = "GrowingBond"
    LONG_TERM_BOND = "LongTermBond"
    REPAIR = "Repair"
    DISTANT = "Distant"


class BoundaryStage(StrEnum):
    """边界阶段：平静、警告、拒绝、冷却。"""

    CALM = "calm"
    WARNING = "warning"
    REFUSAL = "refusal"
    COOLDOWN = "cooldown"


class ProfileFacetType(StrEnum):
    """用户档案维度类型：身份备注、偏好、不喜欢、边界、脆弱点、支持偏好、目标、重要的人/地点、手动备注、人格特质、爱好、习惯、沟通风格、依恋风格、生活作息等。"""

    IDENTITY_NOTES = "identity_notes"
    PREFERENCES = "preferences"
    DISLIKES = "dislikes"
    BOUNDARIES = "boundaries"
    VULNERABILITIES = "vulnerabilities"
    SUPPORT_PREFERENCES = "support_preferences"
    GOALS = "goals"
    IMPORTANT_PEOPLE = "important_people"
    IMPORTANT_PLACES = "important_places"
    MANUAL_NOTES = "manual_notes"
    PERSONALITY_TRAITS = "personality_traits"
    HOBBIES = "hobbies"
    HABITS = "habits"
    COMMUNICATION_STYLE = "communication_style"
    ATTACHMENT_STYLE = "attachment_style"
    LIFE_ROUTINES = "life_routines"


class ProfileItemStatus(StrEnum):
    """档案项状态：活跃、已归档、已替代、已删除。"""

    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


def _serialize(value: Any) -> Any:
    """将 datetime、StrEnum、列表、字典及带 to_dict 的对象递归序列化为可 JSON 兼容的类型。"""
    if isinstance(value, datetime):
        return to_iso(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass(slots=True)
class EmotionVector:
    """情绪向量：标签 + 效价/唤醒/支配/强度/置信度，用于记忆与事实的情绪标注。"""

    label: EmotionLabel = EmotionLabel.NEUTRAL
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    intensity: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "intensity": self.intensity,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "EmotionVector":
        payload = payload or {}
        label = payload.get("label", EmotionLabel.NEUTRAL.value)
        try:
            emotion = EmotionLabel(label)
        except ValueError:
            emotion = EmotionLabel.NEUTRAL
        return cls(
            label=emotion,
            valence=float(payload.get("valence", 0.0)),
            arousal=float(payload.get("arousal", 0.0)),
            dominance=float(payload.get("dominance", 0.0)),
            intensity=float(payload.get("intensity", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(slots=True)
class ConversationTurn:
    """单轮对话：项目/用户/会话、角色、内容、时间戳与元数据。"""

    project_id: str
    user_id: str
    role: str
    content: str
    session_id: str = "default"
    turn_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "created_at": to_iso(self.created_at),
            "metadata": _serialize(self.metadata),
        }


@dataclass(slots=True)
class MemoryTrace:
    """记忆痕迹：类型、摘要、原文、显著性、访问计数、标签/实体/向量、情绪与敏感度等。"""

    project_id: str
    user_id: str
    kind: MemoryKind
    summary: str
    raw_text: str = ""
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "active"
    salience: float = 0.5
    decay_rate: float = 0.01
    access_count: int = 0
    source_turn_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    vector: list[float] = field(default_factory=list)
    emotion: EmotionVector = field(default_factory=EmotionVector)
    emotion_tag: EmotionLabel = field(default=EmotionLabel.NEUTRAL)
    sensitivity: SensitivityLevel = SensitivityLevel.PRIVATE
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    happened_at: datetime = field(default_factory=utcnow)
    last_accessed_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.tags) > 50:
            self.tags = self.tags[:50]
        if len(self.entities) > 50:
            self.entities = self.entities[:50]
        if len(self.source_turn_ids) > 50:
            self.source_turn_ids = self.source_turn_ids[:50]

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed_at = utcnow()
        self.updated_at = self.last_accessed_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "kind": self.kind.value,
            "summary": self.summary,
            "raw_text": self.raw_text,
            "status": self.status,
            "salience": self.salience,
            "decay_rate": self.decay_rate,
            "access_count": self.access_count,
            "source_turn_ids": self.source_turn_ids[:50],
            "tags": self.tags[:50],
            "entities": self.entities[:50],
            "vector": self.vector,
            "emotion": self.emotion.to_dict(),
            "emotion_tag": self.emotion_tag.value,
            "sensitivity": self.sensitivity.value,
            "created_at": to_iso(self.created_at),
            "updated_at": to_iso(self.updated_at),
            "happened_at": to_iso(self.happened_at),
            "last_accessed_at": to_iso(self.last_accessed_at),
            "metadata": _serialize(self.metadata),
        }


@dataclass(slots=True)
class SemanticFact:
    """语义事实：主谓宾三元组，带置信度、显著性、来源追溯与有效期。"""

    project_id: str
    user_id: str
    subject: str
    predicate: str
    object_text: str
    fact_id: str = field(default_factory=lambda: uuid4().hex)
    summary: str = ""
    confidence: float = 0.75
    salience: float = 0.65
    source_trace_id: str | None = None
    source_turn_ids: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    emotion: EmotionVector = field(default_factory=EmotionVector)
    sensitivity: SensitivityLevel = SensitivityLevel.PRIVATE
    status: str = "active"
    valid_from: datetime = field(default_factory=utcnow)
    valid_to: datetime | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_key(self) -> str:
        return f"{self.subject.strip().lower()}|{self.predicate.strip().lower()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_text": self.object_text,
            "summary": self.summary or f"{self.subject} {self.predicate} {self.object_text}",
            "confidence": self.confidence,
            "salience": self.salience,
            "source_trace_id": self.source_trace_id,
            "source_turn_ids": self.source_turn_ids,
            "entities": self.entities,
            "tags": self.tags,
            "emotion": self.emotion.to_dict(),
            "sensitivity": self.sensitivity.value,
            "status": self.status,
            "valid_from": to_iso(self.valid_from),
            "valid_to": to_iso(self.valid_to) if self.valid_to else None,
            "created_at": to_iso(self.created_at),
            "updated_at": to_iso(self.updated_at),
            "metadata": _serialize(self.metadata),
        }


@dataclass(slots=True)
class GraphEdge:
    """知识图谱边：源节点、边类型、目标节点、权重及情绪/状态。"""

    project_id: str
    user_id: str
    source_node: str
    edge_type: str
    target_node: str
    weight: float = 1.0
    edge_id: str = field(default_factory=lambda: uuid4().hex)
    emotion: EmotionVector = field(default_factory=EmotionVector)
    status: str = "active"
    updated_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "source_node": self.source_node,
            "edge_type": self.edge_type,
            "target_node": self.target_node,
            "weight": self.weight,
            "emotion": self.emotion.to_dict(),
            "status": self.status,
            "updated_at": to_iso(self.updated_at),
            "metadata": _serialize(self.metadata),
        }


@dataclass(slots=True)
class UserProfile:
    """用户档案聚合：展示名及各维度列表（身份、偏好、边界、目标、重要的人/地点等）。"""

    project_id: str
    user_id: str
    display_name: str = ""
    identity_notes: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
    vulnerabilities: list[str] = field(default_factory=list)
    support_preferences: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    important_people: list[str] = field(default_factory=list)
    important_places: list[str] = field(default_factory=list)
    manual_notes: list[str] = field(default_factory=list)
    personality_traits: list[str] = field(default_factory=list)
    hobbies: list[str] = field(default_factory=list)
    habits: list[str] = field(default_factory=list)
    communication_style: list[str] = field(default_factory=list)
    attachment_style: list[str] = field(default_factory=list)
    life_routines: list[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=utcnow)

    def merge_lists(self, **fields: list[str]) -> None:
        for key, values in fields.items():
            merged = getattr(self, key)
            seen = {item.lower() for item in merged}
            for value in values:
                text = value.strip()
                if not text or text.lower() in seen:
                    continue
                merged.append(text)
                seen.add(text.lower())
        self.updated_at = utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "display_name": self.display_name,
            "identity_notes": self.identity_notes,
            "preferences": self.preferences,
            "dislikes": self.dislikes,
            "boundaries": self.boundaries,
            "vulnerabilities": self.vulnerabilities,
            "support_preferences": self.support_preferences,
            "goals": self.goals,
            "important_people": self.important_people,
            "important_places": self.important_places,
            "manual_notes": self.manual_notes,
            "personality_traits": self.personality_traits,
            "hobbies": self.hobbies,
            "habits": self.habits,
            "communication_style": self.communication_style,
            "attachment_style": self.attachment_style,
            "life_routines": self.life_routines,
            "updated_at": to_iso(self.updated_at),
        }


@dataclass(slots=True)
class UserProfileItem:
    """单条档案项：维度类型、取值、置信度、来源与证据次数等。"""

    project_id: str
    user_id: str
    facet_type: ProfileFacetType
    value: str
    item_id: str = field(default_factory=lambda: uuid4().hex)
    confidence: float = 0.7
    source: str = "unknown"
    source_trace_id: str | None = None
    first_seen_at: datetime = field(default_factory=utcnow)
    last_seen_at: datetime = field(default_factory=utcnow)
    evidence_count: int = 1
    status: ProfileItemStatus = ProfileItemStatus.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_value(self) -> str:
        return self.value.strip().lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "facet_type": self.facet_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "source_trace_id": self.source_trace_id,
            "first_seen_at": to_iso(self.first_seen_at),
            "last_seen_at": to_iso(self.last_seen_at),
            "evidence_count": self.evidence_count,
            "status": self.status.value,
            "metadata": _serialize(self.metadata),
        }


@dataclass(slots=True)
class PersonaDefinition:
    """人格定义：名称、提示词、策略包、是否启用/激活及来源路径。"""

    project_id: str
    user_id: str
    name: str
    prompt: str
    persona_id: str = field(default_factory=lambda: uuid4().hex)
    policy_pack: str = "default"
    description: str = ""
    is_enabled: bool = True
    is_active: bool = False
    source_path: str = ""
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "name": self.name,
            "prompt": self.prompt,
            "policy_pack": self.policy_pack,
            "description": self.description,
            "is_enabled": self.is_enabled,
            "is_active": self.is_active,
            "source_path": self.source_path,
            "created_at": to_iso(self.created_at),
            "updated_at": to_iso(self.updated_at),
        }


@dataclass(slots=True)
class BehaviorState:
    """行为/边界状态：边界阶段、违规次数、冷却截止、最近违规/修复时间及标记。"""

    boundary_stage: BoundaryStage = BoundaryStage.CALM
    violation_count: int = 0
    cooldown_until: datetime | None = None
    last_violation_at: datetime | None = None
    last_repair_at: datetime | None = None
    last_boundary_reason: str = ""
    recent_flags: list[str] = field(default_factory=list)

    def normalize(self, now: datetime | None = None) -> None:
        current = now or utcnow()
        if self.cooldown_until and self.cooldown_until <= current:
            self.cooldown_until = None
            if self.boundary_stage == BoundaryStage.COOLDOWN:
                if self.violation_count >= 2:
                    self.boundary_stage = BoundaryStage.REFUSAL
                elif self.violation_count == 1:
                    self.boundary_stage = BoundaryStage.WARNING
                else:
                    self.boundary_stage = BoundaryStage.CALM

    def in_cooldown(self, now: datetime | None = None) -> bool:
        self.normalize(now)
        current = now or utcnow()
        return self.cooldown_until is not None and self.cooldown_until > current

    def to_dict(self) -> dict[str, Any]:
        self.normalize()
        return {
            "boundary_stage": self.boundary_stage.value,
            "violation_count": self.violation_count,
            "cooldown_until": to_iso(self.cooldown_until) if self.cooldown_until else None,
            "last_violation_at": to_iso(self.last_violation_at) if self.last_violation_at else None,
            "last_repair_at": to_iso(self.last_repair_at) if self.last_repair_at else None,
            "last_boundary_reason": self.last_boundary_reason,
            "recent_flags": self.recent_flags,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BehaviorState":
        payload = payload or {}
        stage = payload.get("boundary_stage", BoundaryStage.CALM.value)
        try:
            boundary_stage = BoundaryStage(stage)
        except ValueError:
            boundary_stage = BoundaryStage.CALM
        instance = cls(
            boundary_stage=boundary_stage,
            violation_count=max(0, int(payload.get("violation_count", 0))),
            cooldown_until=ensure_utc(datetime.fromisoformat(payload["cooldown_until"]))
            if payload.get("cooldown_until")
            else None,
            last_violation_at=ensure_utc(datetime.fromisoformat(payload["last_violation_at"]))
            if payload.get("last_violation_at")
            else None,
            last_repair_at=ensure_utc(datetime.fromisoformat(payload["last_repair_at"]))
            if payload.get("last_repair_at")
            else None,
            last_boundary_reason=str(payload.get("last_boundary_reason", "")),
            recent_flags=list(payload.get("recent_flags") or []),
        )
        instance.normalize()
        return instance


@dataclass(slots=True)
class RelationshipState:
    """关系状态：亲密度、信任分、依恋猜测、阶段、冲突水平、边界标记与行为状态。"""

    project_id: str
    user_id: str
    intimacy_level: float = 20.0
    trust_score: float = 50.0
    attachment_style_guess: str = "unknown"
    current_stage: RelationshipStage = RelationshipStage.INITIAL_CONTACT
    recent_conflict_level: float = 0.0
    boundary_flags: list[str] = field(default_factory=list)
    last_emotional_peak_at: datetime | None = None
    notes: list[str] = field(default_factory=list)
    behavior_state: BehaviorState = field(default_factory=BehaviorState)
    updated_at: datetime = field(default_factory=utcnow)

    def apply_delta(
        self,
        *,
        intimacy_delta: float = 0.0,
        trust_delta: float = 0.0,
        conflict_target: float | None = None,
        current_stage: RelationshipStage | None = None,
        boundary_flags: list[str] | None = None,
        notes: list[str] | None = None,
        behavior_state: BehaviorState | dict[str, Any] | None = None,
    ) -> None:
        self.intimacy_level = max(0.0, min(100.0, self.intimacy_level + intimacy_delta))
        self.trust_score = max(0.0, min(100.0, self.trust_score + trust_delta))
        if conflict_target is not None:
            self.recent_conflict_level = max(0.0, min(100.0, conflict_target))
        else:
            self.recent_conflict_level = max(0.0, min(100.0, self.recent_conflict_level * 0.92))
        if current_stage is not None:
            self.current_stage = current_stage
        if boundary_flags:
            seen = {item.lower() for item in self.boundary_flags}
            for flag in boundary_flags:
                if flag.lower() not in seen:
                    self.boundary_flags.append(flag)
                    seen.add(flag.lower())
        if notes:
            existing = {item.lower() for item in self.notes}
            for note in notes:
                if note.lower() not in existing:
                    self.notes.append(note)
                    existing.add(note.lower())
        if behavior_state is not None:
            if isinstance(behavior_state, BehaviorState):
                self.behavior_state = behavior_state
            else:
                self.behavior_state = BehaviorState.from_dict(behavior_state)
        self.behavior_state.normalize()
        self.updated_at = utcnow()

    def to_dict(self) -> dict[str, Any]:
        self.behavior_state.normalize()
        return {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "intimacy_level": self.intimacy_level,
            "trust_score": self.trust_score,
            "attachment_style_guess": self.attachment_style_guess,
            "current_stage": self.current_stage.value,
            "recent_conflict_level": self.recent_conflict_level,
            "boundary_flags": self.boundary_flags,
            "last_emotional_peak_at": to_iso(self.last_emotional_peak_at) if self.last_emotional_peak_at else None,
            "notes": self.notes,
            "behavior_state": self.behavior_state.to_dict(),
            "updated_at": to_iso(self.updated_at),
        }


@dataclass(slots=True)
class ProcedureRule:
    """程序/策略规则：名称、提示词、优先级、触发词与标签。"""

    name: str
    prompt: str
    priority: int = 50
    triggers: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    path: str = ""
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "priority": self.priority,
            "triggers": self.triggers,
            "tags": self.tags,
            "path": self.path,
            "enabled": self.enabled,
        }


@dataclass(slots=True)
class RetrievalHit:
    """检索命中：一条记忆痕迹及其相关分数与分数分解。"""

    trace: MemoryTrace
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace.to_dict(),
            "score": self.score,
            "score_breakdown": self.score_breakdown,
        }


@dataclass(slots=True)
class BuildContextResult:
    """构建上下文结果：人格提示、关系提示、用户说明书摘要、召回记忆与触发的程序规则。"""

    persona_prompt: str = ""
    relationship_prompt: str = ""
    user_manual_summary: str = ""
    top_memories: list[RetrievalHit] = field(default_factory=list)
    triggered_procedures: list[ProcedureRule] = field(default_factory=list)
    prompt_sections_capped: list[str] | None = None

    def prompt_sections(self) -> list[str]:
        sections: list[str] = []
        if self.persona_prompt:
            sections.append(f"[人格基线]\n{self.persona_prompt}")
        if self.relationship_prompt:
            sections.append(f"[关系状态]\n{self.relationship_prompt}")
        if self.user_manual_summary:
            sections.append(f"[用户说明书]\n{self.user_manual_summary}")
        if self.top_memories:
            memory_text = "\n".join(
                f"- {hit.trace.summary}"
                for hit in self.top_memories
            )
            sections.append(f"[记忆召回]\n{memory_text}")
        if self.triggered_procedures:
            policy_text = "\n\n".join(
                f"[{rule.name}]\n{rule.prompt}" for rule in self.triggered_procedures
            )
            sections.append(f"[响应策略]\n{policy_text}")
        return sections

    def prompt_sections_with_budget(self, max_tokens: int) -> list[str]:
        from arm_memory.utils import estimate_tokens

        all_sections = self.prompt_sections()
        result: list[str] = []
        used = 0
        for section in all_sections:
            cost = estimate_tokens(section)
            if used + cost > max_tokens:
                remaining = max_tokens - used
                if remaining > 20:
                    truncated = section[: remaining * 2]
                    result.append(truncated)
                break
            result.append(section)
            used += cost
        return result

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "persona_prompt": self.persona_prompt,
            "relationship_prompt": self.relationship_prompt,
            "user_manual_summary": self.user_manual_summary,
            "top_memories": [item.to_dict() for item in self.top_memories],
            "triggered_procedures": [item.to_dict() for item in self.triggered_procedures],
        }
        if self.prompt_sections_capped is not None:
            d["prompt_sections"] = self.prompt_sections_capped
        return d


@dataclass(slots=True)
class UserManualSnapshot:
    """用户说明书快照：某用户在某时刻的说明书全文及更新时间。"""

    project_id: str
    user_id: str
    content: str
    updated_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "content": self.content,
            "updated_at": to_iso(self.updated_at),
        }


@dataclass(slots=True)
class ConsolidationResult:
    """整合结果：各记忆操作计数、情景/语义数量、触发的程序、用户说明书是否更新及备注。"""

    operations: dict[MemoryOperation, int] = field(
        default_factory=lambda: {op: 0 for op in MemoryOperation}
    )
    episodic_count: int = 0
    semantic_count: int = 0
    triggered_procedures: list[str] = field(default_factory=list)
    user_manual_updated: bool = False
    notes: list[str] = field(default_factory=list)

    def bump(self, operation: MemoryOperation) -> None:
        self.operations[operation] = self.operations.get(operation, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "operations": {op.value: count for op, count in self.operations.items()},
            "episodic_count": self.episodic_count,
            "semantic_count": self.semantic_count,
            "triggered_procedures": self.triggered_procedures,
            "user_manual_updated": self.user_manual_updated,
            "notes": self.notes,
        }


def relationship_stage_from_value(value: str | None) -> RelationshipStage:
    if not value:
        return RelationshipStage.INITIAL_CONTACT
    for member in RelationshipStage:
        if value == member.value or value == member.name:
            return member
    return RelationshipStage.INITIAL_CONTACT


def sensitivity_from_value(value: str | None) -> SensitivityLevel:
    if not value:
        return SensitivityLevel.PRIVATE
    for member in SensitivityLevel:
        if value == member.value or value == member.name:
            return member
    return SensitivityLevel.PRIVATE


def trace_from_dict(payload: dict[str, Any]) -> MemoryTrace:
    return MemoryTrace(
        trace_id=payload.get("trace_id") or uuid4().hex,
        project_id=payload["project_id"],
        user_id=payload["user_id"],
        kind=MemoryKind(payload.get("kind", MemoryKind.EPISODIC.value)),
        summary=payload.get("summary", ""),
        raw_text=payload.get("raw_text", ""),
        status=payload.get("status", "active"),
        salience=float(payload.get("salience", 0.5)),
        access_count=int(payload.get("access_count", 0)),
        source_turn_ids=list(payload.get("source_turn_ids") or []),
        tags=list(payload.get("tags") or []),
        entities=list(payload.get("entities") or []),
        vector=list(payload.get("vector") or []),
        emotion=EmotionVector.from_dict(payload.get("emotion")),
        sensitivity=sensitivity_from_value(payload.get("sensitivity")),
        created_at=ensure_utc(datetime.fromisoformat(payload["created_at"])) if payload.get("created_at") else utcnow(),
        updated_at=ensure_utc(datetime.fromisoformat(payload["updated_at"])) if payload.get("updated_at") else utcnow(),
        happened_at=ensure_utc(datetime.fromisoformat(payload["happened_at"])) if payload.get("happened_at") else utcnow(),
        last_accessed_at=ensure_utc(datetime.fromisoformat(payload["last_accessed_at"])) if payload.get("last_accessed_at") else utcnow(),
        metadata=dict(payload.get("metadata") or {}),
    )


def profile_from_dict(payload: dict[str, Any]) -> UserProfile:
    return UserProfile(
        project_id=payload["project_id"],
        user_id=payload["user_id"],
        display_name=payload.get("display_name", ""),
        identity_notes=list(payload.get("identity_notes") or []),
        preferences=list(payload.get("preferences") or []),
        dislikes=list(payload.get("dislikes") or []),
        boundaries=list(payload.get("boundaries") or []),
        vulnerabilities=list(payload.get("vulnerabilities") or []),
        support_preferences=list(payload.get("support_preferences") or []),
        goals=list(payload.get("goals") or []),
        important_people=list(payload.get("important_people") or []),
        important_places=list(payload.get("important_places") or []),
        manual_notes=list(payload.get("manual_notes") or []),
        personality_traits=list(payload.get("personality_traits") or []),
        hobbies=list(payload.get("hobbies") or []),
        habits=list(payload.get("habits") or []),
        communication_style=list(payload.get("communication_style") or []),
        attachment_style=list(payload.get("attachment_style") or []),
        life_routines=list(payload.get("life_routines") or []),
        updated_at=ensure_utc(datetime.fromisoformat(payload["updated_at"])) if payload.get("updated_at") else utcnow(),
    )


def profile_facet_type_from_value(value: str | None) -> ProfileFacetType:
    if not value:
        return ProfileFacetType.MANUAL_NOTES
    for member in ProfileFacetType:
        if value == member.value or value == member.name:
            return member
    return ProfileFacetType.MANUAL_NOTES


def profile_item_status_from_value(value: str | None) -> ProfileItemStatus:
    if not value:
        return ProfileItemStatus.ACTIVE
    for member in ProfileItemStatus:
        if value == member.value or value == member.name:
            return member
    return ProfileItemStatus.ACTIVE


def profile_item_from_dict(payload: dict[str, Any]) -> UserProfileItem:
    return UserProfileItem(
        item_id=payload.get("item_id") or uuid4().hex,
        project_id=payload["project_id"],
        user_id=payload["user_id"],
        facet_type=profile_facet_type_from_value(payload.get("facet_type")),
        value=str(payload.get("value") or "").strip(),
        confidence=float(payload.get("confidence", 0.7)),
        source=str(payload.get("source") or "unknown"),
        source_trace_id=payload.get("source_trace_id") or None,
        first_seen_at=ensure_utc(datetime.fromisoformat(payload["first_seen_at"])) if payload.get("first_seen_at") else utcnow(),
        last_seen_at=ensure_utc(datetime.fromisoformat(payload["last_seen_at"])) if payload.get("last_seen_at") else utcnow(),
        evidence_count=max(1, int(payload.get("evidence_count", 1))),
        status=profile_item_status_from_value(payload.get("status")),
        metadata=dict(payload.get("metadata") or {}),
    )


def persona_from_dict(payload: dict[str, Any]) -> PersonaDefinition:
    return PersonaDefinition(
        persona_id=payload.get("persona_id") or uuid4().hex,
        project_id=payload["project_id"],
        user_id=payload["user_id"],
        name=str(payload.get("name") or "").strip(),
        prompt=str(payload.get("prompt") or "").strip(),
        policy_pack=str(payload.get("policy_pack") or "default"),
        description=str(payload.get("description") or ""),
        is_enabled=bool(payload.get("is_enabled", True)),
        is_active=bool(payload.get("is_active", False)),
        source_path=str(payload.get("source_path") or ""),
        created_at=ensure_utc(datetime.fromisoformat(payload["created_at"])) if payload.get("created_at") else utcnow(),
        updated_at=ensure_utc(datetime.fromisoformat(payload["updated_at"])) if payload.get("updated_at") else utcnow(),
    )


def relationship_from_dict(payload: dict[str, Any]) -> RelationshipState:
    return RelationshipState(
        project_id=payload["project_id"],
        user_id=payload["user_id"],
        intimacy_level=float(payload.get("intimacy_level", 20.0)),
        trust_score=float(payload.get("trust_score", 50.0)),
        attachment_style_guess=payload.get("attachment_style_guess", "unknown"),
        current_stage=relationship_stage_from_value(payload.get("current_stage")),
        recent_conflict_level=float(payload.get("recent_conflict_level", 0.0)),
        boundary_flags=list(payload.get("boundary_flags") or []),
        last_emotional_peak_at=ensure_utc(datetime.fromisoformat(payload["last_emotional_peak_at"])) if payload.get("last_emotional_peak_at") else None,
        notes=list(payload.get("notes") or []),
        behavior_state=BehaviorState.from_dict(payload.get("behavior_state")),
        updated_at=ensure_utc(datetime.fromisoformat(payload["updated_at"])) if payload.get("updated_at") else utcnow(),
    )
