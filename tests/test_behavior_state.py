from __future__ import annotations

from datetime import timedelta

from arm_memory.domain.models import BehaviorState, BoundaryStage, RelationshipState
from arm_memory.utils import utcnow


def test_behavior_state_normalize_steps_down_after_cooldown():
    state = BehaviorState(
        boundary_stage=BoundaryStage.COOLDOWN,
        violation_count=2,
        cooldown_until=utcnow() - timedelta(minutes=1),
    )

    state.normalize()

    assert state.cooldown_until is None
    assert state.boundary_stage == BoundaryStage.REFUSAL


def test_relationship_apply_delta_merges_flags_notes_and_behavior():
    relationship = RelationshipState(project_id="proj", user_id="user")

    relationship.apply_delta(
        trust_delta=-5.0,
        conflict_target=55.0,
        boundary_flags=["boundary_tension", "boundary_tension"],
        notes=["需要修复", "需要修复"],
        behavior_state={
            "boundary_stage": "warning",
            "violation_count": 1,
            "recent_flags": ["warning"],
        },
    )

    assert relationship.trust_score == 45.0
    assert relationship.recent_conflict_level == 55.0
    assert relationship.boundary_flags == ["boundary_tension"]
    assert relationship.notes == ["需要修复"]
    assert relationship.behavior_state.boundary_stage == BoundaryStage.WARNING
    assert relationship.behavior_state.violation_count == 1
