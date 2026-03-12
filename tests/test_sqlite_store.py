from __future__ import annotations

from arm_memory.stores.sqlite_store import SQLiteMemoryStore


def test_remote_sync_outbox_stops_claiming_after_three_failures(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "arm_memory.db")
    task_id = store.enqueue_remote_sync_task(
        backend="qdrant",
        operation="upsert_trace",
        project_id="proj",
        user_id="user",
        payload={"trace_id": "trace-1"},
        error_message="initial",
    )

    for attempt in range(3):
        claimed = store.claim_remote_sync_tasks(limit=10)
        assert [item["task_id"] for item in claimed] == [task_id]
        store.mark_remote_sync_task_failed(task_id, error_message=f"attempt-{attempt}")

    assert store.claim_remote_sync_tasks(limit=10) == []
    summary = store.get_remote_sync_outbox_summary()
    assert summary["failed"] == 1
    assert summary["pending"] == 0


def test_remote_sync_outbox_done_updates_summary(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "arm_memory.db")
    task_id = store.enqueue_remote_sync_task(
        backend="neo4j",
        operation="sync_edge",
        project_id="proj",
        user_id="user",
        payload={"edge_id": "edge-1"},
        error_message="initial",
    )

    claimed = store.claim_remote_sync_tasks(limit=10)
    assert [item["task_id"] for item in claimed] == [task_id]
    store.mark_remote_sync_task_done(task_id)

    summary = store.get_remote_sync_outbox_summary()
    assert summary["done"] == 1
    assert summary["in_progress"] == 0
