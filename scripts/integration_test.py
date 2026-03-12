#!/usr/bin/env python3
"""
集成测试：用 mock 数据依次调用 ingest_turn、apply_extraction、build_context、
get_summary、get_relationship_state，并打印完整输入/输出供 README 实测 log 使用。

运行方式（项目根目录）：
  python scripts/integration_test.py

- 若已配置 .env 且 Qdrant、Neo4j、Embedding 均已启动，将使用真实环境（完整数据库逻辑）。
- 否则自动退化为简化环境（StubVectorizer + 本地 SQLite，数据落在 data/integration_test/）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 项目根与 src
_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# 尽早加载 .env，便于优先尝试真实环境
try:
    from dotenv import load_dotenv
    load_dotenv(_root / ".env")
except ImportError:
    pass

from arm_memory.config import ARMConfig
from arm_memory.service import ARMMemoryService
from arm_memory.stores import Neo4jGraphStore, QdrantVectorStore
from arm_memory.stores.sqlite_store import SQLiteMemoryStore

# Stub 向量化，与 tests/conftest 一致
from collections import Counter


class StubVectorizer:
    dimensions = 8

    def tokenize(self, text: str) -> list[str]:
        return [t.strip().lower() for t in text.split() if t.strip()]

    def embed(self, text: str) -> list[float]:
        counts = Counter(self.tokenize(text))
        vocab = ["music", "coffee", "support", "calm", "repair", "run", "sleep", "focus"]
        return [float(counts.get(t, 0)) for t in vocab]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def healthcheck(self) -> bool:
        return True


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _io(label: str, obj: dict | list) -> None:
    print(f"\n--- {label} ---")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _make_simplified_service():
    """构建简化环境：StubVectorizer + 本地 SQLite，不依赖 Qdrant/Neo4j。"""
    data_dir = _root / "data" / "integration_test"
    data_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = data_dir / "arm_memory.db"
    config = ARMConfig(
        base_dir=_root,
        data_dir=data_dir,
        sqlite_path=sqlite_path,
        policies_dir=_root / "policies",
        companion_profile_path=_root / "personas" / "companion_profile.md",
        qdrant_url="",
        neo4j_url="",
        neo4j_user="",
        neo4j_password="",
        require_qdrant=False,
        require_neo4j=False,
    )
    vectorizer = StubVectorizer()
    sqlite_store = SQLiteMemoryStore(sqlite_path)
    qdrant_store = QdrantVectorStore(url="", collection=config.qdrant_collection, vector_size=vectorizer.dimensions)
    neo4j_store = Neo4jGraphStore(url="", user="", password="")
    return ARMMemoryService(
        config=config,
        sqlite_store=sqlite_store,
        qdrant_store=qdrant_store,
        neo4j_store=neo4j_store,
        vectorizer=vectorizer,
    )


def main() -> None:
    # 优先使用真实环境（Qdrant + Neo4j + 真实 Embedding），否则退化为简化环境
    try:
        service = ARMMemoryService.from_env(base_dir=_root)
        if not service.vectorizer.healthcheck():
            raise RuntimeError("Embedding 未就绪")
        full_env = True
    except (ValueError, RuntimeError) as e:
        full_env = False
        print("未使用真实环境（Qdrant/Neo4j/Embedding 未配置或不可达），改用简化环境运行。")
        print("  原因:", str(e).split("\n")[0])
        service = _make_simplified_service()

    mode_note = "真实环境（完整数据库逻辑：Qdrant + Neo4j + Embedding）" if full_env else "简化环境（仅 SQLite + Stub 向量）"
    print("\n" + "=" * 60)
    print("集成测试运行模式:", mode_note)
    print("=" * 60)

    project_id = "proj_demo"
    user_id = "user_demo"
    session_id = "session_001"

    # ---------- 1. ingest_turn：用户消息 ----------
    _section("1. ingest_turn（写入用户一轮）")
    payload_user = {
        "project_id": project_id,
        "user_id": user_id,
        "role": "user",
        "content": "最近工作压力大，晚上总失眠，想找人说说话。",
        "session_id": session_id,
        "metadata": {},
    }
    _io("请求 payload", payload_user)
    turn_id_user = service.ingest_turn(
        project_id=payload_user["project_id"],
        user_id=payload_user["user_id"],
        role=payload_user["role"],
        content=payload_user["content"],
        session_id=payload_user["session_id"],
        metadata=payload_user["metadata"],
    )
    _io("响应", {"turn_id": turn_id_user})

    # ---------- 2. ingest_turn：助手回复 ----------
    _section("2. ingest_turn（写入助手一轮）")
    payload_assistant = {
        "project_id": project_id,
        "user_id": user_id,
        "role": "assistant",
        "content": "我在这儿呢。压力大的时候有人听会好很多，你愿意说说具体是什么事让你睡不好吗？",
        "session_id": session_id,
        "metadata": {},
    }
    _io("请求 payload", payload_assistant)
    turn_id_assistant = service.ingest_turn(
        project_id=payload_assistant["project_id"],
        user_id=payload_assistant["user_id"],
        role=payload_assistant["role"],
        content=payload_assistant["content"],
        session_id=payload_assistant["session_id"],
        metadata=payload_assistant["metadata"],
    )
    _io("响应", {"turn_id": turn_id_assistant})

    # ---------- 3. apply_extraction：写入结构化抽取结果 ----------
    _section("3. apply_extraction（写入情景/语义/画像/关系）")
    extraction = {
        "episodic": [
            {"summary": "用户提到工作压力大、失眠，想倾诉", "salience": 0.85},
        ],
        "semantic": [
            {"subject": "user", "predicate": "prefers", "object": "有人倾听", "summary": "用户偏好在压力大时有人倾听"},
            {"subject": "user", "predicate": "identity", "object": "工作压力大易失眠", "summary": "用户近期状态"},
        ],
        "profile": {"preferences": ["有人倾听"], "identity_notes": ["工作压力大易失眠"]},
        "profile_items": [
            {"facet_type": "preferences", "value": "有人倾听", "confidence": 0.9},
            {"facet_type": "identity_notes", "value": "工作压力大易失眠", "confidence": 0.85},
        ],
        "relationship": {"trust_delta": 2.0, "intimacy_delta": 1.0, "behavior_state": {"boundary_stage": "calm"}},
        "procedures": ["comfort_user"],
        "safety_flags": [],
    }
    _io("请求 extraction", extraction)
    turn_ids = [turn_id_user, turn_id_assistant]
    result = service.apply_extraction(
        project_id=project_id,
        user_id=user_id,
        extraction=extraction,
        turn_ids=turn_ids,
        persona_prompt="",
    )
    _io("响应 ConsolidationResult", result.to_dict())

    # ---------- 4. build_context：根据当前消息组上下文 ----------
    _section("4. build_context（根据当前用户消息召回记忆与策略）")
    build_payload = {
        "project_id": project_id,
        "user_id": user_id,
        "message": "还是睡不好，你能陪我聊会儿吗？",
    }
    _io("请求 payload", build_payload)
    ctx = service.build_context(
        project_id=build_payload["project_id"],
        user_id=build_payload["user_id"],
        message=build_payload["message"],
    )
    # 精简输出：persona 可能很长，只保留前 200 字符
    out = ctx.to_dict()
    if out.get("persona_prompt") and len(out["persona_prompt"]) > 200:
        out["persona_prompt"] = out["persona_prompt"][:200] + "..."
    if out.get("user_manual_summary") and len(out["user_manual_summary"]) > 300:
        out["user_manual_summary"] = out["user_manual_summary"][:300] + "..."
    _io("响应 BuildContextResult（persona/user_manual 已截断）", out)

    # ---------- 5. get_summary ----------
    _section("5. get_summary（服务与存储摘要）")
    summary = service.get_summary(project_id=project_id, user_id=user_id)
    _io("响应", summary)

    # ---------- 6. get_relationship_state ----------
    _section("6. get_relationship_state（当前关系状态）")
    state = service.get_relationship_state(project_id=project_id, user_id=user_id)
    _io("响应", state.to_dict())

    print("\n" + "=" * 60)
    print("集成测试完成。以上为完整输入/输出 log（运行模式: %s）。" % mode_note)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
