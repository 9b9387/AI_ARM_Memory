# ARM Memory（Adaptive Relational Memory）

## 从 AI 伴侣记忆的痛点说起

AI 伴侣要像「真人」一样长期相处，离不开**记忆**：记得用户说过什么、偏好与边界、关系亲疏、以及哪些事不该再提。但常见实现里往往存在这些痛点：

- **记不住**：对话一刷新就清空，或只靠简短上下文窗口，无法形成长期、可检索的记忆。
- **记不准**：没有结构化沉淀，只能靠模型在有限 token 里「猜」用户是谁，容易张冠李戴或遗忘关键信息。
- **用不上**：即便有数据，也缺少「按当前这句话该召回什么、该用哪条策略」的检索与组装，回复缺乏个性与边界感。
- **关系无状态**：亲密度、信任、冲突、冷却等关系状态若只存在提示词里，难以持久、难以随互动更新，也难以驱动策略切换。

---

## 本服务能解决的问题与方案

**ARM Memory Service** 面向上述痛点，做一件事：**把「记忆与关系」做成可持久、可检索、可组装的独立服务**，让上游只关心对话与生成，记忆的存、取、更新由本服务统一负责。


| 痛点    | 本服务提供的方案                                                                                        |
| ----- | ----------------------------------------------------------------------------------------------- |
| 记不住   | 按用户/项目隔离存储对话轮次与记忆痕迹；支持向量（Qdrant）+ 图（Neo4j）+ 词法 + 时间衰减的混合检索，长期记忆可被稳定召回。                          |
| 记不准   | 由上游完成「对话 → 结构化抽取」后，通过 **apply_extraction** 写入情景/语义记忆、用户画像与关系状态；本服务保证这些结构化数据被正确落库并参与后续检索与上下文组装。  |
| 用不上   | **build_context** 根据当前用户消息做检索与策略匹配，输出人格提示、关系提示、用户说明书、相关记忆与触发的程序规则，供上游一次性拼进 prompt，生成有记忆、有边界的回复。 |
| 关系无状态 | 关系状态（亲密度、信任、冲突、边界与行为阶段）持久化并可更新；与策略规则、用户说明书结合，驱动「该安慰 / 该保持距离 / 该修复信任」等行为选择。                      |


**边界说明**：**对话 → 结构化抽取**（即从原始对话中抽出记忆、画像、关系变化）由调用方在项目外完成（如自建 LLM 或编排服务）；本服务只负责**接收已生成的结构化数据并写入**、检索与组 context，不负责对话生成与抽取本身。

---

## 一、项目设计

### 1.1 定位与职责

- **记忆存储**：按 `project_id` / `user_id` 隔离，持久化对话轮次（turns）、记忆痕迹（traces）、语义事实（facts）、用户画像（profile items）、关系状态（relationship）、人格定义（personas）等。
- **接收抽取结果并写入**：调用方在项目外完成「对话 → 结构化抽取」后，通过 **apply_extraction** 将结果（episodic、semantic、profile、relationship、procedures 等）写入 SQLite + 向量库（Qdrant）+ 图库（Neo4j）。
- **检索与上下文**：根据当前用户消息做混合检索（向量 + 词法 + 图 + 显著性 + 时间衰减），召回相关记忆，并结合人格基线、关系状态、用户说明书、策略规则，组装成 **build_context** 供上游生成回复。

服务**不负责**对话生成与**不负责**对话→抽取；只负责「接收并写入、查什么、怎么组 context」。

### 1.2 技术方案概览


| 层级     | 技术选型                  | 说明                                                                 |
| ------ | --------------------- | ------------------------------------------------------------------ |
| 运行入口   | `arm-memory`（uvicorn） | 默认端口 8788，WebSocket 路径 `/ws`                                       |
| 协议     | WebSocket             | 单一接口，见下文调用示例                                                       |
| 主存储    | SQLite                | 对话、记忆、画像、关系、人格、用户说明书等                                              |
| 向量检索   | Qdrant                | 记忆向量索引，**必须已启动**                                                   |
| 图存储    | Neo4j                 | 语义事实与关系图，**必须已启动**                                                 |
| 嵌入     | 可配置（如 MLX / 远程 API）   | 用于 trace 向量化与检索                                                    |
| 写入抽取结果 | apply_extraction      | 接收外部结构化 JSON（episodic/semantic/profile/relationship/procedures）并写入 |


### 1.3 核心数据概念

- **ConversationTurn**：单轮对话（角色、内容、会话 ID、时间戳）。
- **MemoryTrace**：一条记忆痕迹，含类型、摘要、原文、显著性、向量、情绪、敏感度等。
- **SemanticFact**：主谓宾三元组（如「用户偏好咖啡」），可写入 Neo4j。
- **UserProfile / UserProfileItem**：用户画像聚合与单条维度（偏好、边界、习惯等）。
- **RelationshipState**：亲密度、信任分、关系阶段、边界与行为状态（含冷却、冒犯计数等）。
- **PersonaDefinition**：人格定义（名称、prompt、策略包、是否激活）。
- **BuildContextResult**：组装好的上下文（人格提示、关系提示、用户说明书、召回记忆、触发的程序规则）。

---

## 二、运行方式

### 2.1 前置条件：Qdrant 与 Neo4j 必须已启动

**启动 Qdrant（示例，默认 6333 端口，挂载到项目本地目录持久化）：**

```bash
mkdir -p ./data/qdrant
docker run -d --name qdrant \
  -p 6333:6333 \
  -v "$(pwd)/data/qdrant:/qdrant/storage" \
  qdrant/qdrant
```

配置环境变量：`ARM_QDRANT_URL=http://127.0.0.1:6333`（或你的 Qdrant 地址）。

**启动 Neo4j（示例，默认 7474/7687，挂载到项目本地目录持久化）：**

```bash
mkdir -p ./data/neo4j/data ./data/neo4j/logs ./data/neo4j/plugins ./data/neo4j/import
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v "$(pwd)/data/neo4j/data:/data" \
  -v "$(pwd)/data/neo4j/logs:/logs" \
  -v "$(pwd)/data/neo4j/plugins:/plugins" \
  -v "$(pwd)/data/neo4j/import:/import" \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:latest
```

配置环境变量：`ARM_NEO4J_URL=http://127.0.0.1:7474`、`ARM_NEO4J_USER=neo4j`、`ARM_NEO4J_PASSWORD=your-password`。

上述目录均以**项目根目录**为基准；这样容器重启或重建后，Qdrant 与 Neo4j 的数据仍会保留在本地 `./data/` 下。

### 2.2 启动

在项目根目录执行：

```bash
# 默认 127.0.0.1:8788，WebSocket /ws
python main.py

# 或指定 host/port
python main.py --host 0.0.0.0 --port 8788
```

环境变量见 `.env.example`，主要包括：`ARM_QDRANT_URL`、`ARM_NEO4J_*`、`ARM_DATA_DIR`、`ARM_SQLITE_PATH`、`ARM_EMBEDDING_*` 等。未配置 Qdrant/Neo4j 或服务不可达时，进程会退出并提示上述启动方法。

服务启动后会同时：

- 在控制台输出启动、依赖检查、WebSocket 请求、后台 outbox 重放等关键信息；
- 将日志写入本地文件，默认路径为 `./data/arm_memory/logs/arm_memory.log`；
- 底层日志实现使用 `loguru`，控制台输出更清晰，并支持文件轮转；
- 支持通过 `ARM_LOG_LEVEL`、`ARM_LOG_DIR`、`ARM_LOG_PATH` 自定义日志级别与落盘位置。

---

## 三、WebSocket 接口

连接地址：`ws://<host>:<port><service_ws_path>`，默认 `ws://127.0.0.1:8788/ws`。

### 3.1 消息格式

- **请求**：JSON 对象，字段 `type`、`request_id`、`action`、`payload`。
- **响应**：JSON，`type: "response"`，`request_id`、`action`、`status`、`data`。
- **事件**：`type: "event"`（当前未使用；apply_extraction 同步返回结果）。
- **错误**：`type: "error"`，内含 `error.code`、`error.message`、`error.details`。

以下示例中 `request_id` 可替换为任意字符串，便于对号入座。

---

### 3.2 ping

探测服务与依赖是否可用。

**请求：**

```json
{
  "type": "request",
  "request_id": "req-ping-001",
  "action": "ping",
  "payload": {}
}
```

**响应：**

```json
{
  "type": "response",
  "request_id": "req-ping-001",
  "action": "ping",
  "status": "ok",
  "data": {
    "status": "ok",
    "service": "arm_memory",
    "ws_path": "/ws",
    "qdrant_enabled": true,
    "neo4j_enabled": false
  }
}
```

---

### 3.3 build_context

根据当前用户消息组装回复上下文（人格、关系、用户说明书、召回记忆、触发的策略）。

**请求：**

```json
{
  "type": "request",
  "request_id": "req-bc-001",
  "action": "build_context",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "message": "我最近工作压力好大，晚上总失眠"
  }
}
```

**响应：**

```json
{
  "type": "response",
  "request_id": "req-bc-001",
  "action": "build_context",
  "status": "ok",
  "data": {
    "persona_prompt": "你是温柔、稳定的伴侣角色…",
    "relationship_prompt": "亲密度 35.0 / 100，信任 52.0 / 100；阶段：ShortTermInteraction；冲突水平 10.0。",
    "user_manual_summary": "# 用户说明书\n\n## 身份与偏好\n…",
    "top_memories": [
      {
        "trace": {
          "trace_id": "abc123",
          "project_id": "my_app",
          "user_id": "user_123",
          "kind": "episodic",
          "summary": "用户提到工作压力大、失眠",
          "raw_text": "最近加班多，睡不好",
          "status": "active",
          "salience": 0.75,
          "access_count": 2,
          "tags": ["work", "sleep"],
          "created_at": "2025-03-12T10:00:00Z",
          "updated_at": "2025-03-12T10:05:00Z"
        },
        "score": 0.82,
        "score_breakdown": {
          "semantic": 0.35,
          "lexical": 0.12,
          "salience": 0.15,
          "temporal": 0.20
        }
      }
    ],
    "triggered_procedures": [
      {
        "name": "comfort_user",
        "prompt": "先共情再提供支持…",
        "priority": 60,
        "triggers": ["焦虑", "压力", "难过"],
        "tags": ["support"],
        "path": "",
        "enabled": true
      }
    ]
  }
}
```

---

### 3.4 autonomy_snapshot

获取一次「自主性快照」：当前人格、关系、用户说明书、策略规则、近期对话与记忆，便于离线或定时决策。

**请求：**

```json
{
  "type": "request",
  "request_id": "req-auto-001",
  "action": "autonomy_snapshot",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "query": "系统心跳：评估是否需要主动行动",
    "turns_limit": 20,
    "traces_limit": 12
  }
}
```

**响应：**

```json
{
  "type": "response",
  "request_id": "req-auto-001",
  "action": "autonomy_snapshot",
  "status": "ok",
  "data": {
    "project_id": "my_app",
    "user_id": "user_123",
    "persona_prompt": "你是…",
    "active_persona": {
      "persona_id": "p-default",
      "project_id": "my_app",
      "user_id": "user_123",
      "name": "默认伴侣",
      "prompt": "…",
      "policy_pack": "default",
      "is_enabled": true,
      "is_active": true,
      "created_at": "2025-03-01T00:00:00Z",
      "updated_at": "2025-03-12T00:00:00Z"
    },
    "relationship": {
      "project_id": "my_app",
      "user_id": "user_123",
      "intimacy_level": 35.0,
      "trust_score": 52.0,
      "current_stage": "ShortTermInteraction",
      "recent_conflict_level": 10.0,
      "boundary_flags": [],
      "notes": [],
      "behavior_state": {
        "boundary_stage": "calm",
        "violation_count": 0,
        "cooldown_until": null,
        "recent_flags": []
      },
      "updated_at": "2025-03-12T10:00:00Z"
    },
    "user_manual": {
      "project_id": "my_app",
      "user_id": "user_123",
      "content": "# 用户说明书\n\n…",
      "updated_at": "2025-03-12T09:00:00Z"
    },
    "policy_rules": [
      {
        "name": "comfort_user",
        "prompt": "…",
        "priority": 60,
        "triggers": ["焦虑", "难过"],
        "tags": ["support"],
        "enabled": true
      }
    ],
    "recent_turns": [
      {
        "turn_id": "t1",
        "project_id": "my_app",
        "user_id": "user_123",
        "session_id": "sess-1",
        "role": "user",
        "content": "最近睡不好",
        "created_at": "2025-03-12T10:00:00Z",
        "metadata": {}
      }
    ],
    "recent_traces": [
      {
        "trace_id": "abc123",
        "project_id": "my_app",
        "user_id": "user_123",
        "kind": "episodic",
        "summary": "用户提到工作压力大、失眠",
        "status": "active",
        "salience": 0.75,
        "created_at": "2025-03-12T10:00:00Z"
      }
    ],
    "build_context": {
      "persona_prompt": "…",
      "relationship_prompt": "…",
      "user_manual_summary": "…",
      "top_memories": [],
      "triggered_procedures": []
    }
  }
}
```

---

### 3.5 ingest_turn

写入一条对话轮次（用户或助手）。

**请求：**

```json
{
  "type": "request",
  "request_id": "req-ingest-001",
  "action": "ingest_turn",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "role": "user",
    "content": "我明天想去看电影，你有推荐吗？",
    "session_id": "sess-20250312",
    "metadata": {}
  }
}
```

**响应：**

```json
{
  "type": "response",
  "request_id": "req-ingest-001",
  "action": "ingest_turn",
  "status": "ok",
  "data": {
    "turn_id": "a1b2c3d4e5f6"
  }
}
```

---

### 3.6 manual_remember

用户主动要求「记住」一段话，写入为记忆并更新用户说明书。

**请求：**

```json
{
  "type": "request",
  "request_id": "req-manual-001",
  "action": "manual_remember",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "text": "请记住：我对芒果过敏，不要推荐芒果相关食物。",
    "tags": ["allergy", "food"],
    "sensitivity": "private"
  }
}
```

**响应：**

```json
{
  "type": "response",
  "request_id": "req-manual-001",
  "action": "manual_remember",
  "status": "ok",
  "data": {
    "operation": "ADD"
  }
}
```

`sensitivity` 可选：`public`、`private`、`sensitive`、`restricted`。

---

### 3.7 apply_extraction

接收**外部已生成**的结构化抽取结果并写入记忆/画像/关系。对话→抽取由调用方在项目外完成（如自建 LLM），本接口只做写入。

**请求：**

```json
{
  "type": "request",
  "request_id": "req-apply-001",
  "action": "apply_extraction",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "extraction": {
      "episodic": [
        {"summary": "用户提到工作压力大、失眠", "salience": 0.75, "sensitivity": "private"}
      ],
      "semantic": [
        {"subject": "user", "predicate": "prefers", "object": "安静陪伴", "summary": "用户偏好安静陪伴", "confidence": 0.85}
      ],
      "profile": {"preferences": ["安静陪伴"]},
      "profile_items": [
        {"facet_type": "preferences", "value": "安静陪伴", "confidence": 0.85, "source": "external"}
      ],
      "relationship": {"intimacy_delta": 2.0, "trust_delta": 1.0, "behavior_state": {"boundary_stage": "calm"}},
      "procedures": ["comfort_user"],
      "safety_flags": []
    },
    "turn_ids": ["t1", "t2"],
    "session_id": "sess-20250312",
    "persona_prompt": ""
  }
}
```

- **extraction**（必填）：与既往 LLM 抽取输出同结构的 JSON。若提供 **session_id** 且未提供 **turn_ids**，服务会按该会话解析近期轮次并标记为已巩固。
- **turn_ids**（可选）：要标记为已巩固的轮次 ID 列表。
- **persona_prompt**（可选）：用于生成用户说明书的基线人格文案。

**响应：**

```json
{
  "type": "response",
  "request_id": "req-apply-001",
  "action": "apply_extraction",
  "status": "ok",
  "data": {
    "operations": {"ADD": 3, "UPDATE": 1, "DELETE": 0, "NOOP": 0},
    "episodic_count": 1,
    "semantic_count": 1,
    "triggered_procedures": ["comfort_user"],
    "user_manual_updated": true,
    "notes": []
  }
}
```

---

### 3.8 get_summary

获取该用户下的记忆与关系摘要（统计 + 关系阶段 + 说明书预览等）。

**请求：**

```json
{
  "type": "request",
  "request_id": "req-sum-001",
  "action": "get_summary",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123"
  }
}
```

**响应：**

```json
{
  "type": "response",
  "request_id": "req-sum-001",
  "action": "get_summary",
  "status": "ok",
  "data": {
    "summary": "ARM 记忆摘要\n- 情景记忆：12\n- 语义事实：8\n- 对话轮次：45\n- 画像条目：6\n- Persona 数量：1\n- 当前 Persona：默认伴侣\n- 关系阶段：ShortTermInteraction\n- 亲密度：35.0/100\n- 信任度：52.0/100\n- 冲突水平：10.0/100\n- 边界阶段：calm\n\n近期长期事实：\n- 用户偏好安静陪伴\n- 用户不喜欢被催促\n\n用户说明书预览：\n# 用户说明书\n…"
  }
}
```

---

### 3.9 get_relationship_state / save_relationship_state

读取或保存当前用户的关系状态（亲密度、信任、阶段、边界与行为状态）。

**get 请求：**

```json
{
  "type": "request",
  "request_id": "req-rel-get-001",
  "action": "get_relationship_state",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123"
  }
}
```

**get 响应：**

```json
{
  "type": "response",
  "request_id": "req-rel-get-001",
  "action": "get_relationship_state",
  "status": "ok",
  "data": {
    "project_id": "my_app",
    "user_id": "user_123",
    "intimacy_level": 35.0,
    "trust_score": 52.0,
    "attachment_style_guess": "unknown",
    "current_stage": "ShortTermInteraction",
    "recent_conflict_level": 10.0,
    "boundary_flags": [],
    "last_emotional_peak_at": null,
    "notes": [],
    "behavior_state": {
      "boundary_stage": "calm",
      "violation_count": 0,
      "cooldown_until": null,
      "last_violation_at": null,
      "last_repair_at": null,
      "last_boundary_reason": "",
      "recent_flags": []
    },
    "updated_at": "2025-03-12T10:00:00Z"
  }
}
```

**save 请求：**

```json
{
  "type": "request",
  "request_id": "req-rel-save-001",
  "action": "save_relationship_state",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "state": {
      "project_id": "my_app",
      "user_id": "user_123",
      "intimacy_level": 40.0,
      "trust_score": 55.0,
      "current_stage": "ShortTermInteraction",
      "recent_conflict_level": 5.0,
      "behavior_state": {
        "boundary_stage": "calm",
        "violation_count": 0,
        "recent_flags": []
      }
    }
  }
}
```

**save 响应：**

```json
{
  "type": "response",
  "request_id": "req-rel-save-001",
  "action": "save_relationship_state",
  "status": "ok",
  "data": {
    "status": "saved"
  }
}
```

---

### 3.10 get_recent_turns / get_recent_traces

分页获取近期对话轮次或记忆痕迹。

**get_recent_turns 请求：**

```json
{
  "type": "request",
  "request_id": "req-turns-001",
  "action": "get_recent_turns",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "limit": 20,
    "session_id": "sess-20250312"
  }
}
```

**响应：**

```json
{
  "type": "response",
  "request_id": "req-turns-001",
  "action": "get_recent_turns",
  "status": "ok",
  "data": {
    "turns": [
      {
        "turn_id": "t1",
        "project_id": "my_app",
        "user_id": "user_123",
        "session_id": "sess-20250312",
        "role": "user",
        "content": "明天想去看电影",
        "created_at": "2025-03-12T10:00:00Z",
        "metadata": {}
      },
      {
        "turn_id": "t2",
        "role": "assistant",
        "content": "好的，你更喜欢哪种类型？",
        "created_at": "2025-03-12T10:00:05Z"
      }
    ]
  }
}
```

**get_recent_traces 请求：**

```json
{
  "type": "request",
  "request_id": "req-traces-001",
  "action": "get_recent_traces",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "limit": 20
  }
}
```

**响应：** `data.traces` 为 MemoryTrace 的 `to_dict()` 数组，结构同 `build_context.top_memories[].trace`。

---

### 3.11 get_active_persona / list_personas / save_persona / activate_persona

人格的查询、列表、保存与激活。

**get_active_persona 请求：**

```json
{
  "type": "request",
  "request_id": "req-pers-act-001",
  "action": "get_active_persona",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123"
  }
}
```

**响应：** `data.persona` 为 PersonaDefinition 的 `to_dict()`（见上文 autonomy_snapshot.active_persona）。

**list_personas 请求：**

```json
{
  "type": "request",
  "request_id": "req-pers-list-001",
  "action": "list_personas",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "include_disabled": true
  }
}
```

**响应：** `data.personas` 为 persona 对象数组。

**save_persona 请求：**

```json
{
  "type": "request",
  "request_id": "req-pers-save-001",
  "action": "save_persona",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "persona": {
      "name": "周末陪伴",
      "prompt": "在周末对话时更轻松、偏娱乐向…",
      "policy_pack": "default",
      "description": "周末专用",
      "is_enabled": true,
      "is_active": false
    }
  }
}
```

**响应：** `data.persona` 为保存后的完整 persona（含 `persona_id`、`created_at`、`updated_at`）。

**activate_persona 请求：**

```json
{
  "type": "request",
  "request_id": "req-pers-activate-001",
  "action": "activate_persona",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "persona_id": "p-weekend-001"
  }
}
```

**响应：** `data.persona` 为当前激活的人格。

---

### 3.12 list_profile_items / save_profile_item

用户画像条目的列表与写入。

**list_profile_items 请求：**

```json
{
  "type": "request",
  "request_id": "req-prof-list-001",
  "action": "list_profile_items",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "facet_types": ["preferences", "boundaries"],
    "limit": 200
  }
}
```

**响应：**

```json
{
  "type": "response",
  "request_id": "req-prof-list-001",
  "action": "list_profile_items",
  "status": "ok",
  "data": {
    "items": [
      {
        "item_id": "pi-001",
        "project_id": "my_app",
        "user_id": "user_123",
        "facet_type": "preferences",
        "value": "安静陪伴",
        "confidence": 0.85,
        "source": "llm_profile",
        "first_seen_at": "2025-03-10T12:00:00Z",
        "last_seen_at": "2025-03-12T10:00:00Z",
        "evidence_count": 2,
        "status": "active"
      }
    ]
  }
}
```

**save_profile_item 请求：**

```json
{
  "type": "request",
  "request_id": "req-prof-save-001",
  "action": "save_profile_item",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123",
    "facet_type": "manual_notes",
    "value": "对芒果过敏",
    "confidence": 1.0,
    "source": "manual",
    "source_trace_id": null,
    "metadata": {}
  }
}
```

**响应：** `data.item` 为保存后的画像条目的 `to_dict()`。

---

### 3.13 clear_user_data

清空该用户在本项目下的全部数据（对话、记忆、画像、关系、人格等）。

**请求：**

```json
{
  "type": "request",
  "request_id": "req-clear-001",
  "action": "clear_user_data",
  "payload": {
    "project_id": "my_app",
    "user_id": "user_123"
  }
}
```

**响应：**

```json
{
  "type": "response",
  "request_id": "req-clear-001",
  "action": "clear_user_data",
  "status": "ok",
  "data": {
    "status": "cleared"
  }
}
```

---

### 3.14 replay_remote_sync_outbox

重放远程同步发件箱中的任务（内部用，一般由服务定时执行）。

**请求：**

```json
{
  "type": "request",
  "request_id": "req-outbox-001",
  "action": "replay_remote_sync_outbox",
  "payload": {
    "limit": 50
  }
}
```

**响应：**

```json
{
  "type": "response",
  "request_id": "req-outbox-001",
  "action": "replay_remote_sync_outbox",
  "status": "ok",
  "data": {
    "result": {
      "processed": 3,
      "succeeded": 2,
      "failed": 1
    },
    "summary": {
      "pending": 0,
      "failed": 1,
      "done": 2
    }
  }
}
```

---

### 3.15 错误响应示例

当 action 不存在或执行抛错时，会收到 `type: "error"`：

```json
{
  "type": "error",
  "request_id": "req-xxx",
  "action": "build_context",
  "error": {
    "code": "invalid_request",
    "message": "Unknown websocket action: invalid_action",
    "details": {}
  }
}
```

---

## 四、真实环境实测 log

各 action 的**请求/响应字段定义**见**第三章**，此处不再重复；本节仅提供在**真实环境**（Qdrant、Neo4j、Embedding 均已启动并配置 `.env`）下执行集成脚本的一次完整终端输出，便于对照实际请求与响应。

**运行方式（项目根目录）：**

```bash
python scripts/integration_test.py
```

脚本会优先使用真实环境（完整数据库逻辑：向量写入 Qdrant、关系写入 Neo4j、真实嵌入）；若 Qdrant/Neo4j/Embedding 未配置或不可达，会自动退化为简化环境（仅 SQLite + Stub 向量），输出格式一致。

**以下为真实环境下执行得到的完整 log：**

```text
============================================================
集成测试运行模式: 真实环境（完整数据库逻辑：Qdrant + Neo4j + Embedding）
============================================================

============================================================
1. ingest_turn（写入用户一轮）
============================================================

--- 请求 payload ---
{
  "project_id": "proj_demo",
  "user_id": "user_demo",
  "role": "user",
  "content": "最近工作压力大，晚上总失眠，想找人说说话。",
  "session_id": "session_001",
  "metadata": {}
}

--- 响应 ---
{
  "turn_id": "f5b85e6418f2491c803e7354f124edd4"
}

============================================================
2. ingest_turn（写入助手一轮）
============================================================

--- 请求 payload ---
{
  "project_id": "proj_demo",
  "user_id": "user_demo",
  "role": "assistant",
  "content": "我在这儿呢。压力大的时候有人听会好很多，你愿意说说具体是什么事让你睡不好吗？",
  "session_id": "session_001",
  "metadata": {}
}

--- 响应 ---
{
  "turn_id": "13b79bfecf67470593f40cfa89271da1"
}

============================================================
3. apply_extraction（写入情景/语义/画像/关系）
============================================================

--- 请求 extraction ---
{
  "episodic": [
    {
      "summary": "用户提到工作压力大、失眠，想倾诉",
      "salience": 0.85
    }
  ],
  "semantic": [
    {
      "subject": "user",
      "predicate": "prefers",
      "object": "有人倾听",
      "summary": "用户偏好在压力大时有人倾听"
    },
    {
      "subject": "user",
      "predicate": "identity",
      "object": "工作压力大易失眠",
      "summary": "用户近期状态"
    }
  ],
  "profile": {
    "preferences": ["有人倾听"],
    "identity_notes": ["工作压力大易失眠"]
  },
  "profile_items": [
    {
      "facet_type": "preferences",
      "value": "有人倾听",
      "confidence": 0.9
    },
    {
      "facet_type": "identity_notes",
      "value": "工作压力大易失眠",
      "confidence": 0.85
    }
  ],
  "relationship": {
    "trust_delta": 2.0,
    "intimacy_delta": 1.0,
    "behavior_state": {
      "boundary_stage": "calm"
    }
  },
  "procedures": ["comfort_user"],
  "safety_flags": []
}

--- 响应 ConsolidationResult ---
{
  "operations": {
    "ADD": 3,
    "UPDATE": 0,
    "DELETE": 0,
    "NOOP": 0
  },
  "episodic_count": 1,
  "semantic_count": 2,
  "triggered_procedures": ["comfort_user"],
  "user_manual_updated": true,
  "notes": ["archived_traces=0"]
}

============================================================
4. build_context（根据当前用户消息召回记忆与策略）
============================================================

--- 请求 payload ---
{
  "project_id": "proj_demo",
  "user_id": "user_demo",
  "message": "还是睡不好，你能陪我聊会儿吗？"
}

--- 响应 BuildContextResult（persona/user_manual 已截断） ---
{
  "persona_prompt": "# Companion Profile\n\n你是一个稳定、真诚、富有边界感的虚拟恋人。",
  "relationship_prompt": "- 当前关系阶段：InitialContact\n- 当前亲密度：21.0/100\n- 当前信任度：52.0/100\n- 近期冲突水平：0.0/100\n- 当前边界阶段：calm\n- 回复时要保持连续性、温柔度和清晰边界，不要因讨好而失去人格一致性。",
  "user_manual_summary": "## 用户画像\n- 称呼或身份：工作压力大易失眠\n- 身份信息：工作压力大易失眠\n- 偏好：有人倾听\n\n## 关系状态\n- 阶段：InitialContact\n- 亲密度：21.0/100\n- 信任度：52.0/100\n- 近期冲突：0.0/100\n- 边界阶段：calm\n\n## 长期事实\n- 用户近期状态\n- 用户偏好在压力大时有人倾听\n\n## 近期情景\n- 用户提到工作压力大、失眠，想倾诉\n\n## 使用准则\n- 优先遵守人格基线和关系边界，再结合用户长期事实与近期情景。\n- 遇到高风险、强依赖、操控性请求时，保持温柔但明确的边界。",
  "top_memories": [],
  "triggered_procedures": []
}

============================================================
5. get_summary（服务与存储摘要）
============================================================

--- 响应 ---
"ARM 记忆摘要\n- 情景记忆：1\n- 语义事实：2\n- 对话轮次：2\n- 画像条目：2\n- Persona 数量：1\n- 当前 Persona：companion-default\n- 关系阶段：InitialContact\n- 亲密度：21.0/100\n- 信任度：52.0/100\n- 冲突水平：0.0/100\n- 边界阶段：calm\n\n近期长期事实：\n- 用户近期状态\n- 用户偏好在压力大时有人倾听\n\n用户说明书预览：\n## 用户画像\n- 称呼或身份：工作压力大易失眠\n- 身份信息：工作压力大易失眠\n- 偏好：有人倾听\n\n## 关系状态\n- 阶段：InitialContact\n- 亲密度：21.0/100\n- 信任度：52.0/100\n- 近期冲突：0.0/100\n- 边界阶段：calm\n\n## 长期事实\n- 用户近期状态\n- 用户偏好在压力大时有人倾听\n\n## 近期情景\n- 用户提到工作压力大、失眠，想倾诉\n\n## 使用准则\n- 优先遵守人格基线和关系边界，再结合用户长期事实与近期情景。\n- 遇到高风险、强依赖、操控性请求时，保持温柔但明确的边界。"

============================================================
6. get_relationship_state（当前关系状态）
============================================================

--- 响应 ---
{
  "project_id": "proj_demo",
  "user_id": "user_demo",
  "intimacy_level": 21.0,
  "trust_score": 52.0,
  "attachment_style_guess": "unknown",
  "current_stage": "InitialContact",
  "recent_conflict_level": 0.0,
  "boundary_flags": [],
  "last_emotional_peak_at": null,
  "notes": [],
  "behavior_state": {
    "boundary_stage": "calm",
    "violation_count": 0,
    "cooldown_until": null,
    "last_violation_at": null,
    "last_repair_at": null,
    "last_boundary_reason": "",
    "recent_flags": []
  },
  "updated_at": "2026-03-12T10:52:13.938947+00:00"
}

============================================================
集成测试完成。以上为完整输入/输出 log（运行模式: 真实环境（完整数据库逻辑：Qdrant + Neo4j + Embedding））。
============================================================
```

---

## 五、目录结构（简要）

```
arm_memory/
├── personas/                # 可配置：伴侣人格基线（如 companion_profile.md）
├── policies/                # 可配置：策略规则（.md）
├── src/arm_memory/
│   ├── __init__.py
│   ├── __main__.py          # arm-memory 入口
│   ├── config.py            # 配置（环境变量）
│   ├── consolidation.py     # 接收结构化抽取并写入（apply_extraction）
│   ├── domain/
│   │   ├── models.py        # 领域模型
│   │   └── __init__.py
│   ├── protocol.py          # WebSocket 请求/响应/事件/错误结构
│   ├── retrieval.py         # 混合检索
│   ├── service.py           # 业务门面
│   ├── stores/
│   │   ├── sqlite_store.py
│   │   ├── qdrant_store.py
│   │   ├── neo4j_store.py
│   │   └── __init__.py
│   ├── vectorizer.py        # 嵌入
│   ├── utils.py
│   └── ws_server.py         # WebSocket 路由与分发
├── tests/
├── pyproject.toml
├── .env.example
└── README.md（本文档）
```

---

## 六、License 与贡献

见项目根目录的 LICENSE 与 CONTRIBUTING（如有）。  
接口与数据结构以代码为准，本文档仅作设计与调用示例说明。