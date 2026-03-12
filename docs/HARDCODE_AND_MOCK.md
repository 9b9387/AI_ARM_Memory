# 项目中的 Hardcode 与 Mock 数据检查

本文档汇总项目内**硬编码（hardcode）**与**模拟/测试数据（mock data）**的位置与说明。

---

## 一、Hardcode（硬编码）

### 1.1 配置与网络

| 位置 | 内容 | 说明 |
|------|------|------|
| `config.py` | `service_host: str = "127.0.0.1"` | 服务监听地址默认值，可通过 `ARM_SERVICE_HOST` 覆盖 |
| `config.py` | `service_port: int = 8788` | 服务端口默认值，可通过 `ARM_SERVICE_PORT` 覆盖 |
| `config.py` | `qdrant_collection: str = "arm_memory"` | Qdrant 集合名，未从环境变量读取 |
| `config.py` | `embedding_model: str = "mlx-community/Qwen3-Embedding-0.6B-mxfp8"` | 默认 embedding 模型，可通过 `ARM_EMBEDDING_MODEL` 覆盖 |
| `config.py` | `companion_node_name: str = "companion:self"` | 图/关系中的伴侣节点名，未从环境变量读取 |
| `config.py` | `service_ws_path: str = "/ws"` | WebSocket 路径，未从环境变量读取 |
| `config.py` | 默认路径：`project_root / "personas" / "companion_profile.md"` | 默认伴侣画像文件路径，可通过 `COMPANION_PROFILE_PATH` 覆盖 |

### 1.2 业务默认值与魔法数字

| 位置 | 内容 | 说明 |
|------|------|------|
| `config.py` | `retrieval_top_k=6`, `working_memory_turns=12`, `user_manual_memory_limit=12` | 检索/工作记忆/用户说明书条数上限，均有对应环境变量 |
| `config.py` | `vector_dimensions=1024`, `embedding_timeout=120`, `embedding_batch_size=16`, `embedding_max_length=512` | 向量维度与 embedding 相关，均有环境变量 |
| `config.py` | `profile_item_stale_days=180`, `episodic_half_life_hours=168`, `semantic_half_life_hours=2160` | 画像过期、记忆半衰期等，均有环境变量 |
| `config.py` | `retrieval_weight_*`（如 0.40, 0.15, 0.20…）, `memory_archive_min_salience=0.20`, `memory_merge_similarity_threshold=0.85` | 检索权重与归档/合并阈值，均有环境变量 |
| `ws_server.py` | `_OUTBOX_REPLAY_INTERVAL_SECONDS = 60`, `_OUTBOX_REPLAY_LIMIT = 50` | Outbox 重放间隔（秒）与单次拉取条数，未配置化 |
| `ws_server.py` | `payload.get("turns_limit", 20)`, `payload.get("traces_limit", 12)`, `limit=60`（get_recent_turns）等 | WebSocket 请求中的默认 limit，部分仅来自 payload 默认值 |
| `ws_server.py` | `payload.get("project_id", "default")`, `payload.get("user_id", "default")`, `payload.get("session_id", "default")` | **多处**：当客户端未传 `project_id`/`user_id`/`session_id` 时回退为字符串 `"default"` |
| `service.py` | `limit: int = 20`（list_personas, list_profile_items 等）, `turns_limit=20`, `traces_limit=12` | 服务层默认分页/条数 |
| `service.py` | `session_id: str = "default"`（get_recent_turns） | 未传 session_id 时的默认值 |
| `service.py` | `self.vectorizer.tokenize(text)[:8]`（manual_remember entities 截断） | 实体列表硬编码最多 8 个 |
| `service.py` | `list_semantic_facts(..., limit=6)`（build_context 内） | 语义事实条数硬编码为 6 |
| `service.py` | `snapshot.content[:600]`（preview 截断） | 预览长度 600 字符 |
| `service.py` | `replay_remote_sync_outbox(..., limit=50)` | Outbox 重放条数 |
| `service.py` | `list_profile_items(..., limit=200)`, `confidence=0.75`（save_profile_item） | 列表上限与默认置信度 |
| `service.py` | `metadata.get("priority", 50)`（检索排序） | 优先级默认 50 |
| `domain/models.py` | `RelationshipState`: `intimacy_level=20.0`, `trust_score=50.0` | 关系状态初始值 |
| `domain/models.py` | `session_id: str = "default"`（ConversationTurn） | 会话默认 ID |
| `domain/models.py` | `policy_pack: str = "default"`（PersonaDefinition） | 人设策略包默认名 |
| `domain/models.py` | `MemoryTrace.salience=0.5`, `SemanticFact.confidence=0.75`, `salience=0.65` | 记忆/事实默认分数 |
| `domain/models.py` | `ProcedureRule.priority: int = 50` | 程序记忆优先级默认值 |
| `consolidation.py` | 各 Pydantic 模型 `Field(default_factory=list)` 等 | 字段默认空列表，属正常建模 |
| `consolidation.py` | `salience: float = 0.6`, `confidence: float = 0.75`（payload 模型） | 抽取 payload 默认分数 |
| `consolidation.py` | `[:6]`（用户说明书摘要中偏好/边界/爱好等条数截断） | 说明书每类最多展示 6 条 |
| `consolidation.py` | `float(item.get("confidence", 0.78))` 等按 predicate 不同的 0.68~0.9 | **按语义类型的默认置信度**（prefers→0.78, boundary→0.9, life_routine→0.68 等） |
| `consolidation.py` | `state.recent_conflict_level >= 50`（边界/风险判断） | 冲突水平阈值 50 |
| `consolidation.py` | `source="llm_profile"`（ConsolidationProfilePayload） | 固定来源标识 |
| `consolidation.py` | `operation="upsert_trace"`, `operation="sync_relationship_state"`（outbox payload） | 固定操作类型字符串 |
| `consolidation.py` | `source="consolidation_profile"`, `source="consolidation_semantic"` | 固定来源标识 |
| `retrieval.py` | `limit=32`（list_related_nodes）, `limit=16`（Neo4j search_related）, `limit=max(limit * prefetch_factor, 8)`（Qdrant） | 检索内部预取条数 |
| `retrieval.py` | `floor = 0.45 if trace.kind == SEMANTIC else 0.08`（时间衰减） | 语义/情景记忆衰减下限 |
| `retrieval.py` | `total = clamp(total, 0.0, 10.0)` | 总分上限 10 |
| `stores/sqlite_store.py` | `DEFAULT 0.5`, `DEFAULT 0.75`, `DEFAULT 0.65`（salience/confidence 等） | 表结构默认值 |
| `stores/sqlite_store.py` | `policy_pack TEXT NOT NULL DEFAULT 'default'` | 策略包默认名 |
| `stores/sqlite_store.py` | `limit=20`, `limit=200`, `limit=100`, `limit=500`, `LIMIT 500` 等 | 各类查询的默认 limit |
| `stores/sqlite_store.py` | `claim_remote_sync_tasks(..., limit=50)` | 认领 outbox 任务条数 |
| `stores/sqlite_store.py` | `max_salience: float = 0.20`, `similarity_threshold: float = 0.85`（归档/合并） | 归档与合并阈值 |
| `stores/qdrant_store.py` | `"distance": "Cosine"` | 向量距离类型写死为余弦 |
| `stores/qdrant_store.py` | `timeout=10`（httpx.Client） | 请求超时 10 秒，未从配置读取 |
| `stores/neo4j_store.py` | `timeout=10`（httpx.Client） | 同上 |
| `stores/neo4j_store.py` | `limit: int = 10`（search_related） | 图检索条数 |
| `vectorizer.py` | `dimensions = max(32, dimensions)` | 维度最小值 32 |
| `vectorizer.py` | `max_length = max(16, max_length)` | 最大长度最小值 16 |
| `vectorizer.py` | `timeout: int = 120`（OpenAICompatibleVectorizer） | 默认超时，未从 config 传入时可视为 hardcode |
| `vectorizer.py` | `"return_tensors": "mlx"`（MLX 调用） | MLX 后端固定参数 |

### 1.3 业务字符串与种子数据

| 位置 | 内容 | 说明 |
|------|------|------|
| `service.py` | `_ensure_default_persona_seed(project_id="system", user_id="system")` | 系统级默认人设的 project/user 固定为 `"system"` |
| `service.py` | `name="companion-default"`, `policy_pack="default"`, `description="Seeded from companion_profile.md"` | 默认人设名称、策略包、描述 |
| `service.py` | `prompt = text or "# Companion Profile\n\n你是一个稳定、真诚、富有边界感的虚拟恋人。"` | **当 companion_profile.md 不存在时的 fallback 提示词**，硬编码中文文案 |
| `service.py` | `metadata={"source": "manual"}`, `predicate="manual_note"`, `edge_type="manual_note"`, `source="manual_remember"` | 手动记忆的来源与类型标识 |
| `ws_server.py` | `"service": "arm_memory"`, `"status": "ok"`, `"status": "saved"`, `"status": "cleared"` 等 | 响应固定字段值，属协议约定 |
| `ws_server.py` | `code="invalid_request"` | 错误码 |
| `consolidation.py` | `source="llm_profile"`（ConsolidationProfilePayload） | 抽取中画像来源标识（名称仍为 llm，实际来自外部） |
| `stores/sqlite_store.py` | `"pending"`, `"in_progress"`, `"done"`, `"failed"`（outbox status） | 状态机字符串 |
| `stores/sqlite_store.py` | `definition="TEXT NOT NULL DEFAULT '{}'"`（behavior_state_json） | 列默认值 |

---

## 二、Mock / 测试数据（仅测试代码）

以下仅出现在 `tests/` 或测试相关配置中，**不进入生产代码**。

| 位置 | 内容 | 说明 |
|------|------|------|
| `tests/conftest.py` | `StubVectorizer` | 测试用向量化：固定 `dimensions=8`，`embed` 基于词表 `["music", "coffee", "support", "calm", "repair", "run", "sleep", "focus"]` 的计数向量 |
| `tests/conftest.py` | `make_config(base_dir)` | 测试用配置：`qdrant_url=""`, `neo4j_url=""` 等，以及 `companion_profile_path=base_dir / "personas" / "companion_profile.md"`（base_dir 为测试临时目录） |
| `tests/test_retrieval.py` | `FakeSQLiteStore` | 内存实现的假 SQLite Store，用于检索逻辑单测 |
| `tests/test_consolidation.py` | `extraction = {"episodic": [...], "semantic": [...], "profile": {...}, ...}` 中 `"一起聊了咖啡"`, `"coffee"`, `"proj"`, `"user"` 等 | 单测用的最小可运行 payload，属于**测试数据**，非生产 mock |
| `tests/test_consolidation.py` | `project_id="proj"`, `user_id="user"` | 单测中的项目/用户 ID |

上述测试用 stub/fake 与 payload 是合理用法，无需改为生产配置。

---

## 三、建议汇总

1. **建议配置化（若需运维可调）**
   - `ws_server.py`: `_OUTBOX_REPLAY_INTERVAL_SECONDS`, `_OUTBOX_REPLAY_LIMIT`
   - `ws_server.py`: 各 action 的 `limit` 默认值（如 20, 12, 60, 200）可考虑集中到 config 或文档约定
   - `stores/qdrant_store.py` / `stores/neo4j_store.py`: HTTP `timeout=10` 改为从 config 或环境变量读取
   - `config.py`: `qdrant_collection`, `service_ws_path`, `companion_node_name` 若需多环境区分，可加环境变量

2. **建议保留但可文档化**
   - `project_id`/`user_id`/`session_id` 的 `"default"` 回退：当前行为是「未传则用 default」，若希望强制调用方传参，可改为必填并返回 400，否则在 README 中明确说明默认值语义。
   - 各类 `limit`、`confidence`、`salience` 默认值：多数已有环境变量或 payload 入参，其余可作为 API 契约在 README 中写明。

3. **Mock 数据**
   - 仅存在于 `tests/`，无生产 mock；`StubVectorizer` 与 `FakeSQLiteStore` 仅用于单测，无需修改。

如需对某一类 hardcode（例如所有 `"default"`、或所有 limit 类）做重构或加配置，可指定类别再细化修改方案。
