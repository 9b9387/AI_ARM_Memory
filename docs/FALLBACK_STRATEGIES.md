# 项目中的回退（Fallback）策略汇总

本文档列出当前代码里所有「主路径失败或不可用时采用备用行为」的逻辑，便于审查与简化。

---

## 1. 巩固阶段：Qdrant / Neo4j 写入失败 → 入队远程同步 outbox

**位置**：`src/arm_memory/consolidation.py`（`run_consolidation` 内）

**逻辑**：  
巩固时向 Qdrant 写入 trace、向 Neo4j 写入 fact/edge/relationship_state，任一步失败不中断巩固，而是把该操作入队到 SQLite 的「远程同步发件箱」，由后台或手动重放。

**具体**：  
- **Qdrant**：`qdrant_store.upsert_trace(trace)` 失败 → `sqlite_store.enqueue_remote_sync_task(backend="qdrant", operation="upsert_trace", ...)`。  
- **Neo4j**：`sync_fact` / `sync_edge` / `sync_relationship_state` 任一步失败 → `enqueue_remote_sync_task(backend="neo4j", operation=..., ...)`。

**后续**：  
- `service.replay_remote_sync_outbox()` 与 ws_server 内的定时任务会重放这些任务；单次重放失败则标记 failed，不抛异常。

---

## 2. 巩固后处理：归档 / 合并失败 → 仅打日志

**位置**：`src/arm_memory/consolidation.py`（巩固流程末尾）

**逻辑**：  
- `archive_stale_memory_traces` 抛异常 → 仅 `logger.warning("archive_stale_traces failed", ...)`，不中断、不入队。  
- `find_similar_trace_pairs` 或后续 `merge_trace_pair` 抛异常 → 仅 `logger.warning("merge_similar_traces failed", ...)`，不中断。

即：归档与合并是「尽力而为」，失败只记日志，主流程（巩固结果）仍正常返回。

---

## 3. Qdrant / Neo4j：启动时必选，不满足则退出（无回退）

**位置**：`src/arm_memory/service.py`（`_validate_remote_dependencies`）、`__main__.py`

**逻辑**：  
本服务**不支持** Qdrant 或 Neo4j 未启用时的降级运行。  
- 启动时（`ARMMemoryService.from_env()`）必须已配置 `ARM_QDRANT_URL` 与 `ARM_NEO4J_*`，且 Qdrant / Neo4j 服务可达（`ping()` 成功）。  
- 若未配置或不可达，抛出 `ValueError` / `RuntimeError`，进程退出；异常信息中包含 Docker 启动命令与配置说明，便于用户先启动依赖再重试。  
- 无「未启用时仅用 SQLite/词法」的回退策略。

**说明**：单次写入/检索时 Qdrant 或 Neo4j 请求失败仍可能打日志或入队 outbox（见下节），但**服务启动**依赖两者必须可用。

---

## 4. 远程同步 outbox 重放：单条任务失败 → 标记失败并继续

**位置**：`src/arm_memory/service.py`（`replay_remote_sync_outbox`）

**逻辑**：  
对每条 outbox 任务执行 `_replay_remote_sync_task(task)`；若抛异常则 `mark_remote_sync_task_failed(task_id, ...)`，`result["failed"] += 1`，继续下一条，不抛。

即：重放采用「单条失败不影响其余」的回退策略。

---

## 5. 模型/枚举解析失败 → 默认值

**位置**：多处

**逻辑**：  
- **EmotionVector.from_dict**（`domain/models.py`）：`EmotionLabel(label)` 抛 `ValueError` → 使用 `EmotionLabel.NEUTRAL`。  
- **BehaviorState.from_dict**（`domain/models.py`）：`BoundaryStage(stage)` 抛 `ValueError` → 使用 `BoundaryStage.CALM`。  
- **consolidation._operation_from_value**：解析失败 → 使用 `MemoryOperation.ADD`。  
- **consolidation._profile_item_from_extracted**：`ProfileFacetType(facet_type)` 抛 `ValueError` → 返回 `None`（该条 profile item 被跳过）。

这些属于「脏数据容错」：外部或 LLM 给出非法枚举值时用安全默认，避免整段流程崩溃。

---

## 6. Embedding 健康检查失败 → 返回 False

**位置**：`src/arm_memory/vectorizer.py`（`OpenAICompatibleEmbeddingProvider.healthcheck`、`MLXEmbeddingProvider.healthcheck`）

**逻辑**：  
`self.embed("healthcheck")` 抛异常 → `logger.warning("...healthcheck failed")`，返回 `False`。  
用于启动时或 ping 时判断依赖是否可用，不抛异常。

---

## 7. WebSocket / 后台任务中的异常处理

**位置**：`src/arm_memory/ws_server.py`

**逻辑**：  
- **收包**：`WebSocketDisconnect` 正常断开；其他异常 → `logger.exception` 后 return，不向对端发业务错误。  
- **请求分发**：解析或业务抛异常 → 向对端发送 `ARMError`（如 `invalid_request`），不断开连接。  
- **consolidate_session 后台任务**：巩固抛异常 → 发送 `ARMError(consolidation_failed, ...)` 事件，不拖垮 ws。  
- **outbox 消费循环**：除 `CancelledError` 外异常 → `logger.exception`，等待下一轮再试；超时后继续 `wait`，实现定时重试。

这些是「服务稳定性」层面的回退：单请求或单任务失败不影响连接与其它请求。

---

## 8. Qdrant / Neo4j 未启用时的行为（已取消回退）

**位置**：生产环境通过 `ARMConfig.from_env()` 创建配置时，`require_qdrant` 与 `require_neo4j` 固定为 `True`，不再从环境变量读取。

**逻辑**：  
生产启动必须满足 Qdrant、Neo4j 已配置且可达，否则进程退出并输出启动方法。仅测试中可通过 `make_config(require_qdrant=False, require_neo4j=False)` 跳过该校验。

---

## 汇总表

| 类型           | 位置                     | 主路径失败时行为                         | 可配置/可关闭 |
|----------------|--------------------------|------------------------------------------|---------------|
| 远程写入入队   | consolidation            | 入队 outbox，后台重放                    | 否（固定逻辑） |
| 归档/合并      | consolidation            | 仅打 warning                             | 否 |
| Qdrant/Neo4j 启动 | service + __main__       | 未配置或不可达则退出并输出启动方法，无回退   | 否（生产固定必选） |
| Qdrant 单次失败   | qdrant_store             | 打日志，返回空/不抛                     | 否 |
| Neo4j 单次失败   | neo4j_store              | 打日志，返回 []                         | 否 |
| Outbox 重放    | service                  | 单条标记 failed，继续下一条             | 否 |
| 枚举/模型解析  | domain/models、consolidation | 使用默认枚举或跳过该条                 | 否 |
| 健康检查       | vectorizer               | 返回 False                               | 否 |
| WS/后台异常    | ws_server                | 发 error 或打日志，不断开/不崩溃        | 否 |

嵌入（Embedding）初始化失败时不再回退到哈希向量，仅打印警告并重新抛出异常，由调用方处理。
