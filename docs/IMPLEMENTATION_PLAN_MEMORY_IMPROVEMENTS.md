# 记忆服务优化落地方案

本文档针对 [AI_COMPANION_MEMORY_ANALYSIS.md](AI_COMPANION_MEMORY_ANALYSIS.md) 与责任划分结论，给出**本项目内**需实现的优化项的具体落地方案（文件级、接口级、配置与数据流）。

---

## 一、TMS 与冲突解决、合并防矛盾

### 1.1 目标

- 同一 facet 内互斥档案项（如「素食」与「爱吃牛排」）不同时存在；新写入时标记或移除旧项。
- 同一 `normalized_key` 的语义事实若为「替代」关系，旧事实置为失效。
- 合并相似 trace 时，若两条情感/语义对立则跳过合并。

### 1.2 方案

#### A. Profile 冲突（preferences / dislikes 等）

- **数据与配置**  
  - 在 `config.py` 增加可选配置：`profile_conflict_pairs_path: Path | None`，指向 JSON 文件；格式示例：`{"preferences": [["素食", "肉食"], ["咖啡", "茶"]], "dislikes": []}`，表示同 facet 内这些值互斥。  
  - 若不配置或文件为空，则不做冲突检测，保持当前「追加+去重」行为。

- **逻辑位置**  
  - 在 `consolidation.py` 的 `_apply_profile_updates` 中，在调用 `save_profile_item` 之前（或封装在 `sqlite_store.save_profile_item` 内）：  
    - 读取当前 project/user 下该 facet 的已有 profile_items（`list_profile_items` 已存在）。  
    - 若配置了 conflict_pairs，则检查新 value 是否与某条已有 value 在同一互斥对中；若是，则先调用「将旧条标记为已替代」的接口（见下），再写入新条。

- **存储层**  
  - 在 `profile_items` 表已有 `status` 的前提下，增加对 `ProfileItemStatus` 的使用：新增 `SUPERSEDED = "superseded"`（或复用 `ARCHIVED` 并加 metadata 原因）。  
  - 在 `sqlite_store` 中新增：`mark_profile_item_superseded(project_id, user_id, facet_type, value_or_item_id)`，将对应条目的 `status` 更新为 superseded，并可选写入 `metadata_json` 如 `{"superseded_at": "ISO8601", "replaced_by": "新值"}`。  
  - `list_profile_items` 查询时已按 `status != 'deleted'` 过滤，需确认是否排除 `superseded`（建议排除，即只返回 `status = 'active'`）。

- **调用链**  
  - `apply_extraction` → `_apply_profile_updates` → 对每条要写入的 profile item，先解析 conflict_pairs，若新 value 与已有 value 互斥则调用 `mark_profile_item_superseded`(旧)，再 `save_profile_item`(新)。

#### B. Semantic 事实替代

- **抽取协议扩展**  
  - 在文档中约定：extraction 的 `semantic` 项可含可选字段 `supersedes_fact_id: str | null` 或 `supersedes_object: str | null`（表示替代同一 key 下的某条旧事实）。  
  - 本服务不强制上游传；若传则执行替代逻辑。

- **逻辑位置**  
  - 在 `consolidation.py` 写入 semantic 的循环中，在调用 `apply_semantic_fact` 前：若该项有 `supersedes_fact_id`，则先调用 `sqlite_store.mark_semantic_fact_deleted(fact_id)`；若为 `supersedes_object`，则先查同 `normalized_key` 下 `object_text` 匹配的 fact，再标记为 deleted。  
  - 或在 `sqlite_store.apply_semantic_fact` 内：若 `operation == MemoryOperation.ADD` 且传入可选参数 `supersedes_fact_id` / `supersedes_object`，则先执行对应的 DELETE/UPDATE 再 INSERT。

- **存储层**  
  - `semantic_facts` 已有 `status`、`valid_to`。标记替代时：`UPDATE semantic_facts SET status = 'deleted', valid_to = ? WHERE fact_id = ?` 或按 `normalized_key + normalized_object` 定位旧记录。

#### C. 合并前情感/语义对立检查

- **逻辑位置**  
  - 在 `consolidation.py` 的「merge similar traces」循环中，在 `merge_trace_pair(keep, discard)` 之前增加一次判断：若 `_emotion_contradicts(keep, discard)` 为 True 则 `continue`，不合并。

- **实现**  
  - 在 `consolidation.py` 或 `domain/models.py` 中新增函数：  
    - `_emotion_contradicts(trace_a: MemoryTrace, trace_b: MemoryTrace) -> bool`  
    - 规则示例：若两者 `emotion_tag` 属于对立组（如 JOY vs SADNESS, TRUST vs DISGUST），或 `valence` 一正一负且绝对值均 > 0.3，则返回 True。  
  - 可配置化：在 `config` 中增加 `memory_merge_skip_emotion_contradiction: bool = True`，关闭则不检查。

### 1.3 涉及文件

- `src/arm_memory/config.py`：新配置项。  
- `src/arm_memory/consolidation.py`：冲突对解析、写入前调 supersede、合并前调用 `_emotion_contradicts`。  
- `src/arm_memory/stores/sqlite_store.py`：`mark_profile_item_superseded`、`apply_semantic_fact` 支持 supersedes 参数或单独 `mark_semantic_fact_deleted`。  
- `src/arm_memory/domain/models.py`：可选增加 `ProfileItemStatus.SUPERSEDED`。  
- `docs/` 或 `README`：extraction 协议中 semantic 的 `supersedes_*` 与 profile 冲突对 JSON 格式说明。

---

## 二、Token 预算与用户说明书 JIT

### 2.1 目标

- 组装 context 时支持总 token 上限，按优先级裁剪各段，避免撑爆 LLM 窗口。  
- 用户说明书可按「当前焦点维度」只输出部分 facet，减少无关 token（JIT）。

### 2.2 方案

#### A. Token 估算与按预算裁剪

- **估算方式**  
  - 在 `arm_memory/utils.py`（或新建 `arm_memory/token_utils.py`）中实现：`estimate_tokens(text: str, lang: str = "zh") -> int`。  
  - 简单实现：中文按约 1.5 字/token，英文按约 4 字符/token，混合则按比例；或统一 `len(text) // 2` 作为上界。不依赖真实 tokenizer，仅用于预算分配。

- **BuildContextResult 与 build_context**  
  - `BuildContextResult` 增加可选字段：`max_tokens: int | None = None`（仅用于记录/透传，不改变已有字段语义）。  
  - `BuildContextResult.prompt_sections()` 增加可选参数 `max_tokens: int | None = None`。  
    - 若为 None，保持当前行为（返回全部 sections）。  
    - 若提供，则按固定顺序（人格 > 关系 > 用户说明书 > 记忆召回 > 响应策略）依次拼接，每加一段前用 `estimate_tokens` 检查累计是否超过 `max_tokens`；超过则截断该段或跳过后续段，并返回已组好的列表。  
  - 或：在 `ARMMemoryService.build_context` 内先得到 `BuildContextResult`，若调用方传入 `max_context_tokens`，则对 `user_manual_summary`、`top_memories` 等做截断（例如只保留前 N 条记忆、用户说明书只保留前若干字符），再组装。推荐在 `prompt_sections(max_tokens=)` 内统一做「按段截断」，这样上游只调一次 `prompt_sections(max_tokens)` 即可。

- **接口**  
  - `build_context` 增加可选参数：`max_context_tokens: int | None = None`。  
  - WebSocket `build_context` 的 payload 增加可选字段：`max_context_tokens: int`。  
  - 若传入，则 `BuildContextResult.max_tokens = max_context_tokens`，并在 `to_dict()` 或返回前用 `prompt_sections(max_tokens=max_context_tokens)` 生成最终用于 LLM 的片段列表；可在返回的 `data` 中增加 `prompt_sections: list[str]`（按预算裁剪后的），便于上游直接拼接。  
  - 注意：当前 `to_dict()` 返回的是 persona_prompt、relationship_prompt、user_manual_summary、top_memories、triggered_procedures；若上游自行拼 prompt，则只需在文档中说明「可用 `prompt_sections(max_tokens)` 做裁剪」；若希望服务端直接返回裁剪后的 sections，则可在 `data` 中增加 `prompt_sections`，由 `max_context_tokens` 触发。

- **实现建议**  
  - 在 `BuildContextResult` 上实现 `prompt_sections_with_budget(self, max_tokens: int) -> list[str]`，内部调用 `estimate_tokens`，按顺序累加并截断。  
  - `build_context(..., max_context_tokens=M)` 在构造完 `BuildContextResult` 后，若 M 有值，则用 `prompt_sections_with_budget(M)` 得到列表，写入 `result.prompt_sections_capped`（新字段，仅当传入 max_context_tokens 时存在），并在 `to_dict()` 中暴露为 `data["prompt_sections"]`，这样上游既可拿原始块自行拼，也可直接用 `data["prompt_sections"]`。

#### B. 用户说明书 JIT（按 facet 聚焦）

- **build_user_manual 扩展**  
  - `SleepCycleConsolidator.build_user_manual` 增加可选参数：`focus_facets: list[ProfileFacetType] | None = None`。  
  - 若为 None，行为与现在一致（输出所有 facet）。  
  - 若非空，则只生成 `lines` 中与这些 facet 相关的行（以及关系状态、使用准则等固定块），其它 facet 的 `_get_facet_lines` 不调用或返回空，从而缩短 `UserManualSnapshot.content`。

- **build_context 如何传入 focus_facets**  
  - 方式 1：由上游在 `build_context` 的 payload 中传入 `focus_facets: list[str]`（如 `["boundaries", "attachment_style"]`），本服务转为 `ProfileFacetType` 后传给 `build_user_manual`。  
  - 方式 2：本服务根据当前 `relationship` 与触发的 `procedures` 自动推导：例如若 `recent_conflict_level >= 45` 或触发了 `set_boundary`，则 `focus_facets = [BOUNDARIES, ATTACHMENT_STYLE]`；否则为 None。  
  - 建议先实现方式 1（显式传入），方式 2 可作为后续增强。

- **接口**  
  - `build_context(..., focus_facets: list[str] | None = None)`；  
  - WebSocket payload：`focus_facets: ["boundaries", "attachment_style"]` 可选。  
  - 若传入，则在调用 `build_user_manual` 时传入对应 `ProfileFacetType` 列表；若未传则 `focus_facets=None`。

### 2.3 涉及文件

- `src/arm_memory/utils.py` 或新文件：`estimate_tokens`。  
- `src/arm_memory/domain/models.py`：`BuildContextResult` 增加 `prompt_sections_with_budget`、可选字段 `prompt_sections_capped`。  
- `src/arm_memory/service.py`：`build_context` 增加 `max_context_tokens`、`focus_facets`，调用 `build_user_manual(..., focus_facets)`，可选写入 `prompt_sections`。  
- `src/arm_memory/consolidation.py`：`build_user_manual(..., focus_facets=...)`，仅输出指定 facet。  
- `src/arm_memory/ws_server.py`：从 payload 读取 `max_context_tokens`、`focus_facets` 并传给 `build_context`。  
- `README.md`：新参数说明。

---

## 三、情感参与检索与显著度动态化

### 3.1 目标

- 检索打分时加入「与当前用户情绪一致/可安慰」的记忆权重（情绪一致性）。  
- 被多次访问的记忆，其 salience 随时间温和提升，形成「越用越重要」。

### 3.2 方案

#### A. 情感参与检索

- **检索入口参数**  
  - `HybridRetrievalEngine.retrieve(..., query_emotion_hint: EmotionVector | None = None)`。  
  - `ARMMemoryService.build_context(..., user_emotion_hint: EmotionVector | dict | None = None)`；若为 dict 则 `EmotionVector.from_dict(user_emotion_hint)`。

- **打分**  
  - 在 `retrieval.py` 的 `_score_trace` 中增加参数 `query_emotion_hint: EmotionVector | None`。  
  - 计算 `emotion_similarity`：  
    - 若 `query_emotion_hint` 为 None 或 trace 无有效 emotion，则 `emotion_similarity = 0.5`（中性，不拉高也不拉低）。  
    - 否则：用效价/唤醒度等标量做简单相似度，例如  
      `1 - 0.5 * (|trace.valence - hint.valence| + |trace.arousal - hint.arousal|)` 并 clamp 到 [0,1]，或对 `emotion_tag` 做「同向/反向」查表（同向 1.0，反向 0.0，其它 0.5）。  
  - 总分的加权项增加：`emotion_similarity * retrieval_weight_emotion`。  
  - 归一化：当前权重和为 semantic+lexical+graph+salience+remote；增加 emotion 后需保证总和仍为 1 或保持可配置，例如 `retrieval_weight_emotion` 默认 0.10，相应将 semantic 从 0.40 改为 0.35（或通过配置单独设，不强制和为 1）。

- **配置**  
  - `config.py`：`retrieval_weight_emotion: float = 0.10`，并从环境变量读取。

- **上游如何传情绪**  
  - WebSocket `build_context` 的 payload 增加可选 `user_emotion_hint: { "label": "sadness", "valence": -0.5, "arousal": 0.3, ... }`。  
  - 上游可在「对话→抽取」或当前轮中由 LLM/规则产出该结构，再在调用 build_context 时传入。

#### B. 显著度动态化

- **策略**  
  - 每次 `record_memory_access(trace_id)` 时，除现有 `access_count += 1`、`last_accessed_at = now` 外，对该 trace 的 `salience` 做温和提升，例如：  
    - `new_salience = min(1.0, current_salience + salience_boost_per_access)`，或  
    - `new_salience = min(1.0, 0.5 + 0.1 * math.log1p(access_count))`（随访问次数缓慢上升）。  
  - 需先查出该 trace 的当前 `salience` 与 `access_count`，再写回。

- **实现**  
  - 在 `sqlite_store.record_memory_access` 中：先 `SELECT salience, access_count FROM memory_traces WHERE trace_id = ?`，计算 `new_salience`，再 `UPDATE memory_traces SET access_count = ..., last_accessed_at = ..., updated_at = ..., salience = ? WHERE trace_id = ?`。  
  - 配置：`salience_boost_per_access: float = 0.02`，`salience_cap: float = 1.0`；或使用 log 公式并配置系数。

### 3.3 涉及文件

- `src/arm_memory/config.py`：`retrieval_weight_emotion`、`salience_boost_per_access`、`salience_cap`。  
- `src/arm_memory/retrieval.py`：`retrieve(..., query_emotion_hint)`，`_score_trace(..., query_emotion_hint)`，增加 emotion_similarity 项与权重。  
- `src/arm_memory/service.py`：`build_context(..., user_emotion_hint)`，传给 `retrieval_engine.retrieve(..., query_emotion_hint=...)`。  
- `src/arm_memory/stores/sqlite_store.py`：`record_memory_access` 中更新 salience。  
- `src/arm_memory/ws_server.py`：payload 解析 `user_emotion_hint`。

---

## 四、Query 扩展与图多跳

### 4.1 目标

- 短句/代词导致的检索差：用「当前消息 + 近期对话」拼成扩展 query 再向量检索。  
- 图检索从「单跳实体匹配」升级为多跳路径，提高知识召回。

### 4.2 方案

#### A. Query 扩展（无 LLM）

- **入口**  
  - `retrieve(..., recent_turns: list[ConversationTurn] | None = None)` 或由 service 在调用 retrieve 前取 recent_turns 并拼成 `expanded_query` 再传一次。  
  - 推荐：在 `ARMMemoryService.build_context` 内，先取 `get_recent_turns(project_id, user_id, limit=config.retrieval_query_expand_turns)`；若 `retrieval_query_expand_turns > 0` 且 turns 非空，则 `expanded_query = message + " " + " ".join(t.content for t in recent_turns[-N:])`，然后 `retrieve(..., query=expanded_query)`；否则 `query=message`。  
  - 注意：扩展后 query 变长，可能超出 embedding 最大长度，需对 `expanded_query` 做截断（例如取最后 500 字或按 `embedding_max_length` 折算字符数）。

- **配置**  
  - `retrieval_query_expand_turns: int = 0`（0 表示不扩展）；设为 5 即用最近 5 轮内容拼接扩展。

- **一致性**  
  - 向量检索用扩展 query；词法/图检索仍可用原始 `message` 避免主题漂移，或统一用扩展 query（实现简单）。建议首版统一用扩展 query。

#### B. 图多跳

- **Neo4j**  
  - 在 `neo4j_store.py` 中新增方法：  
    - `search_related_multihop(project_id, user_id, entities: list[str], max_hops: int = 2, limit: int = 20) -> dict[str, float]`。  
  - Cypher 思路：  
    - 从匹配的 Entity 节点出发，沿 `RELATED` 关系做变长路径 1..max_hops，按跳数加权（例如每跳乘 0.7），同一 `m.key` 取最大得分。  
    - 示例：`MATCH (n:Entity {project_id: $p, user_id: $u}) WHERE n.key IN $entities MATCH path = (n)-[r:RELATED*1..2]-(m:Entity) WHERE m.project_id = $p AND m.user_id = $u WITH m, [rel IN rels(path) | 1.0] AS weights RETURN m.key AS key, 1.0 AS score`（简化；实际可按 path 长度折减 score）。  
  - 返回格式与现有 `search_related` 一致：`dict[str, float]`，便于在 retrieval 中合并到 `related_nodes`。

- **Retrieval 调用**  
  - 在 `retrieval.py` 的 `retrieve` 中，若 `neo4j_store.enabled`，除现有 `search_related` 外，再调 `search_related_multihop(..., max_hops=2)`，将返回的 score 与 `related_nodes` 合并（例如取 max），这样图侧既有单跳也有两跳结果。

- **配置**  
  - `neo4j_max_hops: int = 2`，可选从环境变量读取。

### 4.3 涉及文件

- `src/arm_memory/config.py`：`retrieval_query_expand_turns`、`neo4j_max_hops`。  
- `src/arm_memory/service.py`：`build_context` 内取 recent_turns、拼 expanded_query、截断、传 `query=expanded_query or message`。  
- `src/arm_memory/stores/neo4j_store.py`：`search_related_multihop`。  
- `src/arm_memory/retrieval.py`：调用 `search_related_multihop` 并合并结果。

---

## 五、动态衰退与主动遗忘

### 5.1 目标

- 被多次成功回忆的记忆，半衰期变长（衰退更慢）。  
- 在固化后归档「已被语义事实覆盖」的旧情景痕迹，实现基于抽象的遗忘。

### 5.2 方案

#### A. 动态半衰期

- **公式**  
  - 在 `_temporal_decay` 中，将固定 `half_life` 改为有效半衰期：  
    - `effective_half_life = half_life * (1.0 + half_life_access_boost_factor * math.log1p(max(0, trace.access_count)))`  
  - 即访问越多，effective_half_life 越大，同一 age 下 decay 越慢。

- **配置**  
  - `half_life_access_boost_factor: float = 0.5`（可调），环境变量 `ARM_HALF_LIFE_ACCESS_BOOST_FACTOR`。

#### B. 主动遗忘（归档冗余情景）

- **策略**  
  - 在 consolidation 的「archive stale」之后，增加一步：对「年龄超过 N 天且主题/实体已被高置信度语义事实覆盖」的情景 trace 进行归档。  
  - 实现方式之一：  
    - 列出该 user 下 `status=active` 且 `kind=episodic`、`updated_at < now - min_episodic_age_days` 的 traces；  
    - 列出该 user 下高置信度的 semantic facts（如 confidence >= 0.8）；  
    - 若某条 episodic trace 的 entities/tags 与某条 semantic fact 的 subject/object 重叠度高（例如至少一个 entity 与 fact 的 subject 或 object 一致），则视为「已被覆盖」，将该 trace 标记为 archived，并在 metadata 中记录 `archived_reason: "semantic_covered"`。  
  - 配置：`memory_archive_episodic_after_semantic_days: int = 60`，`memory_archive_episodic_semantic_confidence_min: float = 0.8`。

- **存储**  
  - 在 `sqlite_store` 中新增：`archive_redundant_episodic_traces(project_id, user_id, min_age_days, semantic_confidence_min) -> int`，返回归档条数。  
  - consolidation 在 `archive_stale_memory_traces` 之后调用该方法。

### 5.3 涉及文件

- `src/arm_memory/config.py`：`half_life_access_boost_factor`、`memory_archive_episodic_after_semantic_days`、`memory_archive_episodic_semantic_confidence_min`。  
- `src/arm_memory/retrieval.py`：`_temporal_decay` 使用 `effective_half_life`。  
- `src/arm_memory/stores/sqlite_store.py`：`archive_redundant_episodic_traces`。  
- `src/arm_memory/consolidation.py`：在 apply_extraction 后流程中调用 `archive_redundant_episodic_traces`。

---

## 六、检索模式（意图路由的接口侧）

### 6.1 目标

- 上游根据意图选择「偏事实」或「偏情感」检索，本服务只做权重切换，不负责意图识别。

### 6.2 方案

- **参数**  
  - `build_context(..., retrieval_mode: Literal["default", "fact", "emotional"] = "default")`。  
  - WebSocket payload：`retrieval_mode: "fact" | "emotional" | "default"` 可选。

- **权重预设**  
  - 在 `config.py` 或 `retrieval.py` 中定义三种权重预设（或仅对默认权重做缩放）：  
    - `default`：当前配置的 retrieval_weight_*。  
    - `fact`：提高 graph、lexical，降低 emotion（若已实现）、salience 略降。  
    - `emotional`：提高 emotion、salience，略降 graph。  
  - 实现方式：`retrieve(..., retrieval_mode=...)` 内部根据 mode 选择一组临时 weight 覆盖 `self.config` 的对应项（或传入 `overrides: dict`），仅当次调用有效。

- **涉及文件**  
  - `src/arm_memory/config.py`：可选增加三组预设名与权重映射，或直接在 retrieval 内写死。  
  - `src/arm_memory/retrieval.py`：`retrieve(..., retrieval_mode)`，按 mode 设权重后调用 `_score_trace`。  
  - `src/arm_memory/service.py`：`build_context(..., retrieval_mode)` 传给 retrieve。  
  - `src/arm_memory/ws_server.py`：解析 `retrieval_mode`。

---

## 七、实施顺序与依赖

| 阶段 | 项 | 依赖 | 说明 |
|------|----|------|------|
| 1 | 合并防矛盾（情感对立跳过合并） | 无 | 改动小，仅 consolidation 一处判断。 |
| 2 | 显著度动态化（record_memory_access 提升 salience） | 无 | 仅 sqlite_store + config。 |
| 3 | 情感参与检索 | 无 | retrieval + service + ws_server + config。 |
| 4 | Token 预算 + prompt_sections_with_budget | 无 | utils + models + service + ws。 |
| 5 | 用户说明书 JIT（focus_facets） | 无 | consolidation build_user_manual + service + ws。 |
| 6 | Query 扩展 | 无 | service 取 turns + config。 |
| 7 | 图多跳 | 无 | neo4j_store + retrieval + config。 |
| 8 | 动态半衰期 | 无 | retrieval _temporal_decay + config。 |
| 9 | 主动遗忘（归档冗余情景） | 无 | sqlite_store 新方法 + consolidation。 |
| 10 | retrieval_mode | 依赖 3（情感权重） | retrieval 权重覆盖 + service + ws。 |
| 11 | Profile 冲突（conflict_pairs + superseded） | 无 | config + consolidation + sqlite_store + 可选 ProfileItemStatus。 |
| 12 | Semantic 替代（supersedes_*） | 无 | consolidation + sqlite_store + 文档。 |

建议开发顺序：1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10，最后做 11、12（协议与数据变更稍多，需与上游约定）。

---

## 八、配置项汇总（新增/变更）

| 配置键 | 类型 | 默认 | 说明 |
|--------|------|------|------|
| retrieval_weight_emotion | float | 0.10 | 检索情感相似度权重。 |
| salience_boost_per_access | float | 0.02 | 每次访问提升的 salience 增量。 |
| salience_cap | float | 1.0 | salience 上限。 |
| retrieval_query_expand_turns | int | 0 | 用于 query 扩展的最近轮数，0 表示不扩展。 |
| neo4j_max_hops | int | 2 | 图多跳最大跳数。 |
| half_life_access_boost_factor | float | 0.5 | 半衰期随 access_count 的放大系数。 |
| memory_archive_episodic_after_semantic_days | int | 60 | 情景痕迹在「被语义覆盖」后多少天可归档。 |
| memory_archive_episodic_semantic_confidence_min | float | 0.8 | 用于覆盖判断的语义事实最低置信度。 |
| memory_merge_skip_emotion_contradiction | bool | True | 合并时是否跳过情感对立的 trace。 |
| profile_conflict_pairs_path | Path/str | None | 同 facet 互斥值对 JSON 文件路径。 |

---

## 九、WebSocket build_context payload 变更汇总

| 字段 | 类型 | 说明 |
|------|------|------|
| max_context_tokens | int, 可选 | 总 context token 上限，触发按预算裁剪并可选返回 prompt_sections。 |
| focus_facets | list[str], 可选 | 用户说明书只输出这些 facet（如 ["boundaries", "attachment_style"]）。 |
| user_emotion_hint | object, 可选 | 当前用户情绪，用于检索情感一致性；格式同 EmotionVector。 |
| retrieval_mode | "default" \| "fact" \| "emotional", 可选 | 检索权重模式。 |

以上为可直接按文件与接口落地的实现方案，如需某一块的伪代码或测试用例设计可再细化。
