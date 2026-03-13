# AI 伴侣自主层（Autonomy Layer）需求文档

## 1. 背景与定位

### 1.1 为什么需要 Autonomy 层

ARM Memory 是一个**纯记忆服务**：负责存储、检索、组装上下文，但不具备「思考」能力。它不调用 LLM，不理解对话内容，不做任何决策。

一个能长期陪伴的 AI 伴侣，除了「记得住」还需要「会思考」：

- 理解这段对话在说什么、用户的情绪是什么、意图是什么
- 决定什么时候该主动关心用户、什么时候该保持沉默
- 把对话内容提炼为结构化记忆写回 ARM
- 在用户说法前后矛盾时用 LLM 判断该信哪一条
- 学习用户独特的交互模式、内部梗，动态生成新策略规则
- 管理跨轮对话的「当前焦点」，而不是每轮重新猜

这些能力都需要 LLM + 编排逻辑，构成了 ARM Memory 与对话生成层之间的**自主层（Autonomy Layer）**。

### 1.2 架构位置

```
用户消息
    │
    ▼
┌─────────────────────────────────┐
│  对话生成层（Chat / LLM）        │  ← 负责最终回复生成
│  接收 autonomy 组装的完整 prompt  │
└──────────────┬──────────────────┘
               │
    ▼                      ▲
┌──────────────────────────────────┐
│  ★ Autonomy Layer（本文档范围）  │
│  编排、理解、决策、抽取、主动行为  │
└──────┬───────────────┬──────────┘
       │               │
       ▼               ▼
┌────────────┐  ┌──────────────┐
│ ARM Memory │  │   LLM API    │
│ (记忆服务)  │  │ (推理引擎)    │
└────────────┘  └──────────────┘
```

**Autonomy 的职责边界**：
- **向下调用** ARM Memory 的 `build_context`、`apply_extraction`、`ingest_turn`、`autonomy_snapshot` 等接口
- **向下调用** LLM API 完成理解、抽取、Query 重写、决策等需要推理的任务
- **向上暴露** 一套简洁的 API 给对话生成层（或直接包含对话生成）
- **不负责** 记忆的底层存储、检索算法、向量/图索引（这些由 ARM 处理）

---

## 2. 核心功能模块

### 2.1 对话理解与意图识别（Dialogue Understanding）

#### 需求

每一轮用户消息进入后，Autonomy 层需要在调 ARM `build_context` 之前，先用 LLM 做轻量分析：

| 分析维度 | 说明 | 输出 |
|---------|------|------|
| 意图分类 | 闲聊 / 求事实 / 情感倾诉 / 边界试探 / 指令 / 冲突 | `intent: str` |
| 情感识别 | 当前用户的情感状态（效价、唤醒度、标签） | `user_emotion: EmotionVector` |
| 指代消解 | 将「他」「那个」「上次说的」替换为具体实体 | `resolved_message: str` |
| 话题焦点 | 当前在聊什么，是否延续上一轮还是切换了话题 | `topic: str, topic_shift: bool` |
| 安全检测 | 是否包含自伤、违法、操控等高风险内容 | `safety_flags: list[str]` |

#### 与 ARM 的交互

- 分析结果中的 `user_emotion` 传给 `build_context(user_emotion_hint=...)` 实现情感检索
- `intent` 映射到 `retrieval_mode`（`fact` / `emotional` / `default`）
- `resolved_message` 作为 `build_context(message=...)` 的输入替代原始消息
- `safety_flags` 透传给 `apply_extraction` 并触发边界策略

#### 工作记忆管理

Autonomy 层维护一个**会话级工作记忆**（Working Memory）对象，在每轮更新：

```
WorkingMemory:
  current_topic: str           # 当前话题焦点
  topic_history: list[str]     # 近期话题序列
  current_goal: str | None     # 用户当前明确目标（如「帮我想个方案」）
  unresolved_questions: list   # 用户问了但还没答的问题
  emotional_trajectory: list   # 近 N 轮的情感变化趋势
  pending_followups: list      # AI 想追问但还没问的事
```

工作记忆不持久化到 ARM，仅在会话（session）内存活；会话结束时可摘要为一条 episodic trace 写入 ARM。

---

### 2.2 LLM Query 重写（Query Rewriting）

#### 需求

ARM 已实现了基于近期 turns 的无 LLM Query 扩展（`retrieval_query_expand_turns`），但对于复杂指代、省略、模糊表达，仍需 LLM 参与重写。

| 场景 | 原始消息 | 重写后 |
|------|---------|--------|
| 代词 | 「她怎么样了？」 | 「用户的母亲最近身体怎么样了？」 |
| 省略 | 「还有呢？」 | 「除了上次提到的日本旅行，还有什么旅行计划？」 |
| 隐含 | 「又来了」 | 「用户的失眠问题又出现了」 |

#### 实现策略

- 输入：原始消息 + 工作记忆（话题、近期 turns 摘要）
- 输出：适合检索的重写查询（1-2 句）
- LLM 调用可做成可选：短消息 (<5 字) 或指代检测到代词时才触发，避免每轮都调

---

### 2.3 对话→结构化抽取（Extraction Pipeline）

#### 需求

这是 ARM 明确声明「不负责」、由上层完成的核心环节。Autonomy 层需要在每轮对话（或每 N 轮）后，用 LLM 从对话中抽取结构化数据，再通过 `apply_extraction` 写入 ARM。

#### 抽取输出结构

```json
{
  "episodic": [
    {"summary": "...", "salience": 0.8, "emotion": {...}, "entities": [...]}
  ],
  "semantic": [
    {
      "subject": "user", "predicate": "prefers", "object": "咖啡",
      "summary": "用户喜欢喝咖啡",
      "confidence": 0.9,
      "supersedes_object": "茶"
    }
  ],
  "profile": {"preferences": ["咖啡"], "dislikes": ["被催促"]},
  "profile_items": [
    {"facet_type": "preferences", "value": "咖啡", "confidence": 0.9}
  ],
  "relationship": {
    "intimacy_delta": 1.5,
    "trust_delta": 0.5,
    "behavior_state": {"boundary_stage": "calm"}
  },
  "procedures": ["comfort_user"],
  "safety_flags": []
}
```

#### 抽取时机策略

| 模式 | 触发时机 | 适用场景 |
|------|---------|---------|
| 即时抽取 | 每轮用户消息后 | 信息密度高、偏好/边界/冲突类消息 |
| 批量抽取 | 每 N 轮或会话结束时 | 闲聊、低信息密度对话 |
| 事件驱动 | 检测到重大事件（情绪突变、关系变化）时 | 节省 LLM 调用成本 |

Autonomy 层应包含一个 `ExtractionScheduler`，根据消息的信息密度（LLM 快速判断或基于规则）决定本轮是否触发抽取。

---

### 2.4 冲突仲裁（Conflict Arbitration）

#### 需求

ARM 已实现了基于配置文件的 Profile 冲突检测（`profile_conflict_pairs`）和 Semantic 替代协议（`supersedes_*`），但这些机制依赖**上游告知**哪些是冲突。真正的冲突发现与裁决需要 LLM 参与。

#### 功能

1. **冲突发现**：抽取时，LLM 将新事实与现有 profile/semantic facts 对比，识别出逻辑矛盾
   - 输入：新抽取的事实 + ARM 中该 facet/key 的现有项
   - 输出：`is_conflict: bool`、`supersedes_object: str | null`、`confidence: float`

2. **冲突裁决**：
   - 默认规则：时间最近 + 置信度更高的胜出
   - 可选策略：下次对话时委婉向用户确认（「你之前说喜欢喝茶，是改喝咖啡了吗？」）
   - 裁决结果通过 `apply_extraction` 的 `supersedes_*` 字段写回 ARM

3. **矛盾日志**：记录每次冲突与裁决结果，供调试和人工审核

---

### 2.5 动态程序性记忆（Dynamic Procedural Learning）

#### 需求

ARM 当前的程序性记忆（ProcedureRule）是从静态 Markdown 文件加载的预设规则。Autonomy 层需要实现从交互中动态学习新规则。

#### 学习场景

| 场景 | 示例 | 学到的规则 |
|------|------|-----------|
| 用户明确纠正 | 「你以后不要在我加班的时候发长消息」 | 检测到用户在工作时段 → 使用简短回复 |
| 反复出现的模式 | 用户连续 5 次在深夜聊天且情绪低落 | 深夜 + 情绪低 → 触发 comfort_user |
| 内部梗/暗号 | 用户多次说「老规矩」表示想听音乐推荐 | 「老规矩」→ 推荐音乐 |
| 用户反馈 | 用户说「你今天回得真好」或「别这样说话」 | 强化/抑制对应的回复风格 |

#### 实现

1. **规则生成**：LLM 根据近期交互模式总结出候选规则
   - 格式与现有 `ProcedureRule` 兼容（name, prompt, triggers, priority, tags）
   - 标记为 `source: "learned"`、`confidence: float`

2. **规则验证**：
   - 新规则不能与 `companion_profile.md` 中的核心边界冲突
   - 低置信度规则先进入「试用期」（仅在匹配度很高时触发），通过后才提升优先级

3. **规则写入**：通过 ARM 的 `save_procedure` 或扩展的 `apply_extraction` 写入

4. **规则老化**：长期未触发的学习规则逐步降低优先级或归档

---

### 2.6 情景叙事整合（Episode Storyline Synthesis）

#### 需求

ARM 的 episodic trace 是独立的、碎片化的记忆节点。Autonomy 层需要将它们整合为有叙事连贯性的「故事线」。

#### 功能

1. **事件聚合**：定期（或会话结束时）用 LLM 将近期相关的 episodic traces 聚合为一个「事件」
   - 输入：某时间窗口内、共享实体/话题的多条 traces
   - 输出：一条 summary 更丰富的「事件级」trace，关联子 trace ID

2. **故事线追踪**：维护用户的活跃故事线
   - 例：「用户在找新工作」（跨 2 周，涉及 8 条 traces：面试、焦虑、结果、庆祝...）
   - 每条新 trace 写入时判断它属于哪条故事线（或开启新线）

3. **故事线检索**：当用户重提某话题时，能召回整条故事线的上下文而非零散碎片
   - 可通过 ARM 的 `manual_remember` 或专用接口写入故事线摘要

---

### 2.7 主动行为引擎（Proactive Behavior Engine）

#### 需求

真正的伴侣不只是「被动问答」，还会主动关心、主动提起话题、主动跟进之前的事。

#### 主动行为类型

| 行为 | 触发条件 | 示例 |
|------|---------|------|
| 主动问候 | 距上次对话超过 N 小时 | 「今天过得怎么样？」 |
| 事件跟进 | 用户之前提到未来事件（面试、约会、考试） | 「今天面试顺利吗？」 |
| 情感关怀 | 上次对话用户情绪极端负面 | 「昨晚聊完后你有好一些吗？」 |
| 纪念日提醒 | 用户提到过的重要日期 | 「今天是你妈妈生日对吧，有准备什么惊喜吗？」 |
| 共享/分享 | 关系到达 GrowingBond 以上 | 主动分享一首歌/一段话/一个想法 |
| 关系修复 | 上次对话以冲突收尾 | 主动发起修复性对话 |

#### 实现

1. **心跳评估**：
   - 定时（如每小时）调用 ARM 的 `autonomy_snapshot`，获取关系状态、近期 turns、traces
   - LLM 基于 snapshot 评估：「现在需要主动行动吗？如果是，应该做什么？」
   - 输出：`{ should_act: bool, action_type: str, message_draft: str, reason: str }`

2. **事件日历**：
   - 从 semantic facts 和 episodic traces 中提取带时间的事件（如「用户周三有面试」）
   - 维护一个轻量的事件日历，到时间自动触发跟进评估

3. **频率控制**：
   - 不能过度打扰：同一用户 24 小时内主动发起不超过 N 次
   - 根据关系阶段调整主动性：InitialContact 阶段克制，LongTermBond 阶段可更频繁
   - 用户明确说「别烦我」时尊重并记录到边界

---

### 2.8 Prompt 编排与对话生成（Prompt Orchestration）

#### 需求

Autonomy 层负责将 ARM 返回的 `BuildContextResult` 与自身产生的上下文（工作记忆、意图、安全标记等）编排为最终的 LLM prompt。

#### 编排流程

```
1. 调 ARM build_context → 获得人格/关系/说明书/记忆/策略
2. 注入工作记忆（当前话题、未解决的问题）
3. 注入意图/情感分析结果 → 调整语气指令
4. 如果触发了安全标记 → 注入安全约束指令
5. 如果处于 cooldown/refusal → 覆盖为边界回复模板
6. Token 预算裁剪（调 prompt_sections_with_budget）
7. 拼接系统 prompt + 近期对话 + 用户消息
8. 调 LLM 生成回复
9. 回复后：将 AI 回复 ingest_turn 到 ARM、触发抽取流水线
```

#### 回复后处理

- 将 AI 回复通过 `ingest_turn(role="assistant")` 写入 ARM
- 根据抽取调度策略决定是否触发 `ExtractionPipeline`
- 更新工作记忆（话题、情感轨迹）

---

### 2.9 情感轨迹追踪（Emotional Trajectory Tracking）

#### 需求

ARM 追踪单个 trace 的情感标签，但缺乏对用户情感基线随时间演变的宏观视角。

#### 功能

1. **情感基线计算**：
   - 基于近 N 天的对话情感标签，计算用户的「情感基线」（平均效价、平均唤醒度）
   - 用于判断当前情绪是否偏离基线（突然低落 / 异常兴奋）

2. **趋势检测**：
   - 连续 3 天效价下降 → 触发「长期低落」预警
   - 唤醒度突然飙升 → 可能发生了重大事件

3. **数据存储**：
   - 按天/周聚合为情感快照，写入 ARM 作为 semantic fact（如 `user emotional_baseline_week_12 valence:-0.2`）
   - 或在 Autonomy 层自有存储中维护时间序列

---

### 2.10 多人格管理与切换（Persona Management）

#### 需求

ARM 已有 `PersonaDefinition` 和策略包（policy_pack）的概念，但人格的动态切换逻辑应由 Autonomy 层控制。

#### 功能

1. **上下文感知的人格微调**：
   - 不是切换整个人格，而是在当前人格基础上做微调
   - 如：用户深夜情绪低落时，在保持核心人格的前提下增加柔和度
   - 用户开玩笑时，切换到更活泼的语气

2. **人格一致性检查**：
   - 每次生成回复前，Autonomy 层检查回复草稿是否与 `companion_profile.md` 中的核心边界一致
   - 不一致时重新生成或调整

---

## 3. 数据流设计

### 3.1 被动对话流（用户发消息）

```
用户消息 → Autonomy
  │
  ├── 1. ingest_turn(role=user) → ARM
  ├── 2. DialogueUnderstanding(LLM) → intent, emotion, resolved_msg, safety
  ├── 3. QueryRewriting(LLM, optional) → rewritten_query
  ├── 4. build_context(message=rewritten_query, emotion_hint, retrieval_mode, ...) → ARM
  ├── 5. PromptOrchestration → final_prompt
  ├── 6. LLM(final_prompt) → ai_reply
  ├── 7. ingest_turn(role=assistant) → ARM
  ├── 8. ExtractionPipeline(LLM, scheduled) → extraction_json
  ├── 9. apply_extraction(extraction_json) → ARM
  └── 10. 更新 WorkingMemory、情感轨迹
```

### 3.2 主动行为流（无用户消息）

```
定时心跳 → Autonomy
  │
  ├── 1. autonomy_snapshot → ARM
  ├── 2. ProactiveEvaluator(LLM) → should_act, action, draft
  ├── 3. 频率/边界检查
  ├── 4. 若 should_act:
  │     ├── 生成主动消息
  │     ├── ingest_turn(role=assistant) → ARM
  │     └── 推送给用户
  └── 5. 事件日历检查 → 触发事件跟进
```

### 3.3 后台任务流

```
定期 / 会话结束 → Autonomy
  │
  ├── 情景叙事整合 → 聚合 traces → manual_remember / apply_extraction → ARM
  ├── 动态规则学习 → 分析近期模式 → 生成 ProcedureRule → ARM
  ├── 情感轨迹聚合 → 计算基线/趋势 → 写入 fact → ARM
  └── 冲突仲裁扫描 → 检查 profile 矛盾 → supersede → ARM
```

---

## 4. 对 ARM Memory 的接口需求

Autonomy 层依赖 ARM 的现有接口，以下是关键依赖与潜在扩展需求：

| ARM 接口 | Autonomy 用途 | 是否需要扩展 |
|----------|-------------|-------------|
| `build_context` | 获取检索结果与上下文 | 已支持 emotion_hint / retrieval_mode / focus_facets / max_tokens |
| `apply_extraction` | 写入抽取结果 | 已支持 supersedes_* |
| `ingest_turn` | 写入对话轮次 | 现有即可 |
| `autonomy_snapshot` | 心跳评估 | 现有即可 |
| `get_relationship_state` | 主动行为决策 | 现有即可 |
| `manual_remember` | 故事线/规则写入 | 现有即可 |
| `get_summary` | 调试与监控 | 现有即可 |
| 写入 ProcedureRule | 动态规则持久化 | **需新增**：`save_procedure` 接口 |
| 读取 SemanticFact 列表 | 冲突发现时对比 | **需新增或暴露**：`list_semantic_facts` 接口 |
| 情感时间序列 | 基线计算 | 可选：写入 ARM 为 fact 或 Autonomy 自维护 |

---

## 5. 技术选型建议

| 组件 | 建议 |
|------|------|
| LLM 调用 | 支持多模型：轻量模型做意图/情感识别，强模型做抽取/冲突/规则生成 |
| 编排框架 | 异步 Python（asyncio）；可选用 LangGraph / 自研 pipeline |
| 工作记忆 | 内存中的 dataclass / dict，按 session_id 隔离 |
| 事件日历 | 轻量调度（APScheduler / asyncio task），无需重型队列 |
| 与 ARM 通信 | WebSocket 客户端，复用 ARM 的 ws 协议 |
| 配置 | 独立 config（LLM endpoint、心跳间隔、抽取策略、频率限制等） |
| 日志 | 统一 loguru，与 ARM 日志格式一致 |

---

## 6. 功能优先级

| 优先级 | 功能 | 理由 |
|--------|------|------|
| P0 | 对话→结构化抽取 | ARM 运转的核心依赖，没有它记忆无法增长 |
| P0 | Prompt 编排与对话生成 | 最终产出回复的管道 |
| P0 | 对话理解（意图 + 情感 + 安全） | 驱动检索模式、情感检索、边界策略 |
| P1 | 工作记忆管理 | 解决「每轮失忆」和「话题断裂」 |
| P1 | LLM Query 重写 | 大幅提升检索质量 |
| P1 | 冲突仲裁 | 确保长期陪伴不出现明显矛盾 |
| P2 | 主动行为引擎 | 从「被动问答」升级为「主动伴侣」 |
| P2 | 情景叙事整合 | 提升记忆的连贯性和召回深度 |
| P2 | 情感轨迹追踪 | 宏观理解用户情感变化 |
| P3 | 动态程序性记忆 | 个性化交互模式学习 |
| P3 | 多人格微调 | 上下文感知的语气适配 |

---

## 7. 与不在 Autonomy 层实现的能力的边界

| 能力 | 归属 | 理由 |
|------|------|------|
| 记忆存储/检索/索引 | ARM Memory | 已实现且优化 |
| 向量/图/词法混合打分 | ARM Memory | 检索算法内聚在 ARM |
| 记忆衰退/归档/合并 | ARM Memory | 生命周期管理在 ARM |
| Profile/Semantic 冲突的存储侧标记 | ARM Memory | 已实现 superseded/deleted |
| Token 预算裁剪 | ARM Memory | 已实现 prompt_sections_with_budget |
| 用户说明书生成 | ARM Memory | 已实现 build_user_manual |
| **对话理解/意图/情感识别** | **Autonomy** | 需要 LLM |
| **对话→结构化抽取** | **Autonomy** | 需要 LLM |
| **冲突发现与裁决** | **Autonomy** | 需要 LLM |
| **主动行为决策** | **Autonomy** | 需要 LLM + 调度 |
| **动态规则学习** | **Autonomy** | 需要 LLM + 模式识别 |
| **Prompt 编排** | **Autonomy** | 编排逻辑 |
| **工作记忆** | **Autonomy** | 会话级状态 |

---

## 8. 关键设计原则

1. **LLM 调用分级**：不是所有功能都需要调用最强模型。意图/情感识别用轻量模型，复杂抽取/冲突仲裁用强模型，尽量控制延迟和成本。

2. **渐进式降级**：当 LLM 不可用时，Autonomy 应能降级运行 —— 跳过 Query 重写，直接传原始消息给 ARM；跳过抽取，仅做对话写入。

3. **可观测性**：每个决策节点（意图判定、是否抽取、冲突裁决、是否主动行动）都应有日志和可选的人工审核入口。

4. **用户主权**：用户可以查看、纠正、删除 Autonomy 学到的规则和做出的判断。所有学习到的规则和冲突裁决都应可回溯。

5. **人格不可被覆盖**：`companion_profile.md` 中定义的核心边界是硬编码的护栏，Autonomy 学习到的任何规则都不能突破这些边界。
