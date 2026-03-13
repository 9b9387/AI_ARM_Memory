# ARM Memory 独立服务部署说明

本文档描述当前已落地的独立服务形态：

- `arm_memory` 作为独立 WebSocket 服务运行
- `Qdrant` 与 `Neo4j` 通过 Docker Compose 启动
- `telegram_bot` 与 `arm_autonomy` 通过 WebSocket provider 接入
- 本地 embedding 默认使用 MLX + `mlx-community/Qwen3-Embedding-0.6B-mxfp8`

## 1. Python 环境

项目已增加 `.python-version`，固定为：

```text
3.12.13
```

本项目以 `pyproject.toml` 作为依赖声明来源，仓库中**没有** `requirements.txt`；推荐使用 `pyenv` + `venv` 后通过可编辑安装拉起依赖。

推荐安装方式：

```bash
pyenv local 3.12.13
pyenv exec python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

如果需要运行测试，再安装开发依赖：

```bash
python -m pip install -e '.[dev]'
```

说明：

- 运行时依赖来自 `pyproject.toml` 的 `[project.dependencies]`
- 测试依赖当前来自 `[project.optional-dependencies].dev`，目前包含 `pytest`
- 安装完成后可直接使用入口命令 `arm-memory`

如果安装依赖时遇到网络问题，可以使用代理：

```bash
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
```

## 2. 启动 Qdrant 与 Neo4j

在项目根目录执行：

```bash
docker compose -f docker-compose.arm-memory.yml up -d
```

默认端口：

- Qdrant: `6333`
- Neo4j HTTP: `7474`
- Neo4j Bolt: `7687`

默认 Neo4j 认证：

- 用户名：`neo4j`
- 密码：`arm-memory-dev`

如果需要自定义，可以在启动前设置：

```bash
export ARM_NEO4J_AUTH=neo4j/your-password
```

## 3. 启动 ARM Memory WebSocket 服务

最低建议环境变量：

```bash
export ARM_REQUIRE_QDRANT=true
export ARM_REQUIRE_NEO4J=true
export ARM_QDRANT_URL=http://127.0.0.1:6333
export ARM_NEO4J_URL=bolt://127.0.0.1:7687
export ARM_NEO4J_USER=neo4j
export ARM_NEO4J_PASSWORD=arm-memory-dev
export ARM_SERVICE_HOST=127.0.0.1
export ARM_SERVICE_PORT=8788
export ARM_SERVICE_WS_PATH=/ws
export ARM_LOG_LEVEL=INFO
export ARM_LOG_DIR=$(pwd)/data/arm_memory/logs
export ARM_EMBEDDING_PROVIDER=mlx
export ARM_EMBEDDING_MODEL=mlx-community/Qwen3-Embedding-0.6B-mxfp8
export ARM_VECTOR_DIMENSIONS=1024
export ARM_PROFILE_ITEM_LIMIT_PER_FACET=24
export ARM_PROFILE_ITEM_STALE_DAYS=180
```

启动命令：

```bash
python -m arm_memory --host 127.0.0.1 --port 8788
```

服务会在启动时：

- 创建或校验 Qdrant collection
- 检查 Qdrant 连通性
- 检查 Neo4j 连通性
- 初始化控制台 + 本地文件日志（默认文件：`./data/arm_memory/logs/arm_memory.log`）
- 日志底层使用 `loguru`，文件日志默认按大小轮转
- 初始化 WebSocket 路由
- 启动远端同步 outbox 的后台自动补偿循环（默认每 60 秒一次）

运行过程中，日志会记录：

- 服务启动配置与依赖检查结果
- WebSocket 连接、`action`、`request_id`、处理耗时、成功/失败
- `ingest_turn`、`build_context`、`apply_extraction`、outbox 重放等关键动作的摘要信息

日志默认不会写入原始用户消息内容、完整抽取文本或敏感凭据；如需更改落盘文件，可额外设置 `ARM_LOG_PATH`。

如果需要人工触发一次补偿重放，可以通过 WebSocket 发送 `replay_remote_sync_outbox` 动作，服务会返回本次处理结果和当前 outbox 汇总状态。

## 4. 上层业务接入

### telegram_bot

建议使用远程服务模式：

```bash
export ARM_ENABLED=true
export ARM_PROVIDER=ws
export ARM_WS_URL=ws://127.0.0.1:8788/ws
```

### arm_autonomy

默认 provider 已改为 `ws`，建议配置：

```bash
export AUTONOMY_PROVIDER=ws
export AUTONOMY_ARM_WS_URL=ws://127.0.0.1:8788/ws
```

## 5. 当前状态说明

已完成：

- WebSocket 协议模型
- ARM WebSocket 服务骨架
- 独立运行入口 `python -m arm_memory`
- `telegram_bot` provider 抽象
- `arm_autonomy` WebSocket provider
- Qdrant / Neo4j 基础连通性校验
- MLX embedding provider 基础接入与本地模型验证
- persona registry 与 active persona 切换
- 用户画像快照 + 多维 profile item 存储
- apply_extraction 接收外部抽取结果并写入；远程同步 outbox 补偿重试
- `arm_memory` 服务独立打包与基础 Docker 化

未完成：

- 云端 embedding provider
- persona registry
- 多维用户画像条目模型

如果你之前已经用 384 维配置创建过 Qdrant collection，需要在切换到 Qwen3 embedding 前重建 collection，或者显式设置 `ARM_VECTOR_DIMENSIONS` 与正在使用的 embedding 模型保持一致。

## 6. 运行建议

当前推荐拓扑：

1. 在 macOS 主机运行 `mlx_lm.server`
2. 在 macOS 主机运行 `arm_memory` WebSocket 服务
3. 用 Docker Compose 运行 `Qdrant` 与 `Neo4j`
4. 让 `telegram_bot` 和 `arm_autonomy` 通过 WebSocket 接入 `arm_memory`

这是目前最符合 MLX 和 Apple Silicon 限制的运行方式。