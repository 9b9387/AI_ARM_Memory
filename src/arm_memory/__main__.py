from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from loguru import logger

from arm_memory.config import ARMConfig
from arm_memory.logging_config import setup_logging
from arm_memory.service import ARMMemoryService
from arm_memory.ws_server import create_app

# 从项目根加载 .env（便于复制 .env.example 为 .env 后直接运行）
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")

def main() -> None:
    config = ARMConfig.from_env()
    log_path = setup_logging(log_path=config.log_path, log_level=config.log_level)
    parser = argparse.ArgumentParser(description="Run the ARM memory websocket service.")
    parser.add_argument("--host", default=config.service_host)
    parser.add_argument("--port", default=config.service_port, type=int)
    args = parser.parse_args()

    logger.info("ARM logging initialized: level={} path={}", config.log_level, log_path)

    try:
        service = ARMMemoryService.from_env()
    except (ValueError, RuntimeError) as e:
        logger.error("依赖检查未通过，请先启动 Qdrant 与 Neo4j 并配置环境变量:\n{}", e)
        sys.exit(1)

    logger.info(
        "Embedding 检查中（首次运行可能自动下载模型，请稍候）: provider={} model={}",
        config.embedding_provider,
        config.embedding_model,
    )
    if not service.vectorizer.healthcheck():
        logger.error(
            "Embedding 未就绪。请检查 ARM_EMBEDDING_* 配置与网络；"
            "MLX 模式下首次运行会从 Hugging Face 拉取模型。"
        )
        sys.exit(1)
    logger.info("Embedding 就绪: {}", config.embedding_model)

    logger.info(
        "ARM WebSocket Service 启动配置:\n{}",
        "\n".join(config.summary_lines()),
    )
    uvicorn.run(
        create_app(service=service),
        host=args.host,
        port=args.port,
        log_level="info",
        log_config=None,
    )


if __name__ == "__main__":
    main()
