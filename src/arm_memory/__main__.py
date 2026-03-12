from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from arm_memory.config import ARMConfig
from arm_memory.service import ARMMemoryService
from arm_memory.ws_server import create_app

# 从项目根加载 .env（便于复制 .env.example 为 .env 后直接运行）
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    config = ARMConfig.from_env()
    parser = argparse.ArgumentParser(description="Run the ARM memory websocket service.")
    parser.add_argument("--host", default=config.service_host)
    parser.add_argument("--port", default=config.service_port, type=int)
    args = parser.parse_args()

    try:
        service = ARMMemoryService.from_env()
    except (ValueError, RuntimeError) as e:
        logger.error("依赖检查未通过，请先启动 Qdrant 与 Neo4j 并配置环境变量:\n%s", e)
        sys.exit(1)

    logger.info(
        "Embedding 检查中（首次运行可能自动下载模型，请稍候）: provider=%s model=%s",
        config.embedding_provider,
        config.embedding_model,
    )
    if not service.vectorizer.healthcheck():
        logger.error(
            "Embedding 未就绪。请检查 ARM_EMBEDDING_* 配置与网络；"
            "MLX 模式下首次运行会从 Hugging Face 拉取模型。"
        )
        sys.exit(1)
    logger.info("Embedding 就绪: %s", config.embedding_model)

    logger.info(
        "ARM WebSocket Service 启动配置:\n%s",
        "\n".join(config.summary_lines()),
    )
    uvicorn.run(
        create_app(service=service),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
