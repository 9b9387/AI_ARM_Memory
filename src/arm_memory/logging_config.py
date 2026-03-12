from __future__ import annotations

from pathlib import Path

from loguru import logger as loguru_logger


_LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}"
_FILE_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name} | {message}"
_ROTATION = "5 MB"
_RETENTION = 5


def setup_logging(*, log_path: Path, log_level: str = "INFO") -> Path:
    resolved_log_path = log_path.expanduser().resolve()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    loguru_logger.remove()
    loguru_logger.add(
        lambda message: print(message, end=""),
        level=log_level.upper(),
        format=_LOG_FORMAT,
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )
    loguru_logger.add(
        resolved_log_path,
        level=log_level.upper(),
        format=_FILE_LOG_FORMAT,
        rotation=_ROTATION,
            retention=_RETENTION,
        encoding="utf-8",
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )

    return resolved_log_path