from __future__ import annotations

from pathlib import Path

from loguru import logger

from arm_memory.config import ARMConfig
from arm_memory.logging_config import setup_logging


def test_config_from_env_uses_default_log_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ARM_DATA_DIR", raising=False)
    monkeypatch.delenv("ARM_LOG_DIR", raising=False)
    monkeypatch.delenv("ARM_LOG_PATH", raising=False)
    monkeypatch.delenv("ARM_LOG_LEVEL", raising=False)

    config = ARMConfig.from_env(base_dir=tmp_path)

    assert config.data_dir == tmp_path / "data" / "arm_memory"
    assert config.log_dir == config.data_dir / "logs"
    assert config.log_path == config.log_dir / "arm_memory.log"
    assert config.log_level == "INFO"


def test_setup_logging_creates_rotating_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "arm_memory.log"

    resolved = setup_logging(log_path=log_path, log_level="INFO")
    logger.info("hello logging")

    assert resolved == log_path.resolve()
    assert log_path.exists()
    assert "hello logging" in log_path.read_text(encoding="utf-8")
