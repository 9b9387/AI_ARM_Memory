from __future__ import annotations

from collections import Counter
from pathlib import Path

from arm_memory.config import ARMConfig


def make_config(base_dir: Path) -> ARMConfig:
    data_dir = base_dir / "data"
    log_dir = data_dir / "logs"
    log_path = log_dir / "arm_memory.log"
    sqlite_path = data_dir / "arm_memory.db"
    return ARMConfig(
        base_dir=base_dir,
        data_dir=data_dir,
        log_dir=log_dir,
        log_path=log_path,
        sqlite_path=sqlite_path,
        policies_dir=base_dir / "policies",
        companion_profile_path=base_dir / "personas" / "companion_profile.md",
        log_level="INFO",
        qdrant_url="",
        neo4j_url="",
        neo4j_user="",
        neo4j_password="",
        require_qdrant=False,
        require_neo4j=False,
    )


class StubVectorizer:
    dimensions = 8

    def tokenize(self, text: str) -> list[str]:
        return [token.strip().lower() for token in text.split() if token.strip()]

    def embed(self, text: str) -> list[float]:
        counts = Counter(self.tokenize(text))
        vocab = ["music", "coffee", "support", "calm", "repair", "run", "sleep", "focus"]
        return [float(counts.get(token, 0)) for token in vocab]
