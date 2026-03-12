from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime | None) -> datetime:
    if value is None:
        return utcnow()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def to_iso(value: datetime | None) -> str:
    return ensure_utc(value).isoformat()


def parse_iso_datetime(value: str | None) -> datetime:
    if not value:
        return utcnow()
    parsed = datetime.fromisoformat(value)
    return ensure_utc(parsed)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return json.loads(cleaned)

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group(0))


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads_json(value: str | None, *, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def parse_markdown_document(path: Path) -> tuple[dict[str, Any], str]:
    text = read_text_if_exists(path)
    if not text.startswith("---\n"):
        return {}, text

    _, remainder = text.split("---\n", 1)
    if "\n---\n" not in remainder:
        return {}, text

    raw_meta, body = remainder.split("\n---\n", 1)
    metadata: dict[str, Any] = {}
    current_key: str | None = None

    for line in raw_meta.splitlines():
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            metadata.setdefault(current_key, []).append(line[4:].strip())
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        current_key = key
        if not value:
            metadata[key] = []
            continue
        lowered = value.lower()
        if lowered in {"true", "false"}:
            metadata[key] = lowered == "true"
            continue
        if re.fullmatch(r"-?\d+", value):
            metadata[key] = int(value)
            continue
        if re.fullmatch(r"-?\d+\.\d+", value):
            metadata[key] = float(value)
            continue
        metadata[key] = value.strip("\"'")

    return metadata, body.strip()


def normalize_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()
