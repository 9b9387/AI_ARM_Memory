from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class ARMRequest:
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex)
    message_type: str = "request"

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.message_type,
                "request_id": self.request_id,
                "action": self.action,
                "payload": self.payload,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "ARMRequest":
        payload = json.loads(raw)
        if payload.get("type") != "request":
            raise ValueError("Invalid websocket message type.")
        action = str(payload.get("action", "")).strip()
        if not action:
            raise ValueError("Missing websocket action.")
        return cls(
            action=action,
            payload=dict(payload.get("payload") or {}),
            request_id=str(payload.get("request_id") or uuid4().hex),
        )


@dataclass(slots=True)
class ARMResponse:
    request_id: str
    action: str
    data: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    message_type: str = "response"

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.message_type,
                "request_id": self.request_id,
                "action": self.action,
                "status": self.status,
                "data": self.data,
            },
            ensure_ascii=False,
        )


@dataclass(slots=True)
class ARMEvent:
    request_id: str
    action: str
    event: str
    data: dict[str, Any] = field(default_factory=dict)
    message_type: str = "event"

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.message_type,
                "request_id": self.request_id,
                "action": self.action,
                "event": self.event,
                "data": self.data,
            },
            ensure_ascii=False,
        )


@dataclass(slots=True)
class ARMError:
    request_id: str
    action: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    message_type: str = "error"

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.message_type,
                "request_id": self.request_id,
                "action": self.action,
                "error": {
                    "code": self.code,
                    "message": self.message,
                    "details": self.details,
                },
            },
            ensure_ascii=False,
        )
