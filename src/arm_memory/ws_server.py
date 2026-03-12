from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from arm_memory.domain.models import SensitivityLevel, persona_from_dict, relationship_from_dict
from arm_memory.protocol import ARMError, ARMEvent, ARMRequest, ARMResponse
from arm_memory.service import ARMMemoryService

logger = logging.getLogger(__name__)
_OUTBOX_REPLAY_INTERVAL_SECONDS = 60
_OUTBOX_REPLAY_LIMIT = 50


class ARMWebSocketSession:
    def __init__(self, *, websocket: WebSocket, service: ARMMemoryService):
        self.websocket = websocket
        self.service = service

    async def serve(self) -> None:
        await self.websocket.accept()
        while True:
            try:
                raw = await self.websocket.receive_text()
            except WebSocketDisconnect:
                return
            except Exception:
                logger.exception("ARM websocket receive failed")
                return

            try:
                request = ARMRequest.from_json(raw)
                await self._dispatch(request)
            except Exception as exc:
                logger.exception("ARM websocket request handling failed")
                error = ARMError(
                    request_id="unknown",
                    action="unknown",
                    code="invalid_request",
                    message=str(exc),
                )
                await self.websocket.send_text(error.to_json())

    async def _dispatch(self, request: ARMRequest) -> None:
        action = request.action
        payload = request.payload
        request_id = request.request_id

        if action == "ping":
            await self._send_response(
                request,
                {
                    "status": "ok",
                    "service": "arm_memory",
                    "ws_path": self.service.config.service_ws_path,
                    "qdrant_enabled": self.service.qdrant_store.enabled,
                    "neo4j_enabled": self.service.neo4j_store.enabled,
                },
            )
            return

        if action == "build_context":
            result = self.service.build_context(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
                message=payload.get("message", ""),
            )
            await self._send_response(request, result.to_dict())
            return

        if action == "autonomy_snapshot":
            result = self.service.get_autonomy_snapshot(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
                query=payload.get("query", ""),
                turns_limit=int(payload.get("turns_limit", 20)),
                traces_limit=int(payload.get("traces_limit", 12)),
            )
            await self._send_response(request, result)
            return

        if action == "ingest_turn":
            turn_id = self.service.ingest_turn(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
                role=payload.get("role", "user"),
                content=payload.get("content", ""),
                session_id=payload.get("session_id", "default"),
                metadata=dict(payload.get("metadata") or {}),
            )
            await self._send_response(request, {"turn_id": turn_id})
            return

        if action == "manual_remember":
            sensitivity = self._parse_sensitivity(payload.get("sensitivity"))
            operation = self.service.manual_remember(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
                text=payload.get("text", ""),
                tags=list(payload.get("tags") or []),
                sensitivity=sensitivity,
            )
            await self._send_response(request, {"operation": operation.value})
            return

        if action == "apply_extraction":
            extraction = dict(payload.get("extraction") or {})
            turn_ids = list(payload.get("turn_ids") or [])
            if not turn_ids and payload.get("session_id"):
                turns = self.service.get_recent_turns(
                    project_id=payload.get("project_id", "default"),
                    user_id=payload.get("user_id", "default"),
                    limit=60,
                    session_id=payload.get("session_id"),
                )
                turn_ids = [t.turn_id for t in turns]
            result = self.service.apply_extraction(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
                extraction=extraction,
                turn_ids=turn_ids if turn_ids else None,
                persona_prompt=str(payload.get("persona_prompt") or ""),
            )
            await self._send_response(request, result.to_dict())
            return

        if action == "get_summary":
            summary = self.service.get_summary(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
            )
            await self._send_response(request, {"summary": summary})
            return

        if action == "clear_user_data":
            self.service.clear_user_data(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
            )
            await self._send_response(request, {"status": "cleared"})
            return

        if action == "get_relationship_state":
            state = self.service.get_relationship_state(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
            )
            await self._send_response(request, state.to_dict())
            return

        if action == "save_relationship_state":
            state_payload = dict(payload.get("state") or {})
            state = relationship_from_dict(state_payload)
            self.service.save_relationship_state(state)
            await self._send_response(request, {"status": "saved"})
            return

        if action == "get_recent_turns":
            turns = self.service.get_recent_turns(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
                limit=int(payload.get("limit", 20)),
                session_id=payload.get("session_id"),
            )
            await self._send_response(request, {"turns": [turn.to_dict() for turn in turns]})
            return

        if action == "get_recent_traces":
            traces = self.service.get_recent_traces(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
                limit=int(payload.get("limit", 20)),
            )
            await self._send_response(request, {"traces": [trace.to_dict() for trace in traces]})
            return

        if action == "get_active_persona":
            persona = self.service.get_active_persona(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
            )
            await self._send_response(request, {"persona": persona.to_dict()})
            return

        if action == "list_personas":
            personas = self.service.list_personas(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
                include_disabled=bool(payload.get("include_disabled", True)),
            )
            await self._send_response(request, {"personas": [persona.to_dict() for persona in personas]})
            return

        if action == "save_persona":
            persona_payload = dict(payload.get("persona") or {})
            persona_payload.setdefault("project_id", payload.get("project_id", "default"))
            persona_payload.setdefault("user_id", payload.get("user_id", "default"))
            persona = persona_from_dict(persona_payload)
            saved = self.service.save_persona(persona)
            await self._send_response(request, {"persona": saved.to_dict()})
            return

        if action == "activate_persona":
            persona = self.service.activate_persona(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
                persona_id=str(payload.get("persona_id") or ""),
            )
            await self._send_response(request, {"persona": persona.to_dict()})
            return

        if action == "list_profile_items":
            items = self.service.list_profile_items(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
                facet_types=list(payload.get("facet_types") or []),
                limit=int(payload.get("limit", 200)),
            )
            await self._send_response(request, {"items": [item.to_dict() for item in items]})
            return

        if action == "save_profile_item":
            item = self.service.save_profile_item(
                project_id=payload.get("project_id", "default"),
                user_id=payload.get("user_id", "default"),
                facet_type=str(payload.get("facet_type") or "manual_notes"),
                value=str(payload.get("value") or ""),
                confidence=float(payload.get("confidence", 0.75)),
                source=str(payload.get("source") or "manual"),
                source_trace_id=payload.get("source_trace_id"),
                metadata=dict(payload.get("metadata") or {}),
            )
            await self._send_response(request, {"item": item.to_dict()})
            return

        if action == "replay_remote_sync_outbox":
            result = self.service.replay_remote_sync_outbox(
                limit=int(payload.get("limit", 50)),
            )
            summary = self.service.get_remote_sync_outbox_summary()
            await self._send_response(request, {"result": result, "summary": summary})
            return

        raise ValueError(f"Unknown websocket action: {action}")

    async def _send_response(self, request: ARMRequest, data: dict[str, Any], *, status: str = "ok") -> None:
        response = ARMResponse(
            request_id=request.request_id,
            action=request.action,
            status=status,
            data=data,
        )
        await self.websocket.send_text(response.to_json())

    @staticmethod
    def _parse_sensitivity(value: Any) -> SensitivityLevel:
        text = str(value or SensitivityLevel.PRIVATE.value)
        for member in SensitivityLevel:
            if text == member.value or text == member.name:
                return member
        return SensitivityLevel.PRIVATE


async def _run_outbox_consumer(
    *,
    service: ARMMemoryService,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            result = await asyncio.to_thread(
                service.replay_remote_sync_outbox,
                limit=_OUTBOX_REPLAY_LIMIT,
            )
            if any(result.values()):
                logger.info("ARM outbox replay result: %s", result)
            summary = await asyncio.to_thread(service.get_remote_sync_outbox_summary)
            if summary.get("failed", 0) > 0:
                logger.warning("ARM outbox still has failed tasks: %s", summary)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ARM outbox replay loop failed")

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=_OUTBOX_REPLAY_INTERVAL_SECONDS,
            )
        except asyncio.TimeoutError:
            continue


def create_app(*, service: ARMMemoryService | None = None) -> FastAPI:
    service_instance = service or ARMMemoryService.from_env()
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.on_event("startup")
    async def startup() -> None:
        stop_event = asyncio.Event()
        app.state.outbox_stop_event = stop_event
        app.state.outbox_task = asyncio.create_task(
            _run_outbox_consumer(service=service_instance, stop_event=stop_event)
        )

    @app.on_event("shutdown")
    async def shutdown() -> None:
        stop_event: asyncio.Event | None = getattr(app.state, "outbox_stop_event", None)
        task: asyncio.Task | None = getattr(app.state, "outbox_task", None)
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @app.websocket(service_instance.config.service_ws_path)
    async def websocket_endpoint(websocket: WebSocket) -> None:
        session = ARMWebSocketSession(websocket=websocket, service=service_instance)
        await session.serve()

    return app
