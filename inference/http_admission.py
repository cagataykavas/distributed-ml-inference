"""Bounded ASGI request-body admission before application parsing."""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyAdmissionMiddleware:
    """Reject oversized or ambiguous HTTP bodies before JSON decoding."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers", [])
        content_lengths = [value for name, value in headers if name.lower() == b"content-length"]
        transfer_encodings = [
            value for name, value in headers if name.lower() == b"transfer-encoding"
        ]
        if len(content_lengths) > 1:
            await self._reject(scope, send, 400, "multiple Content-Length headers")
            return
        if len(transfer_encodings) > 1:
            await self._reject(scope, send, 400, "multiple Transfer-Encoding headers")
            return
        if content_lengths and transfer_encodings:
            await self._reject(scope, send, 400, "ambiguous request framing")
            return
        if transfer_encodings and transfer_encodings[0].strip().lower() != b"chunked":
            await self._reject(scope, send, 400, "unsupported Transfer-Encoding")
            return

        declared_length: int | None = None
        if content_lengths:
            raw_length = content_lengths[0].strip()
            if not raw_length or not raw_length.isdigit():
                await self._reject(scope, send, 400, "invalid Content-Length")
                return
            declared_length = int(raw_length)
            if declared_length > self.max_body_bytes:
                await self._reject(scope, send, 413, "request body exceeds configured byte limit")
                return

        parts: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                await self._reject(scope, send, 400, "invalid ASGI request message")
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_body_bytes:
                await self._reject(scope, send, 413, "request body exceeds configured byte limit")
                return
            parts.append(chunk)
            if not message.get("more_body", False):
                break

        if declared_length is not None and total != declared_length:
            await self._reject(scope, send, 400, "Content-Length does not match request body")
            return

        body = b"".join(parts)
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(scope: Scope, send: Send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}, separators=(",", ":")).encode()
        headers: list[tuple[bytes, bytes]] = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"cache-control", b"no-store"),
        ]
        if scope.get("http_version") in {"1.0", "1.1"}:
            headers.append((b"connection", b"close"))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
