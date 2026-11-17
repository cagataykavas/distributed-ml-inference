from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.types import Message, Scope

from inference.config import Settings
from inference.http_admission import RequestBodyAdmissionMiddleware
from inference.service import create_app


def test_request_body_limit_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_MAX_REQUEST_BODY_BYTES", "4096")

    assert Settings.from_env().max_request_body_bytes == 4096


def test_request_body_limit_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_MAX_REQUEST_BODY_BYTES", "0")

    with pytest.raises(ValueError, match="must be positive"):
        Settings.from_env()


async def run_middleware(
    chunks: list[bytes],
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    max_body_bytes: int = 8,
) -> tuple[list[Message], list[bytes]]:
    received = [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ]
    sent: list[Message] = []
    downstream_bodies: list[bytes] = []

    async def receive() -> Message:
        return received.pop(0)

    async def send(message: Message) -> None:
        sent.append(message)

    async def downstream(
        scope: Scope,
        replay: Callable[[], Awaitable[Message]],
        downstream_send: Callable[[Message], Awaitable[None]],
    ) -> None:
        del scope
        message = await replay()
        downstream_bodies.append(message.get("body", b""))
        await downstream_send({"type": "http.response.start", "status": 204, "headers": []})
        await downstream_send({"type": "http.response.body", "body": b""})

    middleware = RequestBodyAdmissionMiddleware(downstream, max_body_bytes=max_body_bytes)
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/predict",
        "raw_path": b"/predict",
        "query_string": b"",
        "root_path": "",
        "headers": headers or [],
        "server": ("test", 80),
        "client": ("127.0.0.1", 1234),
    }
    await middleware(scope, receive, send)
    return sent, downstream_bodies


@pytest.mark.asyncio
async def test_replays_an_admitted_chunked_body_once() -> None:
    sent, downstream = await run_middleware(
        [b"abc", b"def"],
        headers=[(b"transfer-encoding", b"chunked")],
    )

    assert sent[0]["status"] == 204
    assert downstream == [b"abcdef"]


@pytest.mark.asyncio
async def test_rejects_chunked_body_over_the_streaming_limit() -> None:
    sent, downstream = await run_middleware([b"12345", b"6789"], max_body_bytes=8)

    assert sent[0]["status"] == 413
    assert downstream == []
    assert (b"connection", b"close") in sent[0]["headers"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        [(b"content-length", b"3"), (b"content-length", b"3")],
        [(b"content-length", b"3"), (b"transfer-encoding", b"chunked")],
        [(b"content-length", b"+3")],
        [(b"transfer-encoding", b"gzip")],
    ],
)
async def test_rejects_ambiguous_or_invalid_framing(headers: list[tuple[bytes, bytes]]) -> None:
    sent, downstream = await run_middleware([b"abc"], headers=headers)

    assert sent[0]["status"] == 400
    assert downstream == []


@pytest.mark.asyncio
async def test_rejects_declared_length_mismatch() -> None:
    sent, downstream = await run_middleware([b"abc"], headers=[(b"content-length", b"4")])

    assert sent[0]["status"] == 400
    assert downstream == []


@pytest.mark.asyncio
async def test_service_rejects_oversized_json_before_route_parsing() -> None:
    app = create_app(Settings(max_request_body_bytes=64))
    async with app.router.lifespan_context(app):  # noqa: SIM117 - explicit lifecycle under test.
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/predict",
                content=b"x" * 65,
                headers={"content-type": "application/json"},
            )

    assert response.status_code == 413
    assert response.json() == {"detail": "request body exceeds configured byte limit"}
