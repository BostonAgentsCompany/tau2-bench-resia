"""A local Resia Fullplex Live API test server (no external service or real key).

It runs its own event loop in a thread. It records what the client sends, with
arrival times, and the test pushes server events to the open session.
"""

import asyncio
import base64
import json
import ssl
import threading
import time
from typing import Self

from websockets.asyncio.server import serve
from websockets.http11 import Response

# The session.start rules of the deployed contract (develop docs/live-api.md, `session.start` table): the accepted
# session fields, backend (delegation.responses) fields, and audio formats. A start outside them gets an error.
START_FIELDS = {"model", "instructions", "input", "audio", "delegation", "store"}
RESPONSES_FIELDS = {"model", "instructions", "tools", "tool_choice", "parallel_tool_calls", "max_output_tokens",
                    "reasoning", "text"}
FORMATS = ({"type": "audio/pcm", "rate": 24000}, {"type": "audio/pcm", "rate": 16000},
           {"type": "audio/pcmu", "rate": 8000}, {"type": "audio/pcma", "rate": 8000})


def start_problem(session: dict) -> str | None:
    """the first field of a session.start that the contract refuses, or None"""
    if set(session) - START_FIELDS:
        return f"unknown_parameter: {sorted(set(session) - START_FIELDS)}"
    if session.get("model") != "resia-fullplex-1":
        return "invalid_value: model"
    audio = session.get("audio") or {}
    if "format" in audio and audio["format"] not in FORMATS:
        return "invalid_value: audio.format"
    delegation = session.get("delegation")
    if isinstance(delegation, dict) and delegation.get("type") == "responses":
        responses = delegation.get("responses") or {}
        if not responses.get("model"):
            return "missing_required_parameter: delegation.responses.model"
        if set(responses) - RESPONSES_FIELDS:
            return f"unknown_parameter: {sorted(set(responses) - RESPONSES_FIELDS)}"
    return None


class FakeFullplex:
    CLOSE_SECONDS = 16.5     # usage.seconds of session.closed
    def __init__(self, *, key: str = "test-key", refuse_status: int | None = None, redirect_to: str | None = None,
                 start_error: str | None = None, ssl_context: ssl.SSLContext | None = None):
        self.key = key
        self.refuse_status = refuse_status
        self.redirect_to = redirect_to      # answer the handshake with 302 to this URL
        self.start_error = start_error      # answer session.start with this error code, then close
        self.ssl_context = ssl_context
        self.received: list[tuple[float, dict]] = []  # (monotonic time, event)
        self.auth_headers: list[str | None] = []
        self.port: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._ready = threading.Event()
        self._started = threading.Event()
        self._audio_changed = threading.Condition()
        self._stop: asyncio.Event | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ------------------------------------------------------------ lifecycle
    def __enter__(self) -> Self:
        self._thread.start()
        assert self._ready.wait(5), "fake server did not start"
        return self

    def __exit__(self, *exc) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(5)

    @property
    def url(self) -> str:
        scheme = "wss" if self.ssl_context else "ws"
        return f"{scheme}://127.0.0.1:{self.port}/v1/live/sessions"

    def _run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(
            self._handler, "127.0.0.1", 0, process_request=self._check, max_size=None,
            ssl=self.ssl_context,
        ) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    def _check(self, connection, request):
        self.auth_headers.append(request.headers.get("Authorization"))
        if self.redirect_to is not None:
            return Response(302, "Found", websockets_headers(Location=self.redirect_to))
        if self.refuse_status is not None:
            return Response(self.refuse_status, "refused", websockets_headers())
        if request.headers.get("Authorization") != f"Bearer {self.key}":
            return Response(401, "unauthorized", websockets_headers())
        return None

    async def _handler(self, ws) -> None:
        self._ws = ws
        async for message in ws:
            event = json.loads(message)
            self.received.append((time.monotonic(), event))
            kind = event.get("type")
            if kind == "session.input_audio.append":
                with self._audio_changed:
                    self._audio_changed.notify_all()
            if kind == "session.start" and not self.start_error:
                problem = start_problem(event.get("session") or {})
                if problem:
                    code, _, param = problem.partition(": ")
                    await ws.send(json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                                                         "code": code, "param": param}}))
                    await ws.close(1008)
                    return
            if kind == "session.start" and self.start_error:
                await ws.send(json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                                                     "code": self.start_error, "message": "refused"}}))
                await ws.close(1008)
                return
            if kind == "session.start":
                await ws.send(json.dumps({"type": "session.started", "session": {"id": "sess_fake", "model": event["session"]["model"]}}))
                self._started.set()
            elif kind == "session.close":
                await ws.send(json.dumps({"type": "session.closed", "reason": "close_requested", "session": {"id": "sess_fake"}, "usage": {"seconds": self.CLOSE_SECONDS}}))
                await ws.close()
                return

    # ------------------------------------------------------------ test helpers
    def push(self, *events: dict) -> None:
        """send server events, in order, now"""
        assert self._started.wait(5), "no session"

        async def send():
            for event in events:
                await self._ws.send(json.dumps(event))

        asyncio.run_coroutine_threadsafe(send(), self._loop).result(5)

    def push_audio(self, ulaw: bytes, frame_bytes: int = 160) -> None:
        self.push(*[
            {"type": "session.output_audio.delta", "delta": base64.b64encode(ulaw[i:i + frame_bytes]).decode()}
            for i in range(0, len(ulaw), frame_bytes)
        ])

    def drop(self) -> None:
        """close the socket with no session.closed, as a lost connection"""
        assert self._started.wait(5), "no session"
        asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop).result(5)

    def of_type(self, kind: str) -> list[dict]:
        return [e for _, e in self.received if e.get("type") == kind]

    def audio_frames(self) -> list[tuple[float, bytes]]:
        return [(t, base64.b64decode(e["audio"])) for t, e in self.received if e.get("type") == "session.input_audio.append"]

    def wait_for_audio(self, byte_count: int, timeout: float = 10) -> None:
        """Wait for actual socket receipt, not a client counter or a sleep."""
        with self._audio_changed:
            assert self._audio_changed.wait_for(
                lambda: sum(len(frame) for _, frame in self.audio_frames()) >= byte_count,
                timeout=timeout,
            ), "caller audio did not arrive"

    def session_start(self) -> dict:
        return self.of_type("session.start")[0]["session"]


def websockets_headers(**values: str):
    from websockets.datastructures import Headers

    return Headers(**values)
