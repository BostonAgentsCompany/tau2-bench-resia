"""Behavior tests of the resia_fullplex tau provider on a local Live API server.

All transport tests use real WebSockets and the real clock, including the budget
and transmission waits, connection loss, cancellation, and traces.
No external service or real key is used.
"""

import asyncio
import audioop
import base64
import itertools
import json
import math
import time
from enum import StrEnum
from types import SimpleNamespace

import pytest
from fake_fullplex import FakeFullplex
from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.data_model.run_config import AudioNativeConfig
from tau2.environment.tool import as_tool
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.voice_gender import selected_agent_voice

TICK_MS = 200
TICK_BYTES = 8000 * TICK_MS // 1000          # 8 kHz mu-law: one byte per sample
POLICY = "# Airline policy\nAuthenticate the user by user ID."


def tone(ms: int, freq: float = 440.0) -> bytes:
    """8 kHz mu-law sine at about half scale"""
    n = 8000 * ms // 1000
    pcm = b"".join(int(12000 * math.sin(2 * math.pi * freq * i / 8000)).to_bytes(2, "little", signed=True)
                   for i in range(n))
    return audioop.lin2ulaw(pcm, 2)


SILENCE_TICK = b"\x7f" * TICK_BYTES
LATE_S = 0.15        # thread, loop, and socket delay that a frame may add on a busy host


def wait_until(condition, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.02)


@pytest.fixture
def env(monkeypatch):
    def setup(server: FakeFullplex, backend: str = "example-backend-a"):
        monkeypatch.setenv("RESIA_FULLPLEX_URL", server.url)
        monkeypatch.setenv("RESIA_API_KEY", server.key)
        monkeypatch.setenv("RESIA_FULLPLEX_BACKEND_MODEL", backend)
        monkeypatch.setenv("OPENAI_API_KEY", "unused-by-these-tests")
    return setup


def adapter_for(reasoning_effort=None):
    adapter, model = create_adapter("resia_fullplex", tick_duration_ms=TICK_MS, reasoning_effort=reasoning_effort)
    return adapter, model


def get_user_details(user_id: str) -> str:
    """Get the details of a user.

    Args:
        user_id: The user ID.
    """
    return user_id


# ---------------------------------------------------------------- prompt, gender, registration

def test_agent_sends_taus_plain_prompt_to_both_models_with_no_added_text(env):
    with FakeFullplex() as server:
        env(server)
        agent = DiscreteTimeAudioNativeAgent(tools=[as_tool(get_user_details)], domain_policy=POLICY,
                                             tick_duration_ms=TICK_MS, provider="resia_fullplex")
        agent.get_init_state()
        try:
            session = server.session_start()
        finally:
            agent.adapter.disconnect()
    prompt = agent.system_prompt
    assert prompt.startswith("You are a customer service agent handling a voice call")
    assert prompt.endswith(POLICY)
    assert session["instructions"] == prompt
    assert session["delegation"]["responses"]["instructions"] == prompt
    assert session["model"] == "resia-fullplex-1"
    assert session["audio"] == {"format": {"type": "audio/pcmu", "rate": 8000}}
    responses = session["delegation"]["responses"]
    assert responses["model"] == "example-backend-a" and "reasoning" not in responses
    assert [t["name"] for t in responses["tools"]] == ["get_user_details"]
    assert set(session) == {"model", "instructions", "audio", "delegation"}   # no store, no voice, no extras
    assert server.auth_headers == [f"Bearer {server.key}"]


def test_reasoning_effort_reaches_the_backend_only(env):
    with FakeFullplex() as server:
        env(server, backend="example-backend-b")
        adapter, _ = adapter_for("medium")
        adapter.connect("PROMPT", [])
        adapter.disconnect()
        session = server.session_start()
    assert session["delegation"]["responses"]["reasoning"] == {"effort": "medium"}
    assert session["delegation"]["responses"]["model"] == "example-backend-b"
    assert "reasoning" not in session


def test_no_voice_gender_and_realtime_generation_on():
    config = AudioNativeConfig(provider="resia_fullplex")
    assert selected_agent_voice(SimpleNamespace(audio_native_config=config)) == (None, None)
    assert config.realtime_generation_enabled is True


def test_settings_are_required_and_the_key_needs_tls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setenv("RESIA_API_KEY", "k")
    monkeypatch.delenv("RESIA_FULLPLEX_BACKEND_MODEL", raising=False)
    with pytest.raises(ValueError, match="RESIA_FULLPLEX_BACKEND_MODEL"):
        adapter_for()
    monkeypatch.setenv("RESIA_FULLPLEX_BACKEND_MODEL", "example-backend-a")
    monkeypatch.setenv("RESIA_FULLPLEX_URL", "ws://example.com/v1/live/sessions")
    with pytest.raises(ValueError, match="wss://"):
        adapter_for()


def test_a_refused_session_raises_without_the_key(env):
    with FakeFullplex(refuse_status=429) as server:
        env(server)
        adapter, _ = adapter_for()
        with pytest.raises(RuntimeError) as error:
            adapter.connect("PROMPT", [])
    assert "HTTP 429" in str(error.value)
    assert server.key not in str(error.value)


# ---------------------------------------------------------------- pacing

def test_feed_paced_sending_after_late_ticks_and_a_stall(env):
    """Each tick waits for its caller audio. A simulator stall adds no fill silence."""
    ticks = [tone(TICK_MS) for _ in range(12)]
    calls: list[float] = []
    with FakeFullplex() as server:
        env(server)
        adapter, _ = adapter_for()
        adapter.connect("PROMPT", [])
        try:
            for k, audio in enumerate(ticks):
                if k == 6:
                    time.sleep(0.7)                  # a simulator stall between ticks
                elif k in (2, 9):
                    time.sleep(0.06)                 # late ticks
                calls.append(time.monotonic())
                adapter.run_tick(audio, tick_number=k)
            wait_until(lambda: sum(len(f) for _, f in server.audio_frames()) >= sum(map(len, ticks)) - 4)
        finally:
            adapter.disconnect()
        frames = server.audio_frames()
    sent = b"".join(f for _, f in frames)
    fed = sum(len(a) for a in ticks)
    assert abs(len(sent) - fed) <= 4                 # resampling edges only; no fill silence
    assert all(len(f) <= 160 for _, f in frames)
    # each frame leaves when it is due, not before and not much later: a byte of tick k is due at its feed plus its
    # offset. The adapter holds the next feed until this tick completes.
    # A late tick cannot advance until its own audio completes its socket send.
    assert all(b - a >= 0.15 for a, b in itertools.pairwise(calls[6:]))
    offset = 0
    for t, frame in frames:
        k = min(offset // TICK_BYTES, len(calls) - 1)
        due = calls[k] + (offset - k * TICK_BYTES) / 8000
        if k + 1 < len(calls):
            due = min(due, calls[k + 1])
        assert due - 0.005 <= t <= due + LATE_S, (offset, t - due)
        offset += len(frame)
    # the stall sends nothing: no audio arrives in its middle
    stall_start, stall_end = calls[5] + TICK_MS / 1000 + LATE_S, calls[6] - 0.05
    assert not [t for t, _ in frames if stall_start < t < stall_end]


def test_tls_sends_caller_audio_without_server_messages(tls_context):
    """A quiet TLS peer must not leave the producer paused on an empty buffer."""
    from tau2.voice.audio_native.resia_fullplex import provider

    with FakeFullplex(ssl_context=tls_context) as server:
        pcm, _ = audioop.ratecv(audioop.ulaw2lin(tone(TICK_MS), 2), 2, 1, 8000, 24000, None)
        converted, _ = audioop.ratecv(pcm, 2, 1, 24000, 8000, None)
        expected = audioop.lin2ulaw(converted, 2)

        async def run():
            client = provider.ResiaFullplexProvider(
                model="resia-fullplex-1", backend_model="test", api_key=server.key,
                url=server.url.replace("wss://", "WSS://", 1),  # URI schemes are case-insensitive.
            )
            try:
                async with asyncio.timeout(2):
                    await client.configure_session("PROMPT", [])
                    await client.send_audio(pcm)
            finally:
                await client.disconnect()

        asyncio.run(run())
        assert b"".join(frame for _, frame in server.audio_frames()) == expected


# ---------------------------------------------------------------- agent audio, transcript, tools, usage

def test_agent_audio_and_its_text_play_on_the_tick_clock(env):
    with FakeFullplex() as server:
        env(server)
        adapter, _ = adapter_for()
        adapter.connect("PROMPT", [])
        try:
            server.push({"type": "session.output_transcript.delta", "delta": "Hello, how can I help?",
                         "start_ms": 0, "end_ms": 0})
            server.push_audio(tone(300))
            server.push({"type": "session.input_transcript.delta", "delta": "wait", "start_ms": 0, "end_ms": 200})
            server.push_audio(tone(300))                  # the caller speaks; the server goes on, so the audio plays
            results = [adapter.run_tick(SILENCE_TICK, tick_number=k) for k in range(6)]
        finally:
            adapter.disconnect()
    played = sum(len(r.agent_audio_data) for r in results)
    assert abs(played - 4800) <= 16                       # 600 ms at 8 kHz mu-law, no more, no less
    assert all(len(r.get_played_agent_audio()) == TICK_BYTES for r in results)
    assert any(r.contains_speech for r in results) and not results[-1].contains_speech
    assert "".join(r.proportional_transcript for r in results) == "Hello, how can I help?"


def test_backend_tool_batch_continues_once_after_all_results(env):
    def item(call_id):
        return {"type": "response.event", "delegation_id": "d1", "event": {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": call_id, "name": "get_user_details",
                     "arguments": '{"user_id": "example_user_1"}'}}}
    completed = {"type": "response.event", "delegation_id": "d1", "event": {
        "type": "response.completed", "response": {"id": "r1", "model": "example-backend-a-2026-01-01", "usage": {
            "input_tokens": 1000, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 800}}}}}
    with FakeFullplex() as server:
        env(server)
        adapter, _ = adapter_for()
        adapter.connect("PROMPT", [as_tool(get_user_details)])
        try:
            server.push({"type": "session.delegation.created", "delegation": {"id": "d1", "target": "responses"}},
                        {"type": "response.event", "delegation_id": "d1",
                         "event": {"type": "response.created", "response": {"id": "r1"}}},
                        item("c1"), item("c2"), completed, completed)
            first = adapter.run_tick(SILENCE_TICK, tick_number=0)
            adapter.send_tool_result("c1", "user 1")
            adapter.send_tool_result("c2", "user 2")
            adapter.run_tick(SILENCE_TICK, tick_number=1)
        finally:
            adapter.disconnect()
        outputs = [e["item"]["call_id"] for e in server.of_type("response.item.create")]
        creates = server.of_type("response.create")
    assert [c.id for c in first.tool_calls] == ["c1", "c2"]
    assert first.tool_calls[0].arguments == {"user_id": "example_user_1"}
    assert outputs == ["c1", "c2"] and len(creates) == 1
    llm = [r for r in first.usage_records if r.component == "llm"]
    assert len(llm) == 1                                   # the duplicate final event counts once
    assert (llm[0].model, llm[0].input_tokens, llm[0].output_tokens, llm[0].input_cached_tokens) == \
        ("example-backend-a-2026-01-01", 1000, 50, 800)


def test_session_seconds_become_one_cumulative_record_with_the_final_value_from_the_close(env):
    with FakeFullplex() as server:
        env(server)
        adapter, _ = adapter_for()
        adapter.connect("PROMPT", [])
        try:
            server.push({"type": "session.usage.updated", "usage": {"seconds": 15.0}})
            adapter.run_tick(SILENCE_TICK, tick_number=0)
        finally:
            adapter.disconnect()
    realtime = [r for r in adapter.get_usage_records() if r.component == "realtime"]
    assert {(r.provider, r.model, r.semantics, r.scope_id) for r in realtime} == \
        {("resia_fullplex", "resia-fullplex-1", "cumulative", "sess_fake")}
    assert [r.audio_input_seconds for r in realtime] == [15.0, FakeFullplex.CLOSE_SECONDS]   # the last value wins


def test_a_call_outside_a_delegation_is_refused_and_not_a_tau_tool(env):
    with FakeFullplex() as server:
        env(server)
        adapter, _ = adapter_for()
        adapter.connect("PROMPT", [])
        try:
            server.push({"type": "response.event", "delegation_id": None, "event": {
                "type": "response.output_item.done",
                "item": {"type": "function_call", "call_id": "h1", "name": "hangup", "arguments": "{}"}}})
            result = adapter.run_tick(SILENCE_TICK, tick_number=0)
        finally:
            adapter.disconnect()
        outputs = server.of_type("response.item.create")
    assert result.tool_calls == []
    assert [o["item"]["call_id"] for o in outputs] == ["h1"] and "error" in outputs[0]["item"]["output"]


def test_a_server_error_event_raises_from_the_tick(env):
    with FakeFullplex() as server:
        env(server)
        adapter, _ = adapter_for()
        adapter.connect("PROMPT", [])
        try:
            server.push({"type": "error", "error": {"type": "server_error", "code": "engine_error", "message": "x"}})
            with pytest.raises(RuntimeError, match="engine_error"):
                adapter.run_tick(SILENCE_TICK, tick_number=0)
        finally:
            adapter.disconnect()


# ---------------------------------------------------------------- key protection, start, loss, reconnect, trace

def test_a_redirect_is_refused_and_the_key_never_reaches_its_target(env):
    with FakeFullplex() as target, FakeFullplex(redirect_to="") as origin:
        origin.redirect_to = target.url
        env(origin)
        adapter, _ = adapter_for()
        with pytest.raises(RuntimeError) as error:
            adapter.connect("PROMPT", [])
    assert "HTTP 302" in str(error.value) and origin.key not in str(error.value)
    assert target.auth_headers == []                      # the target got no request at all


@pytest.mark.parametrize("key", ["", "a b", "a\r\nX-Injected: y", "été", "a" * 4097])
def test_an_invalid_key_is_refused_before_any_connection(env, monkeypatch, key):
    with FakeFullplex() as server:
        env(server)
        monkeypatch.setenv("RESIA_API_KEY", key)
        with pytest.raises(ValueError) as error:
            adapter_for()
    assert server.auth_headers == []
    assert not key or key not in str(error.value)


def test_endpoint_rules(monkeypatch):
    from tau2.config import DEFAULT_RESIA_FULLPLEX_URL
    from tau2.voice.audio_native.resia_fullplex.provider import resolve_endpoint
    monkeypatch.delenv("RESIA_FULLPLEX_URL", raising=False)
    assert resolve_endpoint() == DEFAULT_RESIA_FULLPLEX_URL
    assert resolve_endpoint("ws://localhost:8193/v1/live/sessions").startswith("ws://localhost")
    assert resolve_endpoint("ws://127.0.0.1:8193/v1/live/sessions").startswith("ws://127.0.0.1")
    for url in ("https://api.resia.ai/v1/live/sessions", "ws://10.0.0.5/v1/live/sessions", "wss://"):
        with pytest.raises(ValueError):
            resolve_endpoint(url)


def test_a_refused_session_start_reports_the_server_error(env):
    with FakeFullplex(start_error="invalid_value") as server:
        env(server)
        adapter, _ = adapter_for()
        with pytest.raises(RuntimeError, match="invalid_value"):
            adapter.connect("PROMPT", [])
    assert not adapter.is_connected


def test_a_lost_connection_ends_the_call_without_the_close_wait(env):
    with FakeFullplex() as server:
        env(server)
        adapter, _ = adapter_for()
        adapter.connect("PROMPT", [])
        server.drop()
        wait_until(lambda: adapter.provider._closed.is_set())
        with pytest.raises((RuntimeError, ConnectionError), match="connection_lost"):
            adapter.run_tick(SILENCE_TICK, tick_number=0)
        started = time.monotonic()
        adapter.disconnect()
    assert time.monotonic() - started < 2.0


def test_a_reconnect_starts_a_new_session_with_the_same_settings(env):
    with FakeFullplex() as server:
        env(server, backend="example-backend-b")
        adapter, _ = adapter_for("medium")
        adapter.connect("PROMPT", [])
        adapter.disconnect()
        adapter.connect("PROMPT", [])
        adapter.disconnect()
        starts = [e["session"]["delegation"]["responses"] for e in server.of_type("session.start")]
    assert [(s["model"], s["reasoning"]) for s in starts] == [("example-backend-b", {"effort": "medium"})] * 2


def test_the_trace_holds_the_session_and_the_audio_counts_but_no_key(env, tmp_path):
    trace = tmp_path / "live.jsonl"
    ticks = [tone(TICK_MS) for _ in range(3)]
    tick_durations = []
    with FakeFullplex() as server:
        env(server)
        adapter, _ = create_adapter("resia_fullplex", tick_duration_ms=TICK_MS, trace_path=trace)
        adapter.connect("PROMPT", [])
        try:
            server.push({"type": "error", "error": {"type": "invalid_request_error",
                         "code": "input_audio_rate_limit_exceeded", "message": "dropped"}})
            for k, audio in enumerate(ticks):
                started = time.monotonic()
                adapter.run_tick(audio, tick_number=k)
                tick_durations.append(time.monotonic() - started)
            wait_until(lambda: sum(len(f) for _, f in server.audio_frames()) >= sum(map(len, ticks)) - 4)
        finally:
            adapter.disconnect()
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    config = next(r["event"] for r in records if r["direction"] == "configuration")
    media = next(r["event"] for r in records if r["direction"] == "media")
    assert config["instructions"] == "PROMPT"
    assert media["sent_bytes"] == media["fed_bytes"] and abs(media["fed_bytes"] - 3 * TICK_BYTES) <= 4
    transmitted = [r["event"] for r in records if r["direction"] == "input_transmitted"]
    assert len(transmitted) == len(ticks)
    assert media["pending_bytes"] == 0
    assert media["max_pending_bytes"] <= TICK_BYTES
    assert media["rate_limit_errors"] == 1
    # The last 20 ms frame is due 180 ms after a 200 ms tick starts.
    for record, wall_duration in zip(transmitted, tick_durations, strict=True):
        assert 0.179 <= record["transmit_s"] <= wall_duration
    assert len(ticks) * 0.179 <= media["transmit_s"] <= sum(tick_durations)
    assert 0.179 <= media["max_transmit_s"] <= max(tick_durations)
    assert transmitted[-1]["fed_bytes"] == sum(len(f) for _, f in server.audio_frames())
    assert server.key not in trace.read_text()


# ---------------------------------------------------------------- the Live API's input audio budget

def test_a_long_stall_sends_its_late_audio_within_the_audio_budget():
    """Real arrivals preserve a late burst and respect the client's refill rate."""
    from tau2.voice.audio_native.resia_fullplex import provider

    audio = tone(9000)
    with FakeFullplex() as server:
        async def run():
            client = provider.ResiaFullplexProvider(
                model="resia-fullplex-1", backend_model="test", api_key=server.key, url=server.url,
            )
            await client.configure_session("PROMPT", [])
            try:
                started = time.monotonic()
                # Supply the saved catch-up shape without a timer substitution.
                # The feed API makes every earlier chunk due immediately.
                for offset in range(0, len(audio), TICK_BYTES):
                    client.feed(audio[offset:offset + TICK_BYTES])
                await asyncio.to_thread(server.wait_for_audio, len(audio))
                frames = server.audio_frames()
                assert b"".join(frame for _, frame in frames) == audio
                # The client must spend at least five seconds of refill credit.
                elapsed = frames[-1][0] - started
                assert (9 - 4) / 1.1 - 0.005 <= elapsed <= (9 - 4) / 1.1 + 2
                # Check the cumulative envelope, allowing 5 ms for clock reads.
                received = 0
                for arrived, frame in frames:
                    received += len(frame)
                    assert received / 8000 <= 4 + 1.1 * (arrived - started + 0.005)
                assert client.is_connected
            finally:
                await client.disconnect()

        asyncio.run(run())


def test_a_dropped_append_error_does_not_end_the_call(env):
    with FakeFullplex() as server:
        env(server)
        adapter, _ = adapter_for()
        adapter.connect("PROMPT", [])
        try:
            server.push({"type": "error", "error": {"type": "invalid_request_error",
                                                    "code": "input_audio_rate_limit_exceeded", "message": "dropped"}})
            adapter.run_tick(SILENCE_TICK, tick_number=0)
            adapter.run_tick(SILENCE_TICK, tick_number=1)
            assert adapter.is_connected
        finally:
            adapter.disconnect()
    assert adapter.provider.rate_limit_errors == 1


class TransmissionOutcome(StrEnum):
    COMPLETE = "complete"
    CONNECTION_LOSS = "connection_loss"
    CANCEL = "cancel"


@pytest.mark.parametrize("outcome", list(TransmissionOutcome))
def test_transmission_waits_and_collects_events_on_real_socket(tmp_path, outcome):
    """Caller pacing holds the producer while real server output remains available."""
    from tau2.voice.audio_native.resia_fullplex import provider
    from tau2.voice.audio_native.openai.events import AudioDeltaEvent

    trace = tmp_path / "transmission.jsonl"
    with FakeFullplex() as server:
        async def run():
            client = provider.ResiaFullplexProvider(
                model="resia-fullplex-1", backend_model="test", api_key=server.key,
                url=server.url, trace_path=trace,
            )
            await client.configure_session("PROMPT", [])
            pcm = audioop.ulaw2lin(tone(2000), 2)
            pcm, _ = audioop.ratecv(pcm, 2, 1, 8000, 24000, None)
            converted, _ = audioop.ratecv(pcm, 2, 1, 24000, 8000, None)
            expected = audioop.lin2ulaw(converted, 2)
            started = time.monotonic()
            send = asyncio.create_task(client.send_audio(pcm))
            receive = asyncio.create_task(client.receive_events_for_duration(0))
            try:
                await asyncio.to_thread(server.wait_for_audio, 160)
                assert not send.done()
                await asyncio.to_thread(server.push_audio, tone(20))
                # Give the real receiver one turn; it must not finish before input.
                await asyncio.sleep(0)
                assert not receive.done()
                if outcome == TransmissionOutcome.CANCEL:
                    send.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await send
                elif outcome == TransmissionOutcome.CONNECTION_LOSS:
                    await asyncio.to_thread(server.drop)
                    with pytest.raises(ConnectionError):
                        await asyncio.wait_for(send, 3)
                    assert not client.is_connected
                else:
                    await asyncio.wait_for(send, 5)
                    wall_duration = time.monotonic() - started
                    events = await asyncio.wait_for(receive, 3)
                    assert any(isinstance(event, AudioDeltaEvent) for event in events)
                    await asyncio.to_thread(server.wait_for_audio, len(expected))
            finally:
                receive.cancel()
                send.cancel()
                await client.disconnect()
                await asyncio.gather(receive, send, return_exceptions=True)
            accepted = b"".join(frame for _, frame in server.audio_frames())
            assert accepted == expected[:len(accepted)]
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            media = next(r["event"] for r in records if r["direction"] == "media")
            transmitted = [r["event"] for r in records if r["direction"] == "input_transmitted"]
            assert media["fed_bytes"] == len(expected)
            assert media["max_pending_bytes"] == len(expected)
            assert media["rate_limit_errors"] == 0
            if outcome == TransmissionOutcome.COMPLETE:
                assert accepted == expected
                assert media["sent_bytes"] == len(accepted)
                assert media["pending_bytes"] == 0
                assert len(transmitted) == 1
                assert transmitted[0]["fed_bytes"] == len(expected)
                # Two seconds of audio have a last frame due at 1.98 seconds.
                for duration in (transmitted[0]["transmit_s"], media["transmit_s"], media["max_transmit_s"]):
                    assert 1.979 <= duration <= wall_duration
            else:
                assert 0 < len(accepted) < len(expected)
                # A send can complete locally before socket loss prevents receipt.
                assert len(accepted) <= media["sent_bytes"] < len(expected)
                assert 0 < media["pending_bytes"] <= len(expected) - len(accepted)
                assert not transmitted
                assert media["transmit_s"] == media["max_transmit_s"] == 0

        asyncio.run(run())
