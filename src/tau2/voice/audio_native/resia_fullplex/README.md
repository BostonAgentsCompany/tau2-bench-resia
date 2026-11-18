# Resia Fullplex

The `resia_fullplex` provider connects τ-voice to Resia Fullplex 1 (`resia-fullplex-1`)
through the Resia Live API. That API serves the OpenAI GPT-Live session protocol
over one WebSocket, so this provider is the OpenAI Live adapter with a WebSocket
transport. The benchmark's tools run in τ-voice: each backend function call comes
back in `response.event`, and τ-voice returns the result.

## Setup

Install the `live` extra (the `voice` extra plus `aiortc` and `av`, which the OpenAI
Live provider imports; pyaudio needs the PortAudio library). Set these variables
through your secret-management mechanism:

```bash
uv sync --extra live
export RESIA_API_KEY=...                       # a Resia organization API key (editor scope)
export RESIA_FULLPLEX_BACKEND_MODEL=...        # the delegated backend model; required, no default
export RESIA_FULLPLEX_URL=wss://...            # optional; default wss://api.resia.ai/v1/live/sessions
```

`--reasoning-effort` sets the backend's reasoning effort (`delegation.responses.reasoning`).
It is not sent when absent. `--audio-native-model` defaults to `resia-fullplex-1`.
`OPENAI_API_KEY` must be set too, as for every voice run: the simulator uses it, and
the OpenAI Live adapter base reads it.

## Evaluation

```bash
tau2 run --audio-native --audio-native-provider resia_fullplex --benchmark voice-v1
```

## Session

- Instructions: the benchmark's system prompt, unchanged, for the voice model and for
  the backend. The provider adds no text and registers no agent voice, so the run adds
  no voice-gender sentence. The server has one voice and ignores voice names.
- Audio: G.711 mu-law at 8 kHz in both directions, the benchmark's telephony format.
- Turn-taking: the server's own. No VAD settings are sent.
- `store` is not sent, so the server keeps no copy of the session.

## Audio transport

The provider sends caller audio in 20 ms frames, paced by the simulator's feeds. Audio
fed before the latest feed is due at once. The latest feed's audio is due at real
time from its arrival. The provider never adds silence. Its budget permits a
four-second burst and refills at 1.1 times real time. Audio waits when this
budget runs out. Each `send_audio` call waits for its caller frames to pass the
WebSocket send and drain operations. A zero write high-water mark prevents an
asyncio TCP buffer from remaining after send completion on plain loopback `ws`.
Secure `wss` uses high/low marks of `(1, 0)`: a zero high mark can pause an empty
TLS buffer on CPython 3.12.15. This drains the TLS layer only. The TCP buffer below
TLS retains its default 64 KiB high-water mark. Both paths can retain bytes in
the kernel. Only server input counters can confirm receipt or acceptance.
The provider collects received events through this wait.

A completed tick leaves no audio in the provider queue. A send failure, lost
socket, or cancelled tick can leave unsent bytes. Send failure or socket loss
marks the session as lost and raises `ConnectionError`. Cancellation propagates
`CancelledError`. The final media trace records `pending_bytes` and `max_pending_bytes`
for the provider queue. It excludes TLS, lower TCP, and kernel buffers. The per-send
`input_transmitted` record holds cumulative `fed_bytes` and that send's `transmit_s`.
The final `transmit_s` is the sum for completed sends; `max_transmit_s` is their maximum. These times
include normal frame pacing, budget waits, and socket waits; they do not isolate
backpressure. A 200 ms tick normally takes about 180 ms to submit all its frames.
Session close does not drain unsent bytes from an aborted tick.

Tau keeps its standard catch-up clock after a simulator stall. The transmission
wait now bounds that catch-up: a tick cannot advance while its own frames remain
in the provider queue. This can leave wall time behind the ideal tick schedule.
The provider preserves tau's output playback and metric code.

The server can refuse an append with `input_audio_rate_limit_exceeded`. That
append's audio is lost. The provider does not retry it, and the call continues.
The provider writes a warning and counts reported refusals in `rate_limit_errors`
in the final media trace. The server limits its error reports, so this counter
does not count every lost append or measure the duration of lost audio.

The server paces agent audio at real time with a lead of at most 200 ms. The Live
protocol has no "interrupted" or truncate event: when the voice model stops, its audio
stops. As in the OpenAI Live adapter, the provider plays the agent audio as it arrives
and classifies agent speech from the audio that each tick plays.

A function call outside a delegation (a server-side call tool, such as a hang-up)
is not a benchmark tool. The provider answers it with an error output.

## Usage

Each final backend response gives one `llm` usage record with the model that the
response reports. The server's session seconds give one cumulative `realtime` record.
