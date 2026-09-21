# Milestone 3 — remote edge (brain server + audio satellite): plan

**Status:** Planned
**Date:** 2026-09-21
**Owner:** Kevin Lundell
**Branch:** `milestone-3-remote-edge` off `develop`, to be merged back —
same workflow as `milestone-2-skill-factory`.
**Reference implementation:** [SBRA-Dynamics/Mike](https://github.com/SBRA-Dynamics/Mike)
— its PRD 5a (voice), PRD 6 (latency / hold window) and PRD 8.3 (the voder).

This is the implementation plan for the Milestone 3 item in
`jarvis-2026-rebuild.md`. JARVIS today runs as one process on one machine. The
mic, wake word, VAD, faster-whisper, NLU/skills/persona, Piper and the speaker
are all wired together in `app.build_orchestrator`. This milestone splits it:

- the **brain** (`python -m jarvis serve`) runs on a server with a good GPU. It
  handles NLU, skills, persona, reasoner and skill factory, and it calls two
  warm local services next to it: **`jarvis-whisper`** (STT on the GPU) and
  **`jarvis-voder`** (TTS);
- the **edge** (`python -m jarvis edge`) runs on a Raspberry Pi-class device in
  the room. It owns the microphone and the speaker, cuts what it hears into
  speech **segments**, sends them to the brain, and plays back the spoken
  answers.

The link may cross **the internet**, not just a home LAN.

**The voice pipeline is Mike's, ported, not reinvented.** Mike's version is a
working Node system: a pure client-side segmenter, one message per segment,
Whisper and TTS as separate loopback services, transcript-based addressing
modes instead of a wake word, and a hold window that merges split sentences.
It has been tuned against real rooms, including the noise-floor failure
described in Mike's PRD 6 §8. Each decision below names the Mike file it comes
from. Where JARVIS deliberately differs, it says so.

M1 and M2 already left the right seam in place for the brain side. The
orchestrator never touches hardware directly. It duck-types four injected roles
(`wake`, `mic`, `stt`, `tts`), and `jarvis/audio/text_io.py` already runs the
whole brain from queued *text*. The remote path is a third implementation of
those roles, and the orchestrator is **not** rewritten. No commits happen
without the user asking, per the repo's standing convention.

---

## Key design decisions (resolving PRD ambiguity)

### 1. Topology

```
Pi (edge)                                     GPU server (brain)
─────────                                     ─────────────────────────────────────────
mic ─► Segmenter ─► speaking{on}  ─── wss ──►  intake ─► jarvis-whisper (127.0.0.1, GPU)
                 └► audio{pcm}    ───────────►   │           │ text
                                                 ▼           ▼
                                             heard{text} ◄─ addressing gate + hold window
                                                             │
speaker ◄─ play parts ◄─ speech{part,pcm} ◄──  voder ◄─ Orchestrator (NLU, skills, persona)
PTT button ─► control / interrupt ───────────►               via RemoteLink (like TextIO)
```

| Layer | Edge (Pi) | Brain (GPU server) |
|---|---|---|
| Mic capture, resample to 16 kHz s16le mono | ✓ | — |
| Segmentation (energy VAD, adaptive floor) | ✓ `audio/segment.py` | — |
| Push-to-talk button | ✓ | — |
| STT | — | ✓ `jarvis-whisper` service (faster-whisper, GPU) |
| Addressing gate, mode commands, hold window | — | ✓ |
| NLU, skills, persona, reasoner, factory, knowledge | — | ✓ unchanged |
| TTS synthesis | — | ✓ `jarvis-voder` service (Piper) |
| Playback | ✓ | — |

- **The edge has no wake word and no models.** It installs only `numpy`,
  `sounddevice` and `websockets`. This is Mike's design: *"the addressing modes
  are the wake word"* (Mike PRD 5a, Non-goals).
- **All-in-one stays the default.** `python -m jarvis` keeps its local
  openwakeword path and `text` mode is unchanged. The remote-edge topology is
  opt-in. The canonical PRD's "No network on the hot path" principle is scoped
  to the all-in-one topology. The remote topology puts a *self-hosted* link on
  the hot path, and there is still no third-party cloud. Mike's PRD 0 has the
  same rule: *no speech sent to an external API*.

### 2. Segmenter: port Mike's `client/src/audio/segment.ts` to `jarvis/audio/segment.py`

A **pure** class: `Segmenter(opts).push(frames) -> list[Segment]`, plus
`flush(reason)`, `reset()` and a `speaking` property. It uses no device API and
no clock; everything is counted in samples. A recorded file therefore gives
bit-identical segments, and the tests can drive it without a mic (Mike 5a
acceptance criterion 7).

- **Detector:** RMS of 20 ms frames in dBFS. A frame is speech when it is
  `margin_db` (9) above the noise floor.
- **Noise floor = the quietest frame in the last `floor_window_ms` (1500).**
  This is minimum statistics. Speech has gaps between words, so this minimum is
  the room, whatever is being said. The floor falls fast (`floor_fall_rate`
  0.5) and rises slowly (0.004), and it may also rise while a segment is open
  (`floor_rise_rate_open` 0.02), so a segment that opened on room tone closes
  itself. Frames at or below `gap_db` (−90) are dropouts, not room tone, and are
  ignored. The floor is clamped to [−70, −25] dB.
  This fixes Mike's measured failure (PRD 6 §8). There, one frame of zeros
  dropped the floor to its minimum, room tone turned into "speech", and every
  segment ran to the 15 s maximum, cut mid-word.
- **Segment shape:**
  - It opens after `start_ms` (100) of speech and keeps `pre_roll_ms` (300)
    before that, so the first consonant survives, and `tail_ms` (250) after.
  - It closes after `hangover_ms` (700) of silence. That is longer than a pause
    between clauses and shorter than one between sentences.
  - Less than `min_speech_ms` (250) of actual speech is dropped without being
    sent.
  - It is capped at `max_segment_ms` (15 000). A segment that reaches the cap is
    **cut at the most recent pause of at least `cut_gap_ms` (200)**, not at an
    arbitrary sample, and whatever follows the pause starts the next segment.
- **Every segment carries** `reason` (`silence` / `maximum` / `release` /
  `close`), `floor_db`, `peak_db`, `start_ms`, `end_ms` and `speech_ms`, for
  the server's log.
- **Push-to-talk:** in `hold` mode, silence doesn't close a segment;
  `flush("release")` does. The detector still trims silence at both ends.
- Defaults are Mike's `DEFAULTS`, copied verbatim and exposed as `[edge.segment]`
  config. Mike's comments explaining *why* each value is what it is get ported
  alongside, because they are the tuning record.
- The local all-in-one mic path (`Microphone.record_utterance`,
  `jarvis/audio/capture.py`, which calibrates its floor once per utterance and
  has no pre-roll) **keeps working unchanged in M3**. Switching it to the
  segmenter is listed under polish (M6).

### 3. Wire protocol: port Mike's `src/protocol.js` to `jarvis/remote/protocol.py`

There is one shared schema file, imported by both edge and brain, so the two
can't drift (Mike PRD 1 R1.6). **JSON text frames only, with PCM as base64**,
as in Mike. Binary frames are refused. The ~33 % base64 overhead is irrelevant
at speech segment sizes, and one message per segment keeps the protocol
stateless.

`PROTOCOL_VERSION = 1`. The format is fixed at **16 kHz, signed 16-bit
little-endian, mono**. A segment at any other rate is a named error, not a
stream transcribed at the wrong speed.

**Edge → brain (C2S)**

| type | fields | notes |
|---|---|---|
| `hello` | `protocol, token, device_id` | must be the first message, within 10 s |
| `audio` | `pcm` (base64), `final?, reason?, floor_db?, peak_db?` | one message per segment |
| `speaking` | `on: bool` | sent the moment the segmenter opens or closes, *before* the audio (Mike PRD 6 §7) |
| `interrupt` | — | the button's cancel, or the spoken "stop" |
| `control` | `action, args` | `setMode`, `setSpeech{on, sample_rate}`, `mic{on}`, `playbackDone{id}`, `clientLog{level, text}` |

**Brain → edge (S2C)**

| type | fields | notes |
|---|---|---|
| `ready` | `protocol, mode, speech` | the answer to `hello` |
| `heard` | `text, confidence` | what was understood, sent **before** routing |
| `state` | `value, mode, mic` | `idle` / `listening` / `thinking` / `acting` / `speaking`, plus the addressing mode |
| `text` | `text, from` | the answer as text, for logs or a future display |
| `speech` | `id, part, text, pcm, sample_rate, final` | one synthesized sentence (decision 8) |
| `event` | `kind, data` | e.g. `stopPlayback`, `mic{on}` |
| `error` | `message, fatal` | one line, never a stack trace |

**Close codes:** 4001 unauthorized, 4002 bad protocol (including no or late
`hello`), 4003 bad message, 4004 server shutdown.

`validate_c2s(raw) -> (ok, msg | error)` **never raises**: its input is whatever
arrived from the internet. A malformed message *before* `hello` closes the
socket. *After* `hello`, it gets an `error` reply and the connection stays up:
one bad frame shouldn't cost the user their conversation.

### 4. Server intake: port Mike's `src/audio.js` + `createAudioIntake` to `jarvis/remote/intake.py`

`decode_segment(msg)` checks bounds *before* allocating anything:
- the base64 length must be within `MAX_AUDIO_BASE64` (2 000 000, about 47 s)
  before decoding;
- the decoded size must be within `max_audio_bytes` (1 000 000, about 31 s,
  twice the edge's own 15 s cap);
- the base64 must be strict, because `base64.b64decode` without `validate=True`
  silently skips junk;
- the byte count must be even (whole 16-bit samples);
- **a segment under 120 ms is ignored silently.**

Then:
- **An empty transcription produces nothing:** no turn, no `heard`, and no
  "I didn't catch that". Silence mistaken for speech must cost the user nothing
  (Mike R5a.5). This is a deliberate difference from the all-in-one path's
  "didn't catch that" line, which only fires on *low-confidence* text there.
- **`heard` is emitted before routing**, even when the addressing gate then
  drops the utterance. "It heard me and decided I wasn't talking to it" and "it
  didn't hear me" are different problems with different fixes (Mike R5a.8).
- **One diagnostic log line per segment**, e.g.
  `heard 2300ms silence floor -52 peak -18 in 340ms (model 300ms) en "search black holes"`.
  Mike's lesson: *"15000ms maximum floor -70 peak -38" is a diagnosis;
  "15000ms" was a mystery.*

### 5. Addressing replaces the wake word on the remote path: port Mike's R5a.4 to `jarvis/remote/addressing.py`

Four modes decide what reaches the orchestrator. The table covers **spoken**
input only.

| mode | what reaches JARVIS | edge mic |
|---|---|---|
| **ByName** (default) | only utterances starting with "Jarvis" / "hey Jarvis" (the name is stripped), plus anything inside a conversation window | open |
| **Always** | everything said | open |
| **PushToTalk** | only what is said while the button is held (no name needed) | **off** until held |
| **Ignore** | nothing (input paused) | open, output discarded |

- **The mode commands are always live, in every mode including Ignore.** They
  are matched *before* the gate, never after it, so the user can never reach a
  state they can't speak their way out of:
  - "Jarvis, pause input" → Ignore; "Jarvis, continue input" → the mode used
    before pausing (which may be PushToTalk: resuming must never open a mic
    that PTT had closed).
  - "Jarvis, change input to always / by name / push to talk".
  - "turn on the mic" / "turn off the mic": the brain relays this to the edge
    as `event mic{on}`. It is transient and never replayed.
- **Matching tolerates STT output:** casing, punctuation, "hey Jarvis" or bare
  "Jarvis", and a leading comma. Matching happens on the server, on the whole
  transcript. The same matcher works on typed text in `text` mode.
- **Conversation window:** after JARVIS replies, and during any `_ask` prompt,
  ByName accepts the next utterance **without** the name. This is exactly the
  orchestrator's existing `follow_up_s` / `_ask` window, now driven by
  transcripts. It never applies in Ignore or PushToTalk.
- **PushToTalk** uses a physical button on the Pi (GPIO, or a key on a USB
  keypad). Holding it opens the mic; releasing it flushes the segment
  (`reason: "release"`). The button must *never* be conditional on connection
  state or anything else, because in PTT it is the only way in.
- The mode is **per edge device**. It persists in `data/remote/<device_id>.json`
  across brain restarts, defaults from `[addressing] default_mode`, and is part
  of every `state` message.

### 6. Hold window: port Mike's PRD 6 §4 and §7 (in `jarvis/remote/addressing.py`)

The segmenter closes after 700 ms of silence. That is right for "has the
person stopped talking" and wrong for "has the person finished the thought".
So:

- Addressed fragments are **held** for `hold_ms` (2000; 0 disables this) and
  merged into one utterance. An un-addressed fragment may join a hold that is
  already open, because the user said the name once, as people do.
- **The countdown is silence, not clock.** While the edge reports
  `speaking:on`, the hold doesn't count down. When the next transcript arrives,
  the full `hold_ms` restarts. The `speaking` flag is remembered per device,
  not per turn, because it usually arrives *before* the window exists: the
  user starts the second half while the first is still in Whisper.
- **A cap** of 20 s (`SPEAKING_CAP_MS`), counted from the last words that
  actually arrived, releases whatever is held, so a detector stuck open can't
  hold a turn forever. A closed socket clears the flag.
- `state` distinguishes *still listening* (holding words) from *thinking* (the
  orchestrator has the whole utterance).

### 7. Brain integration: `RemoteLink`, the `TextIO` pattern with no orchestrator rewrite

`RemoteLink` (`jarvis/remote/server.py`) plays the orchestrator's `wake`, `mic`,
`stt` and `tts` roles, just as `TextIO` plays `mic` and `stt` from queued text.
The segments are already transcribed, gated and merged by the time the
orchestrator sees them:

- `triggered(frame)` is True when an *addressed* utterance is queued; the
  equivalent of the wake word firing. `read()` returns silent frames so
  `_await_wake`'s loop runs unchanged, just as `TextWake` does today.
- `record_utterance(window, …)` pops the next queued utterance. Inside a
  session or `_ask`, the conversation window applies: it takes the next
  utterance with or without the name, and returns empty audio (silence) if
  none arrives within `window`. `transcribe()` returns that utterance's text,
  and `last_avg_logprob` comes from Whisper's confidence.
- `say(text)` goes to the voder (decision 8). It blocks until the edge sends
  `control playbackDone{id}`, or until an interrupt arrives. That keeps the
  orchestrator's half-duplex assumption, and `drain()` is a no-op.
- **Barge-in and cancel:**
  - `interrupt` from the edge goes to a new public
    `Interrupter.trigger_cancel()` (`jarvis/core/interrupt.py`, the same path
    as Enter via `_on_stdin`).
  - **Stopping a reply in M3 is the button.** A press during playback stops
    playback on the edge at once, without waiting for the brain, and sends
    `interrupt`. The brain then drops the unsent sentences (Mike R8.3.6; the
    watch there works the same way with a touch). The spoken "stop" / "quiet"
    command does the same whenever it can be heard.
  - **Echo:** while it plays, the edge doesn't feed the segmenter, and it keeps
    ignoring input for a 200 ms tail afterwards. So in M3 the Pi can't hear
    itself, and it can't hear you over itself either. Voice barge-in *over*
    playback (`speaking:on` stopping a reply) needs echo cancellation. That is
    in M6 polish, and the protocol already carries everything it needs.
- `ctx.say` from skills already goes through the injected `tts`, so skills get
  remote speech for free.

### 8. Whisper and voder as separate services: port Mike's `services/whisper/serve.py` and PRD 8.3

Mike's reasoning applies in Python too. A GPU model that takes seconds to load
should stay warm across brain restarts. A transcription or synthesis that wedges
should be survivable by killing one process. And the brain must still start
(text mode, typed input) when a service is down.

**`services/whisper/serve.py`**: a port of Mike's service with the same
contract.
- `POST /transcribe` takes raw 16 kHz s16le mono PCM and returns
  `{text, language, languageProbability, confidence, ms, dropped}`.
- `GET /healthz` returns `{ok, model, device, warm, transcriptions}`.
- It binds **127.0.0.1 only**, with no auth: a port only this machine can open
  needs no password.
- Flags: `--port 0` (the OS picks a port and it is printed; used by tests),
  `--warm-only` (pre-fetch weights), `--model`, `--device`.
- It includes Mike's `_preload_cuda_libraries()` fix for pip-installed cuBLAS
  and cuDNN.
- It ships its own `jarvis-whisper.service` systemd unit and venv notes.
- It uses the same model config as today (`[stt]`), with `device = "cuda"` on
  the server.

The brain side, `jarvis/audio/whisper_client.py`:
- `WhisperClient.transcribe(pcm)` and `health()`.
- `TranscriptionUnavailable` for "the service is down", with a readable reason
  ("the transcription service is not running").
- `NullTranscriber` for `--whisper off`, so there is exactly one code path.

The edge is told `error: "Cannot hear you: …"`, and the voder speaks it if
it's up.

**`services/voder/serve.py`**: Mike's PRD 8.3 contract, with Piper.
- `POST /speak {text, voice, sample_rate}` returns s16le mono PCM, with
  `X-Voder-Ms` for the synthesis time.
- `GET /voices` and `GET /healthz`.
- It uses the persona's Piper voice, runs on the CPU (keeping it off Whisper's
  GPU), and ships a `jarvis-voder.service` unit.

Brain side:
- `jarvis/core/speech.py` makes text speakable (Mike R8.3.3):
  - strip markdown;
  - a code block becomes "code on the screen" once;
  - a path is shortened to its last part and a URL to its host;
  - a cap of 600 characters, cut at a sentence end, then "the rest is in the
    log".
- The answer is **split into sentences**. The first is synthesized and sent at
  once, and the rest follow in order.
- `speech` goes only to a connection that sent `control setSpeech{on:true}`.
  The Pi edge sends it by default. It uses that connection's `send()`, never a
  broadcast, and speech is never replayed.
- If the voder is down, `say()` degrades to sending `text` only and logs one
  line.

### 9. Security: the link is assumed to be on the public internet

- **TLS is required.** Either `[server] tls_cert` / `tls_key` are set, or the
  brain listens on localhost behind a TLS-terminating reverse proxy (Caddy is
  documented). A non-loopback bind without TLS is refused unless
  `allow_insecure = true`, which is LAN/dev only and logs a loud warning. The
  edge verifies the certificate; `[edge] tls_ca` supports self-signed setups.
- **Per-device pre-shared token.** The brain reads `JARVIS_EDGE_TOKENS`
  (`device_id:token,…`) from `.env`, and the edge reads `JARVIS_EDGE_TOKEN`.
  Tokens are compared with `hmac.compare_digest`. A missing or late `hello`
  closes with 4002, and a bad token with 4001, plus a per-IP backoff.
- Size walls: decision 4, plus a WebSocket `max_size` of 4 MB. `clientLog`
  lines are capped at 500 characters and rate-limited.
- **Tokens and audio are never logged or stored.** The `heard` text is logged.
- **Privacy, stated plainly:** with no wake word, *every segment of speech the
  Pi hears is transcribed on the self-hosted server*. ByName decides what
  reaches a skill, not what is transcribed. The ways to stop that are
  **PushToTalk** (the mic is actually off) and "turn off the mic". Ignore still
  listens and discards. This goes in the README next to the setup
  instructions, as Mike's R5a.2 does for a shared room.

### 10. Resilience

- A ping keepalive every 20 s. A peer that stops answering is terminated (the
  normal mobile/Wi-Fi failure).
- **Edge:** reconnects with exponential backoff up to `[edge] reconnect_max_s`.
  While the brain is unreachable, a PTT press or a detected segment gets a
  short local "offline" earcon. Segments recorded while offline are dropped,
  not queued: a command replayed minutes late is worse than a lost one.
- **Brain:** a pending `record_utterance` / `say` raises `Cancelled` on
  disconnect. The session ends cleanly and JARVIS returns to idle.
- **Differs from Mike:** Mike keeps a durable session with replay on reconnect,
  and turns keep running without a client. JARVIS doesn't adopt that in M3.
  Its turns are short, spoken commands, and "the session ends on disconnect" is
  simpler and honest. One edge per brain for now. The same `device_id`
  replaces the old socket; a different one is refused.

### 11. Config

```toml
[server]                    # `python -m jarvis serve`
host = "0.0.0.0"
port = 8765
tls_cert = ""
tls_key  = ""
allow_insecure = false

[whisper]                   # the brain's client for jarvis-whisper
url = "http://127.0.0.1:3461"   # "off" -> NullTranscriber
timeout_s = 20

[voder]
url = "http://127.0.0.1:3462"   # "off" -> text only
sample_rate = 16000

[addressing]
default_mode = "byname"     # byname | always | pushtotalk | ignore
names = ["jarvis", "hey jarvis"]
hold_ms = 2000

[edge]                      # `python -m jarvis edge`
server_url = "wss://jarvis.example.net:8765"
device_id = "livingroom"
tls_ca = ""
reconnect_max_s = 30
ptt_gpio = 0                # 0 = no button
# [edge.segment] overrides Mike's segmenter defaults (decision 2)
```

These are parsed through the existing `_section()` in `jarvis/config.py` into
frozen dataclasses. Secrets live only in `.env`.

### 12. Dependencies

- Both sides: `websockets`. The brain: `httpx` for the service clients, which
  is already installed transitively.
- Services: faster-whisper (plus `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` on
  the GPU box), and piper-tts. Each in its own venv, as in Mike.
- Pi: `pip install numpy sounddevice websockets`, plus
  `sudo apt install libportaudio2`, plus `gpiozero` if there is a PTT button.
  **No ONNX, no models.**
- Add `websockets` and the two service health checks
  (`GET /healthz`: ✓ warm / ✗ not running) to `check_setup.py`.

### 13. Tests never touch a real network, mic or GPU

These follow the seams Mike made testable:

- **Segmenter** on synthetic PCM (tones plus noise plus gaps) and a recorded
  fixture WAV:
  - the same input gives the same segments;
  - pauses of 400 ms stay one segment and 1 s splits it;
  - a single zero frame doesn't collapse the floor;
  - room tone alone closes within a few seconds;
  - a 20 s monologue is cut at a pause, not at 15 000 ms exactly;
  - a 100 ms click is no segment.
- **`protocol`** and **`decode_segment`**:
  - every message round-trips;
  - junk yields `(False, error)` and never raises;
  - oversized base64 is refused before decoding;
  - odd byte counts and non-base64 are refused;
  - under 120 ms is ignored.
- **Addressing:**
  - every mode × {named, un-named, mode command};
  - "pause" then "continue" restores PushToTalk;
  - mode commands work in Ignore;
  - STT punctuation and casing variants are handled.
- **Hold window** with a fake clock:
  - two fragments within 2 s become one utterance;
  - `speaking:on` pauses the countdown;
  - the 20 s cap releases what is held;
  - a disconnect clears the flag.
- **`RemoteLink` round trip** over a loopback `websockets` server on
  `127.0.0.1:0`, with a **fake whisper** (canned text per segment) and a
  **fake voder** (a short PCM), driven with `asyncio.run` like
  `tests/test_orchestrator.py`:
  - "Jarvis, search black holes" gives `heard`, then the search skill, then
    `speech` parts; `playbackDone` returns JARVIS to idle;
  - un-addressed speech gives `heard` and nothing else;
  - `_ask` accepts a reply without the name;
  - silence ends the follow-up window;
  - `interrupt` gives `Cancelled`;
  - `interrupt` during speech drops the unsent parts;
  - whisper down gives the "Cannot hear you" `error`, and the server keeps
    running;
  - disconnect ends the session cleanly.
- **Auth:** a bad token gives 4001, no hello within the timeout gives 4002,
  and a plain `ws://` on a non-loopback host is refused without
  `allow_insecure`.
- **Whisper service:** start `serve.py --port 0` with a tiny model, or a
  stubbed model when faster-whisper is absent (`pytest.skip`, as
  `conftest.embedder` does). Check `/healthz`, and that `/transcribe` refuses
  odd-length bodies.
- **Edge imports:** import the edge entry point in a subprocess and assert
  that none of `fastembed`, `faster_whisper`, `piper`, `sklearn`,
  `anthropic` or `openwakeword` is in `sys.modules`.

---

## New files

- `jarvis/audio/segment.py`: the pure segmenter (decision 2).
- `jarvis/audio/player.py`: PCM playback with `play(pcm, sr)` / `stop()`, split
  out of `Speaker.say` (which keeps using it). The edge uses it to play
  `speech` parts in order, with a 150 ms gap between sentences.
- `jarvis/audio/whisper_client.py`: `WhisperClient`, `NullTranscriber` and
  `TranscriptionUnavailable`.
- `jarvis/core/speech.py`: the speakable-text pass and sentence splitting.
- `jarvis/remote/__init__.py`
- `jarvis/remote/protocol.py`: the schema, `validate_c2s` and message
  constructors (decision 3).
- `jarvis/remote/intake.py`: `decode_segment` and the audio intake (decision 4).
- `jarvis/remote/addressing.py`: the modes, mode commands and hold window
  (decisions 5–6).
- `jarvis/remote/server.py`: the WS server, auth/TLS and `RemoteLink`
  (decision 7).
- `jarvis/remote/edge.py`: the edge loop. Mic → segmenter → `speaking` /
  `audio`; play `speech`; PTT; reconnect and earcon.
- `services/whisper/serve.py` and `services/whisper/jarvis-whisper.service`.
- `services/voder/serve.py` and `services/voder/jarvis-voder.service`.
- `tests/test_segment.py`, `tests/test_protocol.py`, `tests/test_intake.py`,
  `tests/test_addressing.py`, `tests/test_remote_link.py`,
  `tests/test_remote_auth.py`, `tests/test_whisper_service.py`,
  `tests/test_edge_imports.py`, and a fixture WAV under `tests/fixtures/voice/`.

## Modified files

- `jarvis/__main__.py`: `serve` and `edge` subcommands. `app` is imported
  lazily, only in the subcommands that need it, so the edge's import graph
  stays light.
- `jarvis/app.py`: `build_server_orchestrator(config, link)`, which mirrors
  `build_text_orchestrator` (`RemoteLink` in place of
  `TextWake` / `TextIO` / `TextTTS`). It doesn't load `Transcriber` or
  `Speaker` in-process.
- `jarvis/core/interrupt.py`: a public `trigger_cancel()`.
- `jarvis/audio/tts.py`: `Speaker.say` delegates playback to `audio/player.py`.
  The voder service reuses `Speaker.synthesize`.
- `jarvis/config.py` and `config.toml`: the sections in decision 11.
- `check_setup.py`: a `websockets` row, whisper and voder `/healthz` rows, and
  a masked `JARVIS_EDGE_TOKEN(S)` row.
- `CLAUDE.md`: the two new run modes, the services, the Pi install line, and
  the privacy note.
- `PRD/jarvis-2026-rebuild.md`: mark Milestone 3 ✅ DONE with an outcome
  summary, plus `PRD/milestone-3-remote-edge-outcome.md`.

---

## Verification

`python -m pytest` is green, including every test in decision 13.

**Regression:** `python -m jarvis text --script …` output is unchanged,
`python -m jarvis --selftest` exits 0, and all-in-one `python -m jarvis` still
wakes on "hey jarvis". Also dry-run `python -m jarvis text` per CLAUDE.md.

**Acceptance criteria** (in the style of Mike 5a):

1. With the brain, both services and the Pi edge running, saying
   *"Jarvis, search black holes"* at the Pi produces a spoken answer from the
   Pi's speaker, with nothing touched.
2. The `heard` text is logged, and sent to the edge, **before** the answer.
3. Talking in the room without "Jarvis" produces `heard` lines and **no**
   action.
4. *"Jarvis, pause input"* stops everything reaching JARVIS. *"Jarvis, continue
   input"* restores the previous mode. Both work spoken, with nothing touched,
   and "continue" works while paused.
5. In PushToTalk, the Pi's mic is demonstrably off (no `speaking` or `audio`
   messages) unless the button is held.
6. A segment of pure silence or a cough produces no turn and no reply.
7. A sentence with ordinary pauses is **one** turn (the hold window), and
   `state` shows *still listening* while words are held.
8. A recorded WAV fed through the segmenter gives the same segments on the Pi
   and on the dev box.
9. With `jarvis-whisper` stopped, the brain still starts. A spoken command
   gets *"Cannot hear you: the transcription service is not running"*, and
   text mode still works.
10. Pressing the button during a spoken answer stops playback at once, and the
    next utterance is handled.
11. Pulling the network mid-reply: the brain logs a clean session end, the
    edge plays the offline earcon, then reconnects by itself.

**Latency budget**, logged per stage and reported in the outcome doc (Mike
R5a.7 / R8.3.4):

| stage | target |
|---|---|
| segment flush (end of speech → `audio` sent) | < 200 ms |
| transcription (Whisper on GPU) | < 800 ms |
| answer text → first `speech` part sent | < 500 ms |
| end of speech → first sound from the Pi | < 2 s plus the skill's own time |
