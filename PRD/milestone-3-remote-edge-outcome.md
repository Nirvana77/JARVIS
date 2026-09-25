# Milestone 3 — remote edge (brain server + audio satellite): outcome

**Date:** 2026-09-25
**Branch:** `milestone-3-remote-edge` (off `develop`, not yet committed or merged)
**Result:** ✅ JARVIS can be split in two. `python -m jarvis serve` is the
brain; `python -m jarvis edge` is an audio satellite with a microphone, a
speaker and no models at all. The voice pipeline is Mike's, ported: a pure
segmenter, one message per segment, Whisper and Piper as warm loopback
services, addressing modes instead of a wake word, and a hold window that
merges a split sentence. The orchestrator was **not** rewritten.

Plan and design decisions: `PRD/milestone-3-remote-edge.md`.

---

## What shipped

| Area | Module(s) | Notes |
|---|---|---|
| Segmenter | `jarvis/audio/segment.py` (new) | Mike's `client/src/audio/segment.ts`, ported with its tuning comments. Energy VAD in dBFS, minimum-statistics noise floor (window minimum, fast down / very slow up, faster while open), 300 ms pre-roll, 250 ms tail, 700 ms hangover, 250 ms minimum speech, 15 s cap cut at the most recent 200 ms pause. Pure: no device, no clock. |
| Wire protocol | `jarvis/remote/protocol.py` (new) | One schema for both sides. JSON text frames only, PCM as base64, 16 kHz s16le mono fixed. `validate_c2s` never raises. Close codes 4001–4004. |
| Intake | `jarvis/remote/intake.py` (new) | `decode_segment` checks every bound *before* allocating: base64 length, strict base64, decoded size, even byte count, 120 ms floor. An empty transcription produces nothing at all. `heard` goes out *before* routing. One diagnostic line per segment. |
| Addressing + hold | `jarvis/remote/addressing.py` (new) | Four modes, mode commands matched **before** the gate in every mode, the microphone switch on the same footing, `apply_mode_command` (pause remembers where it came from), `HoldWindow` (silence-based countdown, 20 s cap from the last real fragment), `DeviceModes` (per-device JSON under `data/remote/`). |
| Brain server + link | `jarvis/remote/server.py` (new) | `RemoteLink` plays `wake`/`mic`/`stt`/`tts` over a `queue.Queue` and `run_coroutine_threadsafe`; `RemoteServer` does auth, TLS policy, the gate, the hold timer and the intake. One edge per brain; the same `device_id` replaces its socket, a different one is refused. |
| Edge | `jarvis/remote/edge.py` (new) | Mic → segmenter → `speaking` + `audio`; plays `speech` parts in order with a 150 ms gap and reports `playbackDone`; push-to-talk (GPIO) with the mic genuinely closed otherwise; reconnect with backoff and an offline earcon; no feeding the segmenter while playing, plus a 200 ms tail. |
| Services | `services/whisper/serve.py`, `services/voder/serve.py` (+ systemd units, new) | Ports of Mike's whisper service (including `_preload_cuda_libraries`, the VAD filter and the no-speech floor) and PRD 8.3's voder contract with Piper. Neither imports `jarvis`: they run in their own venvs. |
| Service clients | `jarvis/audio/whisper_client.py`, `jarvis/audio/voder_client.py` (new) | `TranscriptionUnavailable` / `SynthesisUnavailable` carry a sentence a person can act on. `"off"` gives a Null implementation, so there is exactly one code path. |
| Speakable text | `jarvis/core/speech.py` (new) | Markdown stripped, a code block becomes "code on the screen" once, a path becomes its last part, a URL its host, 600-character cap cut at a sentence, then sentence splitting for part-by-part synthesis. |
| Playback | `jarvis/audio/player.py` (new); `jarvis/audio/tts.py` | Playback moved out of `Speaker.say` into a `Player` the edge reuses. `Player.stop()` cuts an answer short from another thread. |
| Interrupt | `jarvis/core/interrupt.py`, `jarvis/core/orchestrator.py` | Public `Interrupter.trigger_cancel()`, and `Orchestrator.interrupter` so the link can reach it. The edge's `interrupt` now takes exactly the path Enter takes. |
| Assembly / CLI | `jarvis/app.py`, `jarvis/__main__.py` | `build_server_orchestrator`, `run_server_mode`, `run_edge_mode`; `serve` and `edge` subcommands. **`jarvis.app` is now imported lazily**, so `python -m jarvis edge` never touches the brain's import graph. |
| Config / setup | `jarvis/config.py`, `config.toml`, `check_setup.py` | `[server] [whisper] [voder] [addressing] [edge]` + `[edge.segment]`; `JARVIS_EDGE_TOKEN(S)` from `.env` only; a websockets row, two `/healthz` rows and masked token rows in the setup report. |
| Tests | 9 new files + `tests/remote_harness.py` | **390 passed, 1 skipped** (was 159 passed, 1 skipped after M2.5). |

## Test changes (said explicitly, per CLAUDE.md)

No existing test was loosened, and every M2.5 assertion still runs. One
existing test was **re-pointed, not weakened**:
`tests/test_audio.py::test_speaker_targets_a_pipewire_node_only_while_opening`
drove `Speaker._output`, a private helper this milestone deliberately moved
into `jarvis/audio/player.py` (the edge plays PCM with no Piper in the
process). It now calls `Speaker(...).player._output(...)` — same four
assertions, one indirection further down. Two new tests cover the `Player`
itself: writing in slices so `stop()` takes effect mid-answer, and degrading
to a warning on a box with no output device.

One test written during this milestone was **corrected before it ever passed**:
`test_the_cap_runs_from_the_last_words_that_actually_arrived` originally
asserted that a fragment held for 15 s, extended, then held 15 s more would be
released — which contradicts the plan's own rule that the 20 s cap runs *from
the last words that actually arrived*. The plan is right and the test was
wrong, so the test now checks what decision 6 actually specifies: still held at
+10 s, released past +20 s.

## What the tests caught (and the code, not the test, was fixed)

1. **The conversation window closed too early.** `RemoteLink.conversation` was
   true only while `record_utterance` was blocked. A user answering *while the
   reply was still playing* — or during an `_ask` prompt — arrived a fraction
   of a second before JARVIS started listening and was dropped for not saying
   the name, which is exactly the moment saying the name feels most absurd. It
   is now open for the whole wake session.
2. **`speaking(False)` did not restart the countdown.** The hold window
   measured from the last fragment instead of re-arming when the user stopped
   talking, so a pause that ended just before the window lapsed released
   immediately. `_arm()` now matches Mike's `arm()`.

### 3. Found by the first live run, not by the tests

**The listening window charged the user for the pipeline's own latency.**

To a microphone, `record_utterance(grace=…)` means *"how long to wait for them
to **start** talking"*: once speech starts, `record_utterance` returns only when
the utterance is over, so a long sentence costs the window nothing. `RemoteLink`
made `grace` mean *"how long until a finished, transcribed, merged utterance is
in my hand"* — so the user's speaking time, the 700 ms hangover, Whisper's
second and the 2 s hold window were all charged to it.

With the default 10 s window that needed a reply of about six and a half
seconds to break, which is an ordinary answer to an ordinary question. In the
live run it broke an `_ask` inside a `teach` dialog:

```
heard 6520ms ... "Calculate anything. Plus minus division, etc."
  heard   : ""              <- _ask had already given up
Jarvis: Never mind, then.   <- the dialog aborted
utterance -> 'Calculate anything. Plus minus division, etc.'   <- delivered next
```

The words were logged on the same screen as the giving-up, one second apart,
and then arrived in the *next* turn with no context. The same thing put a
spurious "Standing by, sir." in front of an ordinary follow-up.

The edge already reported the one fact that fixes it — `speaking{on}`, sent the
moment its detector opens — and the link was not listening. `RemoteLink.incoming`
is now true while **any** of three things holds: the edge's detector is open
(they are talking now), a segment is inside Whisper (they have stopped, the
words are in flight), or the hold window is holding a fragment (they paused
mid-thought). While it is true, the capture's deadline keeps moving, bounded by
`inflight_max_s` (60 s) so an edge whose detector jams open cannot hold a turn
for ever.

Verified both ways: the three new tests fail against the old behaviour with
exactly the live symptom (`Jarvis: <standby>`), and a live re-run — an **8.0 s**
follow-up spoken two seconds after the previous answer, ~13 s of wall clock
against a 10 s window — stayed in one session with no standby line.

## Verification

### Automated

```
$ python -m pytest -q
390 passed, 1 skipped, 1 warning in 94.50s
```

New: `tests/test_segment.py` (20), `tests/test_protocol.py` (21),
`tests/test_intake.py` (16), `tests/test_addressing.py` (72),
`tests/test_speech.py` (19), `tests/test_remote_config.py` (8),
`tests/test_whisper_service.py` (21, real HTTP on port 0 with stubbed models),
`tests/test_remote_link.py` (23, loopback WebSocket + real orchestrator),
`tests/test_remote_auth.py` (24), `tests/test_edge_imports.py` (5, subprocess).

No test touches a real network interface, microphone, GPU or speaker.

### Threads (CLAUDE.md step 5)

- **Test run**: `tests/conftest.py`'s autouse `no_leaked_threads` guard was
  already in place and stayed green, including the tests that start a real
  `ThreadingHTTPServer` and a real WebSocket server.
- **Text-mode dry run**: 1 → 28 threads (fastembed/ONNX pools), back to 24 at
  exit, no growth across turns.
- **Brain dry run** (`serve`, after 8 turns and two edge connections):
  **57 threads, stable across repeated samples**, no growth per turn.

### Dry run — the real thing, end to end

Both services and the brain were started for real on this machine
(`services/whisper/serve.py --device cpu --model small.en`,
`services/voder/serve.py`, `python -m jarvis -v serve`), and driven by a
synthetic edge that used **Piper to speak the utterances**, cut them out of a
room recording with the **real segmenter**, and sent them over the real
protocol. Nothing was faked but the microphone and the speaker.

| Acceptance criterion | Result |
|---|---|
| 1. An addressed command produces a spoken answer | ✅ "Jarvis, search black holes." → the Wikipedia summary in 2 `speech` parts |
| 2. `heard` is logged and sent *before* the answer | ✅ `heard` at ~1.0 s, first `speech` at ~3.9 s |
| 3. Talking without the name produces `heard` and no action | ✅ from idle: two ambient sentences, `heard` only, no answer, no speech, no state change |
| 4. "Pause input" / "continue input", spoken, both ways | ✅ `modeChanged` byname→ignore, nothing admitted while paused, `continue input` (unaddressed, while paused) → ignore→byname |
| 6. Silence or a cough produces no turn | ✅ unit-tested (< 120 ms ignored, < 250 ms speech dropped); no empty turns in the dry run |
| 7. A sentence with pauses is one turn | ✅ hold window, unit-tested on a fake clock and end-to-end in `test_remote_link.py` |
| 8. The same recording gives the same segments | ✅ `tests/test_segment.py`, including chunk sizes that straddle frames |
| 9. With whisper stopped the brain still starts and says so | ✅ unit-tested; `serve` also prints "not answering — starting anyway" |

**Not verified on hardware** (no Pi, no GPU box, no second network here):
criterion 5 (the PTT button's GPIO), 10 (a button press stopping real
playback), 11 (pulling the network mid-reply), and the TLS path against a real
certificate over the internet. The logic behind each is unit-tested; the
hardware behaviour is a first-install check.

### Latency: where the turn's time went, and what was done about it

The first live run felt slow, so the turn was measured rather than guessed.
End of speech → first sound out of the speaker, short command, all on one
laptop:

| stage | before | after | note |
|---|---|---|---|
| segmenter hangover | 700 ms | 700 ms | fixed — the segment cannot close sooner |
| transcription | 1050 ms | **415 ms** | `small.en` → `base.en`, int8 CPU; identical transcripts on the test clips |
| hold window | 2000 ms | **1000 ms** | `[addressing] hold_ms` (below) |
| NLU + skill | ~110 ms | ~110 ms | `search` adds 1–2 s of its own Wikipedia fetch |
| voder, first sentence | ~75 ms | ~55 ms | only the first sentence is on the critical path |
| **total** | **~3.9 s** | **~2.3 s** | measured end to end, twice |

**`hold_ms` halved, 2000 → 1000, deliberately against the plan.** The plan takes
Mike's 2000. It is paid on *every* turn, and the fix above changes what it has
to do: because the edge's `speaking{on}` pauses the hold's countdown, the window
only has to be long enough to **notice a continuation starting**, not to swallow
one whole — transcription and all. What it actually buys is a tolerated
mid-command pause of `hangover + transcription + hold`, so even at 1000 ms that
is ~2.1 s here and ~2.0 s on a GPU brain. `test_the_hold_only_has_to_notice_a_
continuation_not_swallow_it` pins the property: with a 100 ms hold and a
continuation whose transcript lands 400 ms late, the two fragments still merge.
One line in `config.toml` puts it back.

**Model choice, measured on this laptop** (CPU, int8, same three clips):

| model | transcribe | transcripts |
|---|---|---|
| `small.en` | 1014–1123 ms | correct |
| `base.en` | 404–421 ms | identical to `small.en` |
| `tiny.en` | 259–272 ms | dropped a comma in "Jarvis, search black holes" |

Cost is nearly flat in utterance length (a 1.9 s and a 3.3 s clip differ by
~15 ms), so it is fixed overhead, not decoding — which is why the bigger model
is pure loss on a CPU brain. `base.en` is the sensible default for a CPU brain;
`small.en` or larger belongs on the GPU box the milestone is for.

A per-turn timing line now goes to the log at `-v`, so the next person to say
"it feels slow" has the number:

```
turn: utterance -> answer 114ms, -> first speech 167ms (1 sentence(s))
```

### Latency, as first measured (CPU, before the tuning above)

| Stage | Budget | Measured here |
|---|---|---|
| transcription | < 800 ms (GPU) | ~1000 ms (`small.en`, int8, **CPU**), of which ~990 ms is the model |
| end of speech → first `speech` part | < 2 s + the skill | ~3.2 s short answer, ~3.9 s with the search skill's own network fetch |
| segment flush | < 200 ms | not measurable with a synthetic edge (no device latency) |

The transcription figure is the one to re-measure on the GPU box the milestone
is actually for; Mike's number on a 5060 Ti was 300–380 ms warm.

## Added after the plan: running behind a tunnel

Decision 9 assumed either the brain's own certificate or "a TLS-terminating
reverse proxy". A Cloudflare Tunnel is the second of those and needs no open
port, but it breaks one assumption the plan did not notice: **through a tunnel,
every connection reaches the brain from 127.0.0.1**. The per-address auth
backoff keyed on that cannot tell the Pi in the hall from somebody hammering the
public hostname — so a stranger's failed guesses would lock out the real edge,
which is a self-inflicted denial of service created by the act of publishing.

`[server] trusted_proxy_header` (default `""`, off) names the header that
carries the real client: `CF-Connecting-IP` for Cloudflare, `X-Forwarded-For`
for most proxies. It is believed **only when the connection itself came from
loopback** — a proxy on this machine is the only thing that can set a header we
did not put there ourselves; from any other address it is the client's own
claim, and believing it would hand every guesser a way to wipe their backoff by
inventing an address. `RemoteServer.client_key()` is that rule, and
`tests/test_remote_auth.py` pins it in both directions.

The backoff itself was fixed at the same time: it now doubles per address
(2 s → 30 s) with its own failure counter instead of scaling with the *size of
the table*, a success clears that address, and the table is pruned — on the
open internet its key space is every address there is.

`[server] trusted_proxy_peers` (a list of addresses/CIDRs, default empty)
extends that trust to a proxy that is not on loopback — `cloudflared` in a
container reaches the brain over the Docker bridge, and a proxy on another host
reaches it over the LAN. Both are ordinary deployments, and loopback-only trust
would have failed them *silently*, which is the worst way for this particular
thing to fail. A mistyped entry stops the brain at startup instead.

The setup for all three shapes (native `cloudflared`, `cloudflared` in Docker,
and tunnel replicas across machines) is in CLAUDE.md, including the point that
a second tunnel replica cannot make JARVIS more available — the brain runs on
exactly one machine.

## Known behaviour worth stating plainly

- **The conversation window is wide.** In ByName, once JARVIS has answered,
  *anything* said in the room for the next `[capture] follow_up_s` (10 s by
  default) reaches it without the name — and each admitted utterance starts a
  fresh window. This is the all-in-one path's own follow-up semantics, not
  something new, but on an always-listening edge it means ambient conversation
  right after a command can be acted on. Shorten `follow_up_s`, or use
  PushToTalk, to narrow it. Worth revisiting in M6.
- **Privacy.** With no wake word, *every* segment of speech the edge hears is
  transcribed on the self-hosted brain. ByName decides what reaches a skill,
  not what is transcribed. The ways to stop that are PushToTalk (the mic is
  genuinely off) and "turn off the mic"; Ignore still listens and discards.
- **One edge per brain.** `device_id` is in the protocol and the mode is stored
  per device, so multi-room is a small step, but a second device is refused
  today rather than silently sharing one conversation.
- **Echo.** The edge cannot hear itself — and cannot hear you over itself.
  Voice barge-in during playback needs AEC (M6); the button is the way to stop
  an answer now.
