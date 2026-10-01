# Known issues

Things that are wrong and not yet fixed, with enough written down that the fix
doesn't start from zero. Fixed entries move out of here into the milestone
outcome doc that fixed them.

---

## 2. The edge blames the network for a microphone it cannot open

**Reported:** 2026-09-27, setting up brain + edge on one desktop (Fedora 44,
PipeWire 1.6.9).
**Status:** open, not fixed.
**Severity:** misleading, not broken — but it sends you debugging the wrong half.
**Owner:** unassigned.

### What was seen

```
edge 'desktop' connected.
  …connected after 1 attempt(s).
INFO jarvis.remote.edge: not connected (PortAudioError: Error opening RawInputStream: Unanticipated host error [PaErrorCode -9999]: 'No such file or directory' [ALSA error -2])
! cannot reach ws://127.0.0.1:8765 — PortAudioError: …
```

The socket was fine (the brain logged `edge desktop connected from 127.0.0.1`);
the **microphone** failed to open, which drops the connection, and
`jarvis/remote/edge.py:345` reports every such failure as "cannot reach
<server_url>". It then retries forever with the same wrong message.

### When the mic fails to open (this machine)

Through PortAudio's `pipewire` ALSA device, capture opens **only** when
`PIPEWIRE_NODE` names a source that currently exists:

| `input_pipewire_node` | result |
|---|---|
| `""` (PipeWire default source) | ENOENT |
| an existing source (`alsa_input.pci-…analog-stereo`) | opens |
| a source that is absent — a USB headset switched off | ENOENT |
| a `.monitor` | ENOENT |

So the empty default does not work here, and a pinned USB headset makes the
edge unusable while it is off.

### Fix, when someone takes it

Tell the two apart where the exception is caught: a `PortAudioError` from
opening the stream should say *"cannot open the microphone <node or device>"*
(naming `[edge] input_pipewire_node`), not "cannot reach". Worth considering
too: `check_setup.py` could try opening the configured input and output.

---

## 3. The brain serves one question at a time: nothing is taken while it speaks

**Reported:** 2026-09-27, by the owner, from live use of `serve` + `edge`.
**Status:** open — a design limit of M3, not a regression.
**Severity:** a second question is lost or waits behind the first; a second
device cannot be used at all.
**Owner:** unassigned.

### What happens now

The brain is one pipeline, strictly one turn at a time, end to end:

- **One device.** `RemoteServer.handle` (`jarvis/remote/server.py`, "One edge
  per brain for now") closes a second `device_id` with *"another device is
  connected"*. There is one `RemoteLink`, one orchestrator and one utterance
  queue for the whole brain.
- **Speaking blocks thinking.** `RemoteLink.say()` synthesises sentence by
  sentence and then **waits for the edge's `playbackDone`** (up to the audio's
  length + 10 s, capped at 120 s) before returning. The orchestrator thread is
  inside `say()` for that whole time, so no next utterance is routed, no skill
  runs, nothing is thought about.
- **The edge does not listen while it plays.** `jarvis/remote/edge.py` stops
  feeding its segmenter during playback plus a 200 ms tail (echo protection),
  so a question asked over the answer is never even transcribed.

### What it should do

Split **thinking** from **speaking**:

1. The orchestrator works out the answer. When it has text to say, it hands it
   to a **speech worker for the device that asked** (synthesis → `speech`
   parts → wait for `playbackDone`) and goes straight back to taking requests.
2. **A. Two devices, two questions at once.** Each device gets its own
   session — queue, turn, and speech worker — so the living room and the
   kitchen are answered in parallel, each on its own speaker.
3. **B. A second question from the same device is queued, not dropped.** It is
   heard and worked on while the first answer plays; its answer is queued
   behind the first on that device's speech worker and plays when the first
   finishes. Per device, answers play in the order the questions were asked.

### What has to be decided before building it

- **Orchestrator state is single-session.** Standby/awake, the follow-up
  window, and multi-turn dialogs (`teach`'s `_ask`, "Shall I keep it?") live in
  one orchestrator. Per-device sessions mean either an orchestrator per device
  or that state keyed by device — and a dialog's reply must reach the dialog
  that asked, not the other queued question.
- **Half-duplex on the edge.** B needs the edge to listen while playing. Safe
  with a headset (this desktop's Audeze); on a speaker it hears itself until
  there is AEC (M6). Likely a per-device `[edge]` setting.
- **Interrupt and cancel become per device**, and "stop" has to say whether it
  stops the current answer only or the whole queue.
- **Shared services.** Two devices mean concurrent requests to `jarvis-whisper`
  and `jarvis-voder` (both `ThreadingHTTPServer`) — check the models are safe
  to call concurrently, or serialise per service.
- **Threads.** A speech worker per device must be bounded and joined when the
  device disconnects; per CLAUDE.md step 5 the thread count has to return to
  baseline after each turn and each disconnect.
- **Protocol.** `speech` already carries an id; queued answers need the edge to
  play parts of successive ids in order without the brain waiting in between.

### Where to start

`RemoteLink.say()` (the blocking wait), `RemoteServer.handle` (the one-edge
rule) and `RemoteLink` itself (one per brain → one per device), plus the
edge's mute-while-playing in `jarvis/remote/edge.py`. Probably a milestone of
its own rather than a fix.

---

## 4. `tests/test_remote_auth.py` reads the machine's real `config.toml`

**Reported:** 2026-10-01, while running the suite for the factory fix.
**Status:** open, not fixed.
**Severity:** four tests fail on any machine whose `config.toml` sets
`[server] allow_insecure = true`; they pass in a clean checkout.
**Owner:** unassigned.

`test_a_routable_address_without_tls_is_refused`,
`test_the_proxy_header_is_ignored_unless_it_is_configured`,
`test_loopback_stays_trusted_with_no_peers_configured` and
`test_trusted_peers_without_a_header_name_do_nothing` build their config from
the local `config.toml`, so the owner's real `[server]` settings leak in and
the "refuses without TLS" path never triggers. Fix: build the `[server]`
section explicitly in the test (or load from a `tmp_path` config) instead of
inheriting the developer's file.
