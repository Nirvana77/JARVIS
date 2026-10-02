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

---

## 6. `tests/test_skills.py` reads the machine's real edge tools

**Reported:** 2026-10-01, while running the suite for M4.
**Status:** open, not fixed.
**Severity:** `test_registry_discovers_the_builtins` fails on any machine an
edge has declared tools to; it passes in a clean checkout. Same family as #4.
**Owner:** unassigned.

The test passes `packages=(BUILTIN_PACKAGE,)` precisely so that locally
learned skills don't leak in — but `Registry.discover` also registers every
tool stored under `config.remote_dir` (`data/remote/tools`), whatever
`packages` says. On this machine that adds 12 names (`set_timer`,
`find_watch`, `notify`, …). Fix: point the test's config at an empty
`tmp_path` data dir, or give `discover` a way to skip the edge store.

---

## 7. A learned skill's params are never filled

**Reported:** 2026-10-01, from the M4 dry run.
**Status:** open, not fixed.
**Severity:** a learned skill that takes an argument always runs with its
default. "flip three coins" flips one coin.
**Owner:** unassigned.

`Orchestrator._params_for` extracts typed params only for `origin == "edge"`
(`_slots.extract_typed`). Every other skill goes through `_slots.extract`,
which knows four builtin labels (`search`, `play`, `open_app`, `note`) and
returns `{}` for anything else — so `flip_a_coin`'s declared
`count: integer` is never set. Likely fix: use `extract_typed` for any
manifest whose `params` declare types, not just edge tools; check what the
factory's generated manifests use for type names first (`"integer"` here,
where edge tools use their own vocabulary).


---

## 8. A slow firmware download can be cut off after 10 seconds

**Reported:** 2026-10-01, while looking into #9. Not the cause of #9.
**Status:** open, not fixed.
**Severity:** none behind a proxy that takes the image at once (cloudflared
does); a truncated image, and a failed update, for an edge that fetches
`/firmware` straight from the brain over a slow link.
**Owner:** unassigned.

`RemoteServer.process_request` returns the whole image as the `Response` of
the WebSocket handshake. `websockets` (17.1) treats any non-101 response as a
rejected connection: the handshake, including the write, runs under
`open_timeout` (10 s), and after the write it waits `close_timeout` (10 s)
for the peer to close and then **aborts the transport**, dropping whatever is
still in its buffer.

Reproduced on loopback with a throttled reader: a 24 MB image read at 1 MB/s
arrives as 12.0 MB and stops. A 1.7 MB image survives there only because the
kernel's socket buffers swallow all of it at once; over Wi-Fi to an ESP32
writing flash (~100 KB/s) they would not.

Fix, when someone takes it: either pass longer `open_timeout`/`close_timeout`
to `websockets.serve` (they also govern how long an unauthenticated
connection may sit in the handshake, so not for free on an internet-facing
port), or serve `Range` requests so each response is small (the watch's
`esp_https_ota` has `partial_http_download`).

---

## 9. The watch rejects a forced update within a second of fetching it

**Reported:** 2026-10-01, by the owner: "force update" of a locally built
`13ac6be-dirty` image failed five times (21:14–21:27 local).
**Status:** open — the cause is on the watch and not yet known.
**Severity:** a forced update of that image did not install.
**Owner:** unassigned.

What is established:

- **The brain did its part every time.** HAProxy (`jarvis-lb`) logged each
  fetch as a complete, cleanly closed 1,718,029-byte connection in 57–256 ms —
  the same shape as the two updates to `13ac6be` that worked earlier the same
  evening (20:41, 21:11). The `426 Upgrade Required` lines around it are
  HAProxy's health probe, not the watch.
- **Forcing is not the cause.** The 21:11 update that worked was itself
  forced: the watch had run `13ac6be` since 20:42 and was offered `13ac6be`
  again, which neither side allows without `force`. So `force: true`, a
  same-version offer and a spoken answer just before it were all present in
  a success.
- **The image was sound**: an ESP32-S3 app, IDF v5.5.5, project
  `jarvis_edge`, version `13ac6be-dirty`, appended SHA-256 correct, 1.7 MB
  against 4 MB slots; its static RAM layout matched the CI build's, and the
  same build booted over USB. (It is no longer staged — `jarvis-builder`
  replaced `watch.bin` with `342d829` at 22:25 local.)
- **The watch gave up about a second after the image began to arrive**
  (power log, 21:27:39 `update to 13ac6be-dirty` → 21:27:40 `hold over`; a
  working update takes ~23 s). That is before any real download, so not the
  size check, the checksum or `esp_https_ota_finish`.

Ruled out by reading `main/ota.c` against ESP-IDF 5.5.5 with those facts: the
request never leaving (it reached the brain), the 15 s timeouts, the app
description / project / version / chip checks (the bytes pass all of them),
the header buffer, chunked handling, and a partition conflict.

What is left, most likely first:

1. **Internal, DMA-capable RAM running out as the TLS stream starts.** Free
   internal heap when an update began: 16 KB and 14 KB on the two successes,
   13 KB on the one failure that has a row. `update()`'s buffers (≤ 4 KB
   each) go to internal RAM first, and decrypting TLS needs small
   internal-only buffers with no PSRAM fallback. Against it: the 21:11 update
   kept going with 10 KB free. Serial would show `esp-aes: Failed to allocate
   memory` or a transport read error before `E ota: … failed (ESP_FAIL)`.
2. **The Cloudflare-to-watch leg delivered something other than the body**
   (a reset, an edge error page). Nothing ties that to this image, and the
   server cannot see that leg.
3. **`ESP_ERR_OTA_ROLLBACK_INVALID_STATE`** (running image still "pending
   verify") for the first four — unlikely, the same boot → `ready` →
   `ota_confirm` sequence took effect after the first OTA — and it cannot
   explain the fifth.

**Why this cannot be told apart after the fact, and the fix for that** (watch
repo, `jarvis-edge/main/ota.c`, the failure branch of `ota_task`): the reason
only goes to the serial console. Record it —
`powerlog_event("update failed: %s", esp_err_to_name(err)); powerlog_flush();`
(the missing flush is also why the first four failures left no rows), send it
to the brain as a `clientLog`, and have `update()` say which step failed. Add
the largest free internal block to the `update to` event. If (1) is it:
`.buffer_caps = MALLOC_CAP_SPIRAM` in `esp_https_ota_config_t` moves the OTA
buffer to PSRAM.

**Shortest experiment:** with the watch on USB, `idf.py monitor` (no flash)
and say "force update". The `E ota:` line and the `esp_https_ota` /
`esp-aes` / `HTTP_CLIENT` lines before it name the failing call. If it
simply works (it would now install `342d829`), the cause was specific to
that image or that half hour.

---

## 10. The loop's thread pool adds an idle worker now and then

**Reported:** 2026-10-01, from the M4 / M4.5 thread monitoring.
**Status:** open, not fixed. Pre-dates both milestones.
**Severity:** cosmetic — a few idle threads, capped at `min(32, cpus + 4)` (20
here). But `ps -T` shows a count that creeps, which CLAUDE.md step 5 says to
treat as a bug, so it is written down.
**Owner:** unassigned.

Every blocking call the orchestrator makes goes through `asyncio.to_thread`,
i.e. the loop's default `ThreadPoolExecutor`. That executor starts a new
worker whenever a job is submitted and no worker has *marked itself* idle —
and a worker that has just finished a job has not done so yet for a few
microseconds. Two calls back to back (a skill's `dispatch`, then `tts.say` for
its answer; `explain`, then `tts.say` for the plain line) sometimes land in
that window.

Measured in text mode with stdout not a terminal, 300 turns of
"flip a coin" / "thanks" / gibberish: 1 worker became 3 at `HEAD` before M4,
and 3–4 on the current tree. With output on a terminal it is rarer (a `print`
between the two calls gives the worker its chance).

M4 hit it on every turn (guess, then speak — 2 workers became 9 in 80 turns),
which is why M4 and M4.5 make their reasoner and compound calls on short-lived
threads of their own (`run_detached`). The pre-existing pairs are untouched.

Fix, when someone takes it: give the orchestrator an executor of its own with
a small fixed size (it needs more than one: barge-in listens while Whisper
decodes, and under `serve` the link transcribes while a turn is in flight),
or start its workers up front so there is nothing left to add.

---

## 11. An utterance queued for an edge that left starts empty sessions

**Reported:** 2026-10-01, by the review of M4.5 (read, not reproduced end to end).
**Status:** open, not fixed. Pre-dates M4.5.
**Severity:** JARVIS talks to nobody — "Yes, sir?", "Standing by, sir.", about
once a second — until an edge reconnects.
**Owner:** unassigned.

`RemoteLink.triggered()` is "the queue is not empty". While no edge is
connected, `record_utterance` returns silence *without* taking anything off
the queue (`_disconnected` is set), so an utterance that was submitted and not
yet taken keeps `triggered()` true: `_await_wake` fires, the session hears
nothing, acknowledges, stands by, and it starts again.

M4.5 made one consequence worse and fixed that one: the leftover utterance
used to be answered for whichever edge connected next, and would now have
been *remembered* under it too. `RemoteLink.connect` drops what a different
device left queued. The same device coming back still gets its words, and the
loop above is untouched. Fix: do not report `triggered()` while disconnected,
or decide that a disconnect drops the queue.

---

## 12. A device id may end in a newline

**Reported:** 2026-10-01, by the review of M4.5.
**Status:** open in `jarvis/remote/protocol.py`; fixed in `jarvis/core/memory.py`.
**Severity:** low. Not a path traversal — an odd filename.
**Owner:** unassigned.

`_DEVICE_ID_RE` ends in `$` and is used with `.match`; `$` also matches just
before a trailing newline, so `"watch\n"` is a valid device id at `hello` and
becomes `data/remote/…/watch\n.json`. `jarvis/remote/firmware.py` uses the
same pattern. Fix: `\Z`, or `fullmatch` (what `memory.is_device_id` does).

---

## 13. "I'm unsure" removes a skill

**Reported:** 2026-10-01, by the review of M4.5.
**Status:** open in `jarvis/factory/flows.py`; M4 / M4.5 no longer use it.
**Severity:** a skill is quarantined, or a teach dialog proceeds, on a reply
that was not a yes.
**Owner:** unassigned.

`flows.ask_yes_no` looks for the yes-words *anywhere* in the reply: "unsure"
contains "sure", "incorrect" contains "correct", "yesterday" contains "yes".
It is what confirms *"Remove the 'x' skill for good, sir?"* (`RemoveSkillFlow`)
and *"… Is that right, sir?"* (`TeachFlow`). `ask_yes_no_or_none`, next to it,
matches whole words and is what background questions already use. Fix: make
the two flows use it (and decide what an unclear reply should do there).
