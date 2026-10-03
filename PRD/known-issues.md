# Known issues

Things that are wrong and not yet fixed, with enough written down that the fix
doesn't start from zero. Fixed entries move out of here into the milestone
outcome doc that fixed them (#4, #6, #7, #11, #12, #13 and #15:
`PRD/steady-ship-outcome.md`).

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

## 14. A question about the documents that does not sound like one is "unknown"

**Reported:** 2026-10-01, M5 dry run (`PRD/milestone-5-knowledge-base-outcome.md`).
**Status:** open — a limit of routing by intent, not a regression.
**Severity:** the knowledge base has the answer and is never asked.

"How big is the water tank" is in the manual and scores 0.50 against the right
chunk — but the NLU has no recall-shaped frame to hang it on, so it is
`unknown` (the out-of-domain guard: similarity 0.22 to any training phrase) and
nothing is retrieved. "What
does the manual say about the water tank" works. Relatedly, "note that …" with
a subject the classifier has not seen can be heard as `remember` (measured:
"note that the bins go out on thursday" → remember 0.46, note 0.24); both end up in the
knowledge base, but the line is missing from `notes.txt`.

### Fix, when someone takes it

Ask the knowledge base before giving up on an `unknown`: if a chunk clears
`search_min_score`, answer from it; otherwise say the usual line. That belongs
with Milestone 4 (misheard-command reasoning), which already owns what happens
to an utterance the NLU could not place — on the voice path a garbled
transcription must not be answered from a note it happens to resemble.

---

## 16. `Persona.phrase` — a model per call, on the event loop, and after the answer

**Reported:** 2026-10-01, reading the answer path for M5.
**Status:** open. Not reproduced live: there is no Ollama on the dev machine.
**Severity:** latency, and possibly a lost source, only when a reasoner is up.

- `Persona._nearest_style_lines` (`jarvis/core/persona.py:165-181`) builds a
  new `fastembed.TextEmbedding` on every call (twice on the first).
- `handle()` calls `persona.phrase(...)` synchronously, so the Ollama request
  (30 s timeout) blocks the event loop.
- A knowledge answer composed by Ollama is then rewritten by Ollama again, by
  `persona.phrase`: two generations per recall, and the rewrite is free to
  drop the source the first one was made to name.

### Fix, when someone takes it

Give the persona one embedder, run `phrase` in `asyncio.to_thread`, and let a
skill mark its line as already in voice (or pass the persona's system prompt
to the knowledge compose step and skip the rewrite).

---

## 17. Knowledge base: small things the M5 review found and left

**Reported:** 2026-10-01, independent review of `jarvis/knowledge/`.
**Status:** open, each judged not worth fixing yet.

- **"Did the model name its source?" is a substring check.** A file called
  `notes.md` is "named" by any reply containing the word "notes", so its
  source is not appended. And the source that is appended is the best hit's,
  which need not be the excerpt the model used.
- **One file reached through two symlinks is indexed twice** — paths are not
  resolved.
- **Two processes with different backends on one database.** If one process
  cannot load sqlite-vec and writes between another's read and write of the
  same source, the index can be marked in step when it is not. Both normally
  share a venv, so both have it or neither does.
- **Document text goes into the local model's prompt as it is.** A document
  can steer the spoken answer. It stays on this machine, and Claude is never
  involved, but the docs folder is trusted input.

---

## 18. "Forget how to flip a coin" flips a coin

**Reported:** 2026-10-03, the steady-ship dry run.
**Status:** open.
**Severity:** a request to remove a skill runs it instead. Harmless for a coin;
not for a skill that acts.

"forget how to flip a coin" classified as `flip_a_coin` 0.43, against
`remove_skill` 0.22. Then `flip_a_coin` ran, because a learned skill has no
lead-in or meta-action bar to clear. The skill's own name in the sentence
outweighs the frame around it. "remove the flip a coin skill" routes correctly
(it asks which skill, then confirms). M4.5 already measured `remove_skill`
losing confidence as classes are added ("you can forget how to flip a coin"
0.68 → 0.56).

### Fix, when someone takes it

A leading "forget how to" / "remove" / "delete" frame should decide this
before the skill's examples can, much as `slots.has_lead_in` gates
`note`/`remember`/`forget_fact`. The orchestrator could check those frames
first. Alternatively, a dispatch to a learned skill whose utterance starts with
a removal verb could be treated as unclear.

---

## 19. The pod and the dev brain keep learned skills in different places

**Reported:** 2026-10-03, while planning M7.
**Status:** open; out of M7's scope by plan.
**Severity:** a skill the pod builds or repairs by itself (M7) is unknown to the
dev brain, and the reverse. The two share `data/` but not the learned code.

The pod mounts `/app/jarvis/skills/learned` from its `jarvis-state` PVC. The dev
brain reads `jarvis/skills/learned/` in its checkout. Everything else that is
learned lives in the shared `data/`: phrasings, the NLU model versions, a
skill's version history in `data/skills/_versions/`. Two consequences:

- A learned phrasing whose label is a skill only the pod has is ignored by the
  dev brain's corpus (`build_corpus` drops labels it has no skill for). This is
  correct, but it means the dev brain's model differs from the pod's.
- `data/skills/_versions/<name>/` can hold history for a skill the other brain
  has never seen, and "go back to the previous version" there restores into
  the wrong tree.

### Fix, when someone takes it

Move learned skills under `data/` (say `data/skills/learned/`, imported by path
the way `Registry` already imports the package) so both brains share them, or
mount the PVC's directory into the dev checkout. Then decide which brain may
write there: only the one with `[learning] enabled`.
