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

## 5. A question about the documents that does not sound like one is "unknown"

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

## 6. Changing `intents.json` or a skill's examples does not retrain the NLU

**Reported:** 2026-10-01, M5 dry run.
**Status:** open.
**Severity:** silent — the old model keeps answering, and nothing says so.

`ensure_nlu` (`jarvis/app.py`) retrains only when a registered skill is missing
from the model's labels. Editing patterns or examples for intents the model
already knows changes nothing until `python -m jarvis nlu rebuild`. It showed
up as a dry run still classifying with the examples from an hour earlier. The
legacy code watched `intents.json`'s mtime; the rebuild has no equivalent.

### Fix, when someone takes it

Store a hash of the corpus in the model's `meta.json` and have `ensure_nlu`
compare it.

---

## 7. `Persona.phrase` — a model per call, on the event loop, and after the answer

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

## 8. Knowledge base: small things the M5 review found and left

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

