# Known issues

Things that are wrong and not yet fixed, with enough written down that the fix
doesn't start from zero. Fixed entries move out of here into the milestone
outcome doc that fixed them.

---

## 1. The skill factory rejects what it builds: "sandbox tests failed", repeatedly

**Reported:** 2026-09-25, from live use (`python -m jarvis serve` + edge).
**Status:** open, not investigated.
**Severity:** M2's headline feature — `teach` — does not produce a skill.
**Owner:** unassigned.

### What was seen

Teaching a new skill fails at the sandbox stage over and over: Claude generates
the skill, and its tests fail, so the job sets it aside and nothing is learned.
Spoken, that comes out as *"'<name>' failed its own tests, sir. I've set it
aside."* (`jarvis/factory/jobs.py`).

### Where it fails

`LearningJob._generate_validate_sandbox()` in `jarvis/factory/jobs.py`:

```
build (Claude) → validate (AST denylist) → permission → sandbox.run_tests   ← here
                                                      → sandbox.dry_run
```

The failing branch is the `test_result.ok is False` one, which already logs the
subprocess's stderr at WARNING:

```
log.warning("sandbox tests failed for %s:\n%s", manifest.name, test_result.stderr)
```

So **the diagnosis is almost certainly already in the terminal scrollback of a
failing run** — that stderr is the first thing to read, and nothing below is
worth guessing about until it has been.

### Confound to rule out first

M3 fixed a bug (outcome doc, "Found by the first live run") where `_ask` timed
out during a `teach` dialog and aborted it with *"Never mind, then."* That
looked like the factory failing and was not. **Re-run a `teach` dialog on the
fixed code before investigating anything here**; the symptom may be different
or gone.

### Candidates, in the order worth checking

1. **`RLIMIT_AS` = `[factory] sandbox_mem_mb` (512 MB).** `config.toml` already
   carries the note that "pytest's plugin-rewrite discovery alone needs >256MB
   in a venv this size (scans every installed dist-info)". This venv has since
   grown faster-whisper, ONNX Runtime, fastembed, websockets and more, so 512 MB
   may no longer be enough for pytest to *start*. A `MemoryError` or a bare
   non-zero exit with no test output points straight here. Cheap to test: raise
   `sandbox_mem_mb` to 1024 and see whether the same skill passes.
2. **`RLIMIT_CPU` = `sandbox_cpu_s` (5 s) / `sandbox_timeout_s` (10 s).** Same
   shape of problem — pytest's collection on a large venv is not free. A
   `SandboxError` (timeout) rather than a test failure points here.
3. **The generated tests themselves.** Claude may be writing tests that need
   the network (`unshare` is unavailable on this box — `--selftest` reports
   `net-isolation no`, so a network test would *pass* locally and fail on a box
   where isolation works, or vice versa), or that import something `python -I`
   cannot see. Read the generated test file: failures are kept, not deleted —
   see `config.skill_quarantine_dir` (`data/skills/_quarantine/`).
4. **The prompt.** If the tests are wrong in a consistent way, the fix is in
   `jarvis/factory/claude_client.py`'s prompt rather than in the sandbox.

### What to capture next time it happens

- the `sandbox tests failed for <name>` WARNING and the stderr under it;
- the quarantined skill and its test file from `data/skills/_quarantine/`;
- whether `python -m jarvis --selftest` reports `net-isolation yes` or `no`;
- the skill name and the words used to teach it.

### Why the test suite doesn't catch it

`tests/test_factory_*.py` exercise the factory against a **fake Claude** and a
real sandbox with small, hand-written skills, and they pass. Whatever is wrong
is in the live path: either real generated code, or resource limits that the
small test skills never reach.

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
