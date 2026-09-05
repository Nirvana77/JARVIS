# Milestone 2 — skill factory + seamless hot-swap: plan

**Status:** Implemented — see `PRD/milestone-2-skill-factory-outcome.md` for the outcome
**Date:** 2026-09-05
**Owner:** Kevin Lundell
**Branch:** `milestone-2-skill-factory` off `develop`, to be merged back —
same workflow as `milestone-1-voice-loop`.

This is the concrete implementation plan for the Milestone 2 item in
`jarvis-2026-rebuild.md` (skill factory, `teach` meta-skill, worker-process
retrain + idle merge-gate, `edit_skill`/`revert_skill`). M1 shipped the seams
this plugs into on purpose — `Orchestrator._staged` / `_merge_gate()` in
`jarvis/core/orchestrator.py` are a documented no-op, and `SkillManifest` /
`Registry` already carry the `origin` and `permissions` vocabulary M2 needs —
so this milestone plugs in rather than refactors. No commits happen without
the user asking, per the repo's standing convention.

---

## Key design decisions (resolving PRD ambiguity)

### 1. Merge-gate timing

Today `_merge_gate()` only fires in `run()`'s outer loop — between _wake
sessions_, not between turns within a follow-up session — and only swaps
`self.nlu`. Per the PRD ("in every turn's `finally` and on the idle tick"),
extend it: reset `self.state = "idle"` right after each `await self.handle(...)` inside `_session()`, then call the (now `async`)
`_merge_gate()` there too. `_merge_gate` also swaps `self.registry` and speaks
a queued announcement (`self._pending_announcement`) via `self._speak()`.
This makes "try me" immediately true in the fast retrain path, without
weakening the idle-only invariant.

### 2. Promotion happens last, exactly per the PRD stage table

The generated module is validated → sandboxed → _retrained against a corpus
built from `registry.manifests() + [new manifest]` without touching disk_ →
self-check → **voice confirm** → only on "yes" is the file moved
`jarvis/skills/staging/<name>.py` → `jarvis/skills/learned/<name>.py` and
`self._staged` set. On "no" or a failed self-check, both the staging file and
the unused trained-model version directory are discarded — nothing is
promoted, nothing is swapped.

### 3. Fast/slow UX split, simplified for voice

The retrain step gets a fast budget (~8s — fitting a `LogisticRegression`
over a few hundred cached MiniLM embeddings is CPU-light, so this is the
expected case, not the exception):

- **Fast** (worker returns within budget): self-check inline, then the
  step-7 voice confirm ("I can now `<description>`. Shall I keep it, sir?")
  happens in the same turn.
- **Slow** (budget exceeded): say "I'll practice that and let you know, sir,"
  detach the worker into `self._pending_retrains`, and — since an unprompted
  async yes/no is awkward for a voice UI — **auto-confirm** on success and
  announce "I've learned '`<name>`', sir. My capabilities are updated." at the
  next safe merge point.

    This is a **deliberate deviation** from the literal "present report → voice
    confirm" step, and only for the slow path — flagged here up front the same
    way M1's outcome doc flagged its own deviations (e.g. folding `sleep` into
    `goodbye`).

### 4. Directory layout (extends, doesn't fight, the PRD diagram)

- `jarvis/skills/staging/` — transient generated-but-unconfirmed modules (PRD
  location; gitignored, created at runtime).
- `jarvis/skills/learned/` — promoted skills, discovered by `Registry`
  alongside `builtin/` (new package; doesn't exist yet).
- `data/skills/_versions/<name>/v<N>.py` + `meta.json` — last 3 versions per
  learned skill, mirroring the existing `data/models/nlu/v<N>/` convention.
  Chosen over the PRD's literal `timer.v1.py` naming because a dotted
  filename isn't an importable module name — the same kind of pragmatic
  deviation M1 documented for the `sleep` intent.
- `data/skills/_quarantine/<name>.<timestamp>.py` — self-check failures and
  skills displaced by `revert_skill`, kept for inspection, never imported.

### 5. AST allowlist mapping (`factory/validate.py`)

Matches the PRD's denylist verbatim: `socket`, `http`, `urllib`, `requests`,
`httpx`, `ftplib`, `smtplib` (import or attribute-call root) require `"net"`;
`subprocess`, `ctypes`, `os.system`, `os.popen`, bare `eval`/`exec`/
`__import__` require `"shell"`; a builtin `open(...)` call requires
`"fs_read"` or `"fs_write"`. Anything the generated manifest requests beyond
`{"pure", "notify"}` also triggers an explicit voice grant/deny step in the
flow (PRD: "anything more is granted explicitly at install time by voice") —
in addition to, not instead of, the static check.

### 6. Tests never call the real Claude API or need real sandbox root/namespace support

`build()`, the flows, and the orchestrator all take an injectable
`generate` / `sandbox`, mirroring the DI pattern M1 already uses for
`wake`/`mic`/`stt`/`tts`/`nlu`/`reasoner`. Sandbox tests that need real
`unshare` net-isolation skip cleanly when it's unavailable, matching the
existing `embedder` fixture's skip-on-unavailable pattern in
`tests/conftest.py`.

---

## New files

- `jarvis/factory/__init__.py`
- `jarvis/factory/spec.py` — `SkillSpec(name, description, examples:
list[str], permission_hint: frozenset[str] = frozenset({"pure"}))`.
- `jarvis/factory/claude_client.py` — thin wrapper around the `anthropic`
  SDK, same env-var handling as `libs/anthropic_helper.py`
  (`ANTHROPIC_API_KEY` / legacy `api_key`, `ANTHROPIC_MODEL` default
  `claude-opus-5`, `ANTHROPIC_WORKSPACE_ID` → `anthropic-workspace-id`
  header). Exposes `generate_skill(spec, existing_source: str | None) ->
GeneratedSkill` that prompts Claude (system prompt built from
  `jarvis/skills/contract.py`'s own docstring + `jarvis/skills/builtin/note.py`
  as a worked example) to return two fenced blocks (```python skill /
    ```python test) and parses them with a regex. This is the *default*
    generator; nothing else in the factory imports `anthropic` directly.
    ```
- `jarvis/factory/build.py` — `GeneratedSkill(name, module_source,
test_source)`, `BuildError`, `build(spec, *, generate=default_generate,
existing_source=None)`. Pure orchestration + name-sanity checks; no
  network itself (delegates to the injected `generate`).
- `jarvis/factory/validate.py` — `ValidationError`, `validate(module_source,
known_names: set[str]) -> SkillManifest`. Steps: `ast.parse`; locate the
  top-level `MANIFEST = SkillManifest(...)` call, `ast.literal_eval` its
  keyword args, construct a real `SkillManifest` (reuses
  `jarvis/skills/contract.py`'s own validation, e.g. the identifier check);
  reject if `manifest.name` collides with `known_names` (unless it's an
  edit-in-place, handled by the caller passing an exclusion); locate `def
run(ctx, **...)`; run the AST denylist walk from decision 5 above; return
  the manifest.
- `jarvis/factory/sandbox.py` — `SandboxResult(ok, stdout, stderr,
returncode)`, `SandboxError`, `SubprocessSandbox(timeout_s=10.0,
mem_mb=256, cpu_s=5)`. `net_isolated` probed once via
  `shutil.which("unshare")` + a no-op `unshare --user --map-root-user --net
true`. `run_tests(module_path, test_path, permissions)` and
  `dry_run(module_path, manifest, sample_params)` both build a small on-disk
  runner script, invoke `sys.executable -I <runner> <args>` (wrapped in
  `unshare --user --map-root-user --net` when `net_isolated and "net" not in
permissions`), with a `preexec_fn` setting `resource.RLIMIT_AS`/
  `RLIMIT_CPU`, `capture_output=True`, `timeout=timeout_s`.
- `jarvis/factory/flows.py` — `TeachFlow`, `EditSkillFlow`,
  `RevertSkillFlow`. Each takes `ask: Callable[[str], Awaitable[str]]`,
  `say: Callable[[str], Awaitable[None]]`, `registry`, `config`, `generate`,
  `sandbox`, and returns a small outcome object; **they never touch
  `orchestrator._staged` or `registry` directly** — the orchestrator's
  `_promote_and_retrain()` (see below) is the sole writer, called by
  `handle()` after a flow returns an accepted outcome. Shared helpers here:
  `_ask_yes_no`, `_match_skill_name` (fuzzy-ish substring match against
  `registry.names()`), the permission-grant sub-dialog from decision 5.
- `jarvis/nlu/retrain_worker.py` — `RetrainWorker` wrapping a
  `multiprocessing.Process` + `multiprocessing.Queue` around
  `jarvis.nlu.train.train()` (reused as-is, no changes needed there);
  `start(examples, embedding_model, out_dir)`, `poll(timeout) -> ("ok",
TrainResult) | ("error", str) | None`.
- `jarvis/skills/learned/__init__.py` — empty package so `Registry.discover`
  can import it even with zero skills.
- `tests/test_factory_validate.py`, `tests/test_factory_build.py`,
  `tests/test_factory_sandbox.py`, `tests/test_retrain_worker.py`,
  `tests/test_flows.py`, plus a merge-gate timing test added to
  `tests/test_orchestrator.py`.
- `PRD/milestone-2-skill-factory-outcome.md` — outcome doc written once this
  plan is implemented, same shape as `PRD/milestone-1-voice-loop.md`.

## Modified files

- `jarvis/skills/registry.py` — `discover`/`rebuilt` scan both
  `jarvis.skills.builtin` and `jarvis.skills.learned` (skip a package that
  doesn't exist/is empty); nothing else changes (dispatch, manifest lookup
  are already origin-agnostic).
- `jarvis/core/orchestrator.py`:
    - `_META_ACTIONS` gains `"teach": "teach"`, `"edit_skill": "edit_skill"`,
      `"revert_skill": "revert_skill"`.
    - `handle()` grows branches for those three actions (build the matching
      `*Flow`, run it, hand any accepted outcome to `_promote_and_retrain`),
      and the `UNKNOWN` branch grows an "offer to learn" yes/no before falling
      back to the plain unknown line (seeds `TeachFlow` with the failed
      utterance as a candidate description).
    - New: `_ask(prompt) -> str` (speak + `_next_utterance`, reusing the
      existing capture/transcribe path verbatim), `_ask_yes_no(prompt) ->
bool`.
    - New: `_promote_and_retrain(name, module_source, test_source, manifest,
*, existing_version_dir=None)` — implements design decisions 2 and 3:
      stage → sandbox already done by the flow before this is called →
      retrain (via `RetrainWorker`) → self-check → confirm (fast) or
      auto-confirm (slow, via `_pending_retrains` + the background drain
      task) → promote file + version bookkeeping → set `_staged` +
      `_pending_announcement`.
    - `_merge_gate` becomes `async def _merge_gate()`: swaps `self.nlu` _and_
      `self.registry`, then if `self._pending_announcement` speaks it via
      `self._speak()` and clears it. Called from `_session()` after every
      `handle()` (state reset to `"idle"` first) in addition to the existing
      call site in `run()`.
    - New background task `_drain_retrain_results()`, started alongside the
      main loop in `run()`, polling `self._pending_retrains` and routing
      completed results through the same self-check/promote path used by the
      fast path.
- `jarvis/app.py` — `build_orchestrator` constructs a `ClaudeClient` (best
  effort — missing key degrades to "teach unavailable" the same way the
  reasoner degrades to "no Ollama", never a hard failure) and a
  `SubprocessSandbox`, passing both into `Orchestrator`; `selftest()` prints
  a `factory:` line (key present/absent, sandbox net-isolation
  available/not).
- `jarvis/config.py` / `config.toml` — new `[factory]` section:
  `FactoryConfig(model: str = "claude-opus-5", sandbox_timeout_s: float =
10.0, sandbox_mem_mb: int = 256, sandbox_cpu_s: int = 5, fast_budget_s:
float = 8.0)`; `Config.anthropic_api_key` / `Config.anthropic_workspace_id`
  properties reading env exactly like `hf_token` does today.
- `intents.json` — three new intents: `teach` (action `teach`), `edit_skill`
  (action `edit_skill`), `revert_skill` (action `revert_skill`), each with a
  handful of patterns and empty `responses` (the flows speak everything).
- `.gitignore` — add `jarvis/skills/staging/`, `data/skills/_versions/`,
  `data/skills/_quarantine/`.
- `PRD/jarvis-2026-rebuild.md` — mark Milestone 2 ✅ DONE with a summary and a
  link to the outcome doc, once implemented (same convention used for Phase 0
  / M1).

---

## Verification

- `python -m pytest` — all new + existing tests green; factory/sandbox tests
  that need real `unshare` skip cleanly where it's unavailable (documented,
  same pattern as the `embedder` fixture).
- New **merge-gate timing test**, per the PRD's own M2 verification bullet:
  force `state="acting"`, stage a fake replacement, assert no swap; set
  `state="idle"`, call `_merge_gate()`, assert the swap _and_ the queued
  announcement fired.
- `factory.validate` unit tests: a module importing `socket` without `"net"`
  is rejected; a module with an incomplete `MANIFEST` is rejected; a clean
  minimal skill (mirroring `note.py`'s shape) passes.
- `factory.sandbox` unit test: a skill whose test opens a socket
  fails/is network-isolated (skips if `unshare` is unavailable in this
  environment).
- `python check_setup.py` stays green (aside from the pre-existing
  `ANTHROPIC_WORKSPACE_ID` row, unchanged).
- Manual (for the user, needs a working Anthropic key +
  `ANTHROPIC_WORKSPACE_ID` and a live mic): _"Jarvis, learn how to set a
  timer"_ → full teach dialog → confirm → _"Jarvis, set a timer for 10
  seconds"_ fires the new skill; _"Jarvis, revert the timer skill"_ rolls it
  back.
