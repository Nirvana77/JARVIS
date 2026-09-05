# Milestone 2 — skill factory + seamless hot-swap: outcome

**Date:** 2026-09-05
**Branch:** `milestone-2-skill-factory` (off `develop`, not yet merged)
**Result:** ✅ The skill factory, the `teach`/`edit_skill`/`revert_skill` voice
dialogs, worker-process retraining, and the idle merge-gate all work
end-to-end against real subprocesses (real `pytest` runs, real `unshare`
network isolation, a real `multiprocessing` retrain), verified by actually
conversing with the running program via `python -m jarvis text` (see below —
this is what caught the one real bug this milestone shipped with). Not
exercised here: a live Claude call (blocked by the pre-existing
`ANTHROPIC_WORKSPACE_ID` gap — see below) and anything needing a live mic.

This closes Milestone 2 in `PRD/jarvis-2026-rebuild.md`. Design decisions that
resolved ambiguity in the PRD are recorded in `PRD/milestone-2-skill-factory.md`
(the plan this outcome implements) and aren't repeated here except where the
implementation deviated from that plan.

---

## What shipped

| Area | Module(s) | Notes |
|---|---|---|
| Skill spec | `jarvis/factory/spec.py` | `SkillSpec` — name/description/examples gathered by voice, nothing else |
| Claude client | `jarvis/factory/claude_client.py` | `ClaudeClient`, capability-probed like `Reasoner`; parses two fenced code blocks out of Claude's reply |
| Build | `jarvis/factory/build.py` | Pure orchestration around an injected `generate`; no network of its own |
| Validate | `jarvis/factory/validate.py` | AST-only: manifest completeness/literal-ness, `run()` presence, the PRD's import/call denylist mapped to `net`/`shell`/`fs_read`/`fs_write` |
| Sandbox | `jarvis/factory/sandbox.py` | `SubprocessSandbox` — `python -I`, `RLIMIT_AS`/`RLIMIT_CPU`, `unshare --user --net` when available |
| Dialogs | `jarvis/factory/flows.py` | `TeachFlow`, `EditSkillFlow`, `RevertSkillFlow` — voice-only, never touch orchestrator state |
| Retrain | `jarvis/nlu/retrain_worker.py` | `RetrainWorker` — `multiprocessing.Process` + `Queue` around the unchanged `jarvis.nlu.train.train()` |
| Orchestrator glue | `jarvis/core/orchestrator.py` | `_promote_and_retrain`/`_finish_retrain`/`_merge_gate`/`_drain_retrain_results` — the sole writer of `_staged`/`registry` |
| Skills | `jarvis/skills/learned/` | New, empty package; `Registry.discover` now scans it alongside `builtin/` |
| Config | `jarvis/config.py`, `config.toml` | `[factory]` section; `Config.anthropic_api_key`/`anthropic_workspace_id` (`.env` only) |
| Intents | `intents.json` | `teach`, `edit_skill`, `revert_skill` — empty `responses`, the flows speak everything |
| Text-mode rig | `jarvis/audio/text_io.py`, `jarvis/app.py` (`build_text_orchestrator`/`run_text_mode`), `jarvis/__main__.py` (`text` subcommand) | Not in the original plan — added mid-milestone so the whole pipeline (including `teach`/`edit_skill`/`revert_skill`) could be driven and verified by conversing with the real program without a mic. This is what the "Bug found" section below and the CLAUDE.md workflow update both depend on. |
| Tests | `tests/test_factory_{validate,build,sandbox}.py`, `test_retrain_worker.py`, `test_flows.py`, `test_text_io.py`, + additions to `test_orchestrator.py`/`test_skills.py` | 113 tests total (was 67), `python -m pytest` |

## Build → validate → sandbox

- `ClaudeClient.generate_skill()` sends one non-streaming `messages.create`
  call (`claude-opus-5`, adaptive thinking) with a system prompt built from
  the skill contract's own rules plus `note.py` as a worked example, and
  parses out a ```python skill``` block and a ```python test``` block.
- `validate()` never imports or executes the generated code — pure `ast`.
  It extracts `MANIFEST = SkillManifest(...)` via `ast.literal_eval` on each
  keyword (rejecting anything that isn't a literal), constructs a real
  `SkillManifest` (reusing its own identifier/permission-vocabulary checks),
  confirms a top-level `run()` exists, and walks every `Import`/`Call` node
  for the PRD's denylist. A name collision with an existing skill is rejected
  unless `allow_name` (the skill being edited) matches.
- `SubprocessSandbox` runs the generated pytest module against the generated
  skill module, then a dry-run `run()` call with sample params synthesized
  from `MANIFEST.params`'s declared types. Both go through `python -I` with
  `RLIMIT_AS`/`RLIMIT_CPU` and, when `unshare --user --map-root-user --net`
  is available (probed once at construction), no network namespace unless
  `"net"` was granted.
- Anything beyond `{"pure", "notify"}` in the generated manifest triggers an
  explicit voice grant/deny in the flow — *in addition to* validate.py's
  static check, per the PRD ("granted explicitly at install time by voice").

**Found via smoke-testing, fixed before writing the test suite:** the default
`sandbox_mem_mb` (256) was too tight — `pytest.main()`'s own plugin-rewrite
discovery (`importlib.metadata` scanning every installed package's dist-info)
raised a bare `MemoryError` under that `RLIMIT_AS` in this venv, before it
ever got to running the generated test. Confirmed the floor is between 256
and 384 MB; the default is now **512 MB** (`config.toml [factory]
sandbox_mem_mb`).

## Retrain + merge gate

- `RetrainWorker` pickles `Example` rows as plain `(text, label, source)`
  tuples (not the dataclass) so the worker process doesn't need `jarvis` on
  its import path before it's re-added — measured **~7s** for a 432-example
  corpus (embed + fit), comfortably inside the 8s `fast_budget_s` default,
  confirming the PRD's own assumption that the fast path is the common case.
- `_merge_gate()` now fires in two places instead of one: the existing
  wake-session boundary in `run()`, and — new — right after every
  `handle()` call inside `_session()`'s follow-up loop (state is reset to
  `"idle"` first). This is what makes "try me" true in the same follow-up
  window a `teach` just ran in, per design decision 1 in the plan doc.
- `_promote_and_retrain()` never writes `skills/learned/<name>.py` until
  *after* retrain + self-check + (fast-path) voice confirm all pass — the
  staging file lives in `jarvis/skills/staging/` until then. A declined
  confirm or failed self-check deletes both the staging file and the
  never-staged trained-model version directory, leaving `self.nlu` and disk
  state consistent with each other.
- Self-check (`_self_check`) reuses the same three seed probes
  `--selftest` already prints, plus every example of the new/changed skill.

## Versioning / quarantine

- `edit_skill`: the pre-edit `learned/<name>.py` is copied to
  `data/skills/_versions/<name>/v<K>.py` before being overwritten; pruned to
  the last 3.
- `revert_skill`: the currently-live file is moved (not deleted) to
  `data/skills/_quarantine/<name>.<UTC timestamp>.py`; the restored version's
  file is removed from `_versions/` since it's live again.
- Deviation from the PRD's literal `timer.v1.py` naming: versions live under
  `data/skills/_versions/<name>/vN.py`, mirroring the existing
  `data/models/nlu/v<N>/` convention, because a dotted filename isn't an
  importable module name. Recorded up front in the plan doc.

## Bug found (and fixed) via the `python -m jarvis text` dry-run

Every unit test for `teach`/`edit_skill`/`revert_skill` passed with the
implementation as first written — the bug below only surfaced by actually
*conversing* with the running program end-to-end, which is exactly why
that's now a required step (see `CLAUDE.md` → "Implementing PRD work") and
not just "the test suite is green."

**Symptom:** after a real `teach`, saying "edit the coin flip skill"
answered *"I don't have any skills I can do that with yet, sir"* — even
though the skill had just been learned and was already dispatching
correctly.

**Cause:** nothing in the factory prompt (`claude_client.py`'s `_CONTRACT`)
or `validate.py` ever set/checked `MANIFEST`'s `origin` field, so every
generated skill's manifest kept `SkillManifest`'s own default,
`origin="builtin"`. `EditSkillFlow`/`RevertSkillFlow` filter candidates by
`origin == "learned"` — so *every* real teach would have silently made that
skill un-editable and un-revertable, forever. `validate()` couldn't have
caught this by construction (it validates the module *before* promotion,
and origin only matters *after* the file is re-imported from
`skills/learned/`).

**Fix:** `Registry.discover()` now normalizes `manifest.origin` to match the
package a module was actually found in (`jarvis/skills/registry.py`,
`_ORIGIN_FOR_PACKAGE`), overriding whatever the module's own code claims —
authoritative by construction, regardless of what a generated (or
hand-written) manifest gets wrong. Regression tests:
`test_a_learned_skills_origin_is_forced_even_if_the_code_omits_it` and
`test_a_builtins_origin_is_forced_to_builtin_even_if_misdeclared` in
`tests/test_skills.py`.

**Also found and fixed in the same dry-run pass, before writing the test
suite:** `test_registry_discovers_the_four_builtins` used the default
`Registry.discover(config)`, which — now that `skills/learned/` is real,
mutable, per-install state — picked up whatever the developer's machine
happened to have actually taught. Fixed by having that one test call
`Registry.discover(config, packages=(BUILTIN_PACKAGE,))` instead; this is a
test-isolation fix, not a loosened assertion — the test still asserts the
exact four-name list.

## Verification

Automated (ran here):
```
python -m pytest                     # 113 passed (was 67 before this milestone)
python -m jarvis nlu rebuild         # 450+ examples incl. teach/edit_skill/revert_skill
python -m jarvis --selftest          # factory: claude:claude-opus-5, sandbox: net-isolation yes
python check_setup.py                # unchanged; only the pre-existing ANTHROPIC_WORKSPACE_ID row fails
```

`tests/test_factory_sandbox.py` and `tests/test_retrain_worker.py` run real
subprocesses (real `pytest`, real `unshare`, a real `multiprocessing.Process`
retrain) rather than mocking them — including a test that a socket-opening
skill's outbound connection is actually blocked by kernel network isolation
when `unshare` is available (it is, in this environment), skipping cleanly
where it isn't.

**End-to-end dry-run via `python -m jarvis text`** (the text-mode rig added
alongside this milestone — see `jarvis/audio/text_io.py`), against the real
`run()` loop with real NLU/registry/persona/sandbox/retrain and only the
Claude network call replaced (the pre-existing `ANTHROPIC_WORKSPACE_ID` gap
makes a live call fail before it ever reaches the factory logic — this
substitutes a fixed response at exactly that boundary, nothing downstream):

- Web search variety ("search black holes", "who is albert einstein", "what
  is the eiffel tower", "tell me about the roman empire") — all correct,
  real Wikipedia calls.
- `teach` → name → description → example → real retrain (~7s, comfortably
  inside `fast_budget_s`) → self-check → voice confirm → **"flip a coin" in
  the same wake session, no re-wake, correctly dispatches to the brand-new
  skill** — the concrete proof of design decision 1 ("try me" is immediate).
- `edit_skill` → matched the just-taught skill by name → real re-retrain →
  confirm → the *edited* behavior (a louder response) took over immediately.
- `revert_skill` → matched it again → the pre-edit file was restored from
  `_versions/`, the edited one moved to `_quarantine/` with a timestamp, a
  real re-retrain ran, and the original behavior came back — all without a
  second voice confirm, per the design.
- This pass is what found the `origin` bug above — worth calling out because
  every one of the unit tests for these three flows was green throughout.

A prior, smaller ad hoc pass (kept out of the repo, scratchpad only) also
confirmed `TeachFlow` against the **real** Anthropic API hits exactly the
known `ANTHROPIC_WORKSPACE_ID` gap (400: "API key is not scoped to a
workspace") and degrades exactly as designed — not a new blocker, already
tracked as the user's to resolve.

Not verified here (no audio device): the mic/wake-word/STT path itself, and
the "slow path" auto-confirm-on-background-completion behavior (only
exercised via the merge-gate unit tests, not a real >8s retrain — 432-450
examples reliably retrains in ~7s, under budget, so the slow path is the
uncommon case by design).

**For the user to run** once `ANTHROPIC_WORKSPACE_ID` is set (this exercises
the real Claude call, the one thing the dry-run above couldn't):
```
python -m jarvis                                  # say "hey jarvis, learn how to set a timer"
                                                   #   -> full teach dialog -> confirm
                                                   #   -> "hey jarvis, set a timer for 10 seconds"
                                                   # say "hey jarvis, edit the timer skill"
                                                   # say "hey jarvis, revert the timer skill"
# or, without a mic at all:
python -m jarvis text   # same dialog, typed instead of spoken
```

## Dependencies

None new — `anthropic` was already a dependency (`libs/anthropic_helper.py`);
`multiprocessing` and `ast` are stdlib. `unshare` is a system binary (already
present on this Fedora box via `util-linux`), probed at runtime rather than
required.

## Deferred / out of scope

- M3: knowledge base (RAG), `remember` builtin, doc-folder ingestion.
- M4: barge-in over TTS, systemd unit, `pyproject.toml`, README/CLAUDE.md
  rewrite.
- Not attempted: `DockerSandbox`/`PodmanSandbox` (PRD marks these optional,
  stronger backends — `SubprocessSandbox` is the PRD's primary path and is
  what's implemented).
