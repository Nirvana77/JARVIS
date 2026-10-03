# Milestone 8, part A: JARVIS rewrites its own builtins: outcome

**Date:** 2026-10-03
**Branch:** `self-rewrite` (off `m8-plan`, off `develop`)
**Result:** ✅ Suite 1227 passed, 2 skipped. In a real dry run (the cluster's
Ollama, the real Claude API, the real sandbox):
- a planted bug in `clock` made it fail;
- JARVIS rewrote it in the background, passed the gate, swapped it in with no
  restart, and said so;
- asked again, the rewrite answered;
- "undo that" asked first (0.59, just under the bar), then put the previous
  version back.

Plan: `PRD/milestone-8-self-rewrite-seamless-redeploy.md`. The owner chose:
skills and builtins only; deploy, then tell; commits to its own branch,
pushed.

## Built

| Piece | Where |
|---|---|
| Overrides: `data/skills/overrides/<builtin>.py` replaces the packaged builtin for both brains. A broken, misnamed or non-builtin file there is ignored and the packaged one stays | `Registry._apply_overrides`, `jarvis/skills/overrides/`, `Config.skill_overrides_dir` |
| A builtin that raises is rewritten (it is still written to `builtin-failures.jsonl`). Past `max_repairs_per_skill_per_day` it is left for a person, never switched off | `Orchestrator._after_failure`, `_start_repair` |
| The gate for a builtin: validation, generated tests, dry run, the failing call replayed, **its recent good calls from the interaction log replayed**, and **the repo's own tests for it** run in the sandbox with the rewrite standing in for `jarvis.skills.builtin.<name>` (`-k <name>`). The packaged builtin's permissions count as granted | `LearningJob._regressions`, `SubprocessSandbox.run_repo_tests`, `tests_for_builtin`, `LearningRequest.pre_granted` |
| Probation: the first `[learning] probation_calls` (5) uses. A failure or a correction puts the previous version back on its own (the override's previous source, or the packaged builtin) and says so at the next pause | `_self_changes`, `_on_probation`, `_revert_self_change` |
| "Undo that" (intent `undo_change`) puts back the last self-rewrite, asking first when the classifier is unsure. No new rewrite of that skill the same day | `_undo_change` |
| Learned-skill repairs (M7) get the same probation, undo and regression replay | the same |
| Publishing: a kept rewrite and a revert are committed to `jarvis/self` through GitHub's contents API (the pod has no git), authored "JARVIS (on behalf of Kevin Lundell)" at the owner's address. `develop`/`master`/`main` are refused. No `JARVIS_GITHUB_TOKEN`: logged only | `jarvis/learning/publish.py` |

## Found on the way, fixed

- **The sandbox handed generated code the brain's secrets.** Sandboxed runs
  inherited the whole environment (`ANTHROPIC_API_KEY`, `HF_TOKEN`, edge
  tokens), and on this machine `unshare` is unavailable, so there was no
  network isolation either. Now a short allowlist plus `JARVIS_SANDBOX=1`,
  under which `jarvis.config` reads no `.env`. This also protected M7's
  learned skills.
- **The dry run's stand-in `ctx` had no `now()`**, which M7 tells Claude to
  use; also no `config`/`edges`/`memory`/`knowledge`.
- **The validator refused every builtin** (`permissions=frozenset({...})`);
  it now accepts exactly that form.
- **"Asked again" ruled out a skill that had just been rewritten**, so the
  rewrite never got its first use (seen in the dry run). A question re-asked
  after its skill was rewritten is a normal turn now.

## Tests changed, said plainly

- `test_a_failing_builtin_is_written_down_not_repaired` (M7) is now
  `…_written_down_and_rewritten`. "Builtins are never rewritten" is what the
  owner chose to change.

## Dry run (text mode)

- **The bug:** a planted `hour // (now.minute - now.minute)` in
  `data/skills/overrides/clock.py`.
- **The conversation:**
  - "What time is it?" → "Something went wrong, sir."
  - In the background: Claude rewrote the clock and the gate passed,
    including `test_skills.py` and `test_dev_brain_fixes.py`'s clock tests in
    the sandbox.
  - → "I've rewritten 'clock', sir, after it failed. Say 'undo that' to go
    back."
  - "What time is it?" → "It's 20:11 on Saturday the 3rd of October, sir."
  - "Undo that." → "Shall I put back the previous 'clock', sir?" → "Yes." →
    "Undone, sir: 'clock' is back as it was."
- **Not published:** there is no token yet.
- **Threads:** 48 at startup, about 65 per turn, 35–51 after swaps; no growth.

## Needs the owner

- **`JARVIS_GITHUB_TOKEN`**: a fine-grained token, contents:write on
  `Nirvana77/JARVIS` only, in `.env` and the pod's `jarvis-env` Secret. Plus a
  ruleset protecting `develop` and `master`.

## Not done

- **Probation and "undo that" live in RAM:** a restart ends a probation and
  forgets what to undo. The override and its saved previous versions stay on
  disk.
- **Part B**, the redeploy that does not drop the watch.
