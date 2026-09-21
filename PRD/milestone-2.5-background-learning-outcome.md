# Milestone 2.5 — background skill learning: outcome

**Date:** 2026-09-21
**Branch:** `milestone-2.5-background-learning` (off `develop`, not yet merged or committed)
**Result:** ✅ After the teach/edit/revert dialog, learning runs as a
background job and the session keeps serving commands. The job's questions
(permission grant, "Shall I keep it?") and notices are spoken only at safe
points. This was verified by tests and by conversing with the real program in
text mode, with a deliberately slow fake Claude at the network boundary.

Plan and design decisions: `PRD/milestone-2.5-background-learning.md`.

---

## What shipped

| Area | Module(s) | Notes |
|---|---|---|
| Dialogs | `jarvis/factory/flows.py` | `TeachFlow`/`EditSkillFlow`/`RevertSkillFlow` now end after the questions and return a `LearningRequest` (or `None`). They take `busy_names`, so a skill already being learned can't be taught, edited or reverted again. New `ask_yes_no_or_none` uses whole-word matching, so "know"/"now" aren't read as "no". |
| Background job | `jarvis/factory/jobs.py` (new) | `LearningJob`: build → validate → *permission decision* → sandbox. It only calls the injected `notify`/`decide`. `run_detached` runs Claude and the sandbox on daemon threads, so a cancelled job never makes shutdown wait for them. |
| Orchestrator | `jarvis/core/orchestrator.py` | `_start_learning`/`_learn` (serialized job chain), `_retrain_and_stage` (retrains on the *latest* registry, self-check, keep decision, promote, stage), `_decide`/`_notify` queues, `_safe_point()` (after every turn and before the drop to standby), `_idle_tick` (standby: notices and merges only, never questions), `cancel_learning()` on shutdown. The retrain is injectable (`train_and_load=`). |
| Retrain worker | `jarvis/nlu/retrain_worker.py` | `exited()`, and `terminate()` so a cancelled job stops its retrain process. |
| Config | `jarvis/config.py`, `config.toml` | `[factory] fast_budget_s` removed (decision 6). |
| Tests | `tests/test_background_learning.py`, `tests/test_jobs.py` (new); `tests/test_flows.py` (rewritten); `tests/conftest.py` (thread-leak guard) | 159 passed, 1 skipped (was 113 after M2). |

## Test changes (said explicitly, per CLAUDE.md)

`tests/test_flows.py` was **rewritten, not loosened**. Its old assertions
described the M2 spec, in which a flow built, validated and sandboxed inside
the dialog and returned a `FlowOutcome`. M2.5 changes that spec on purpose
(decision 1), so those checks moved to `tests/test_jobs.py` (build, validate,
permission, sandbox) and `tests/test_background_learning.py` (orchestrator),
where they now run against the background job. Every M2 check still exists.

There were no fast/slow-split orchestrator tests to rewrite. M2 had exercised
that path only through the merge-gate tests, which pass unchanged.

`tests/conftest.py` gained an autouse `no_leaked_threads` fixture (CLAUDE.md
step 5). Any test that leaves a thread running now fails. I checked that it
catches a deliberately leaked thread.

## Verification

Automated:
```
python -m pytest     # 159 passed, 1 skipped
```
The 13 tests in `test_background_learning.py` map to the plan's Verification
items 1–11:
- control returns after the dialog while `generate` is blocked;
- questions are asked only at safe points, including before standby;
- a "no" to "keep it?" discards the staging file *and* the unused model version;
- the permission question comes before the sandbox, and a "no" means the sandbox never runs;
- an unclear answer stays pending, and 3 unclear answers count as "no";
- two teaches in a row run one after the other and keep both skills;
- a duplicate teach is refused;
- a failure is announced at a safe point;
- in standby, notices are spoken but no questions are asked;
- shutdown cancels running jobs cleanly, including one waiting on a question;
- revert runs in the background with no keep question.

**Thread monitoring (CLAUDE.md step 5):**
- **Tests:** after every one of the 160 tests, the thread count was back to
  1 (the main thread), with no leftovers. This is now enforced by the conftest guard.
- **Dry run:** I sampled `ps -T` on the live process every 0.5 s through
  searches, two learning jobs (one kept, one declined) and shutdown.
  - Steady state is **15 threads**. The count went back to exactly 15 after
    every turn and after each job ("yes" at 44.8 s, "no" at 82.5 s).
  - It peaked at 16–19 while a retrain was being polled or a search was
    running.
  - It never climbed from one job to the next, and it dropped to 11 at shutdown.
  - There are 2 child processes (the multiprocessing forkserver and the
    resource tracker) and no stray ones.

**End-to-end dry run.** The real text-mode pipeline was used: real NLU,
registry, persona, sandbox and worker-process retrain. Only
`ClaudeClient.generate_skill` was replaced, by a fake that sleeps 8–20 s, the
same boundary M2's outcome used. The driver script was kept in the
scratchpad, not the repo.
- *"learn how to flip a coin"* → dialog → *"I'll work on that in the
  background, sir."* → *"search black holes"* answered at **t=1.2 s**, while
  Claude was still generating.
- The job finished during a silent stretch. Its keep question was asked at
  the next safe point, right after *"thanks"* was answered, and not before.
  → "yes" → *"I've learned 'coin_flip'"* → *"flip a coin"* → *"Heads, sir."*
  on the very next line.
- A skill asking for `net` → *"'weather_check' needs net access. Allow it,
  sir?"*, asked before anything ran in the sandbox → "no" → *"Very well, I
  won't build 'weather_check'."*
- A second teach while one was running → *"I'll get to that after
  'dice_roll', sir."*
- *"shut down"* with a 20 s Claude call in flight: the process exited as soon
  as stdin closed (6 s). It did not wait for Claude, and no staging files were
  left behind.
- All dry-run state was deleted afterwards: the learned `coin_flip.py` and the
  model versions the run trained.

## Found along the way

- **Driver pitfall, not a JARVIS bug.** Python 3.14's default
  `multiprocessing` start method is `forkserver`. It re-imports the main
  script in the retrain worker. My first dry-run driver had no
  `if __name__ == "__main__"` guard, so the worker started a second JARVIS
  reading the same stdin (a duplicate banner and interleaved output).
  `python -m jarvis` already has the guard. Any future driver script needs
  one too.
- **Model-version pruning happens at training time.** This is pre-existing
  M2 behaviour and was left unchanged. `jarvis.nlu.train.train()` keeps the 3
  newest versions *when it trains*, before the user has decided whether to
  keep the skill. So a declined or cancelled job can still push out the
  oldest good version. In the dry runs this removed the old
  `data/models/nlu/v1`. `data/` is gitignored and rebuildable, and v2 (the
  model that loads) was untouched. Now that every teach and edit asks "keep
  it?", declines will happen more often. It's worth pruning only on promote,
  as a follow-up.
- **The sandbox had no network isolation in this environment.**
  `unshare --user --map-root-user --net` fails here with
  `write /proc/self/uid_map: Operation not permitted`, and
  `SubprocessSandbox` fell back to the AST ban as designed. M2 had `unshare`
  working, so this is an environment change, not something this milestone
  caused.

## Not verified here

- A live Claude call. The `ANTHROPIC_WORKSPACE_ID` gap from M2 is still open.
- The mic, wake-word and STT path. In voice mode the "before standby" safe
  point fires when the follow-up window runs out. Text mode has no timeout, so
  there the keep question comes after the user's next line. The behaviour
  itself is unit-tested (`test_pending_question_is_asked_before_standby`).

## Open questions (carried from the plan)

- Should an unclear reply to a background question be re-run as a command?
  Currently it is dropped and the question asked again later. Suggested for M4.
- Should there be a cap on queued jobs? None for now.
