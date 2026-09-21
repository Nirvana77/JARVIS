# Milestone 2.5 — background skill learning: plan

**Status:** Implemented — see `PRD/milestone-2.5-background-learning-outcome.md`
**Date:** 2026-09-21
**Owner:** Kevin Lundell
**Builds on:** Milestone 2 (`PRD/milestone-2-skill-factory.md`,
`PRD/milestone-2-skill-factory-outcome.md`). This is an addition to M2, not a
new capability area. The knowledge base stays in Milestone 5, unchanged.

---

## Problem

The PRD says **retraining never blocks the main loop**. M2 met that for the
model swap, but not for the rest of learning. After `teach` / `edit_skill` /
`revert_skill` collects its answers, `handle()` stays inside the same turn for
the whole pipeline:

| Step (today, all inside the turn) | Where | Typical wait |
|---|---|---|
| Claude generates module + test | `flows._generate_validate_sandbox` → `build()` | 10–60 s |
| AST validate | same | < 0.1 s |
| Permission grant (voice) | same | user |
| Sandbox tests + dry-run | same | 1–10 s |
| Retrain, polled up to `fast_budget_s` | `Orchestrator._promote_and_retrain` | ~7 s |
| Self-check + "Shall I keep it?" (voice) | `_finish_retrain` | user |

During all of that JARVIS is `acting`. It does not listen, so the user cannot
ask for anything else until learning finishes or fails.

## Goal

Once the **dialog** is over (the questions only the user can answer), learning
runs **in the background**. The session carries on straight away: the user can
say *"search black holes"* while *"flip a coin"* is still being built. JARVIS
only interrupts to ask the questions it truly needs (a permission grant, "keep
it?"), and only at a **safe point** between turns, never in the middle of one.

---

## Key design decisions

### 1. Split every flow into *dialog* (foreground) and *job* (background)

- **Dialog** — unchanged voice questions: name, description, extra example
  (teach); which skill and what to change (edit); which skill (revert). It
  returns a `LearningRequest` (spec, `existing_source`, `allow_name`,
  `versioning`, plus the source to restore for revert). No Claude call and no
  sandbox.
- **Job** — `jarvis/factory/jobs.py`, `LearningJob.run()`: build → validate →
  *(permission decision)* → sandbox → retrain → self-check → *(keep
  decision)* → stage. It never calls `ask`/`say` directly. It gets two
  injected callables:
  - `notify(text)` — queues a line to speak at the next safe point.
  - `decide(prompt) -> bool` — queues a yes/no question and **awaits** its
    answer. The orchestrator asks it at a safe point and resolves the future.

  Unit tests can resolve `decide` directly, with no orchestrator.
- After the dialog, `handle()` says *"I'll work on that in the background,
  sir."*, starts the job and returns. The session's follow-up loop continues
  as normal.

### 2. Safe points: where background questions and announcements are spoken

A safe point is a moment when no turn is in flight **and** the user is
present:

1. After each turn in `_session()`, next to the existing `_merge_gate()`.
2. When the follow-up window lapses, **before** the spoken drop to standby.
   JARVIS asks its pending questions first, so a finished job isn't left
   waiting for the next wake word.

At a safe point the orchestrator: (a) runs the merge gate, (b) speaks queued
`notify` lines, (c) asks each queued `decide` question through the existing
`_ask`.

In **standby**, questions are never asked unprompted; they wait for the next
session's first safe point. Plain `notify` lines (for example *"I set 'timer'
aside, sir"*) are still spoken while idle in standby, as M2's drain task does
today.

### 3. Unclear answers don't count as "no"

`ask_yes_no` treats anything that isn't "yes" as "no". That is fine inside a
dialog, but a background question can land while the user's mind is
elsewhere. Background questions use a three-way `ask_yes_no_or_none`:

- **yes** / **no** resolve the future.
- **anything else** leaves the question pending and asks it again at the next
  safe point, at most 3 times. After that it resolves as "no", with a notify:
  *"I've set 'timer' aside, sir; you can teach it again any time."*

The unclear reply is **not** re-run as a command in 2.5 (see Open questions).

### 4. Permissions are still granted before any generated code runs

Sandbox tests execute generated code. In M2 the permission grant came before
the sandbox, so code with `net`/`shell`/`fs_*` never ran without consent.
That order is kept. If the validated manifest asks for more than
`{"pure","notify"}`, the job **pauses** on `decide("'timer' needs network
access. Allow it, sir?")` before sandboxing. A "no" ends the job and runs
nothing.

When there are no extra permissions, the job asks only one question at the
end.

### 5. One job at a time; later jobs see earlier ones

Two jobs that retrain side by side would each build their corpus without the
other's skill, and the second swap would erase the first. So:

- Jobs are serialized: one runs and the rest queue. (Implemented as a chain —
  each job waits on the one ahead of it — rather than an `asyncio.Lock`, so
  cancelling a queued job never cancels the one ahead.)
  Queueing a second one says *"I'll get to that after 'timer', sir."*
- The job's corpus is built from the **latest** registry: `self._staged`'s
  registry if one is staged but not yet merged, otherwise `self.registry`.
- The dialog refuses to start a job whose skill name is already queued or
  running (*"I'm already working on 'timer', sir."*). `edit`/`revert` of a
  skill with a job in flight is refused the same way.

### 6. The M2 fast/slow split goes away

Every job is now background, so the inline `fast_budget_s` wait, *"One moment,
sir… done"* and the slow path's **auto-confirm** (M2 decision 3) are all
removed. Every new or edited skill gets the explicit "Shall I keep it?" that
the PRD's stage table always asked for, now at a safe point. This drops a
documented M2 deviation instead of adding one. `revert_skill` keeps its "no
confirm" behavior. `[factory] fast_budget_s` is removed from `config.toml` /
`Config`.

### 7. Merge and announce are unchanged

After "yes": `_promote_files` → `self._staged` → the existing `_merge_gate()`
swaps at the same safe point. It then says *"I've learned 'timer', sir. My
capabilities are updated."* So "try me" is true on the very next turn.

### 8. Shutdown and cancellation

- `run()`'s `finally` cancels the running job and any queued ones. It deletes
  their `staging/` files and any trained model version that was never staged.
  A pending `decide` future is cancelled, not answered.
- A crash inside a job is logged, and the job notifies *"Something went wrong
  while I was learning 'timer', sir."* The loop never dies because of a job.
- The Enter/barge-in `Interrupter` only cancels the foreground turn. It never
  touches background jobs.

---

## Changes by file

| File | Change |
|---|---|
| `jarvis/factory/flows.py` | Flows return `LearningRequest` after the dialog. `_generate_validate_sandbox` moves to `jobs.py`. Add `ask_yes_no_or_none`. |
| `jarvis/factory/jobs.py` (new) | `LearningRequest`, `LearningJob` (build → validate → decide perms → sandbox), with `notify`/`decide` injected. |
| `jarvis/core/orchestrator.py` | `_start_learning()`, job lock and queue, `_pending_notices`, `_pending_decisions`, `_safe_point()` called after each turn and before standby. `_promote_and_retrain` runs inside the job (no fast budget). The drain task only speaks notices while in standby. Cancel on shutdown. |
| `jarvis/config.py`, `config.toml` | Remove `fast_budget_s`. |
| `PRD/jarvis-2026-rebuild.md` | Add the Milestone 2.5 section. Replace the "UX split" bullet under *Concurrency & seamless hot-swap*. |

## Verification (tests written first)

Dependency-injected fakes as in `tests/test_orchestrator.py` /
`tests/test_flows.py`. The fake `generate` blocks on a `threading.Event` so
"still learning" is deterministic.

1. **Control returns after the dialog.** `handle("teach")` returns while
   `generate` is still blocked. Then `handle("search", "search black holes")`
   dispatches to `search` straight away, with the job still running.
2. **Keep question at a safe point only.** Release `generate`. The keep
   question is **not** asked during the in-flight turn. It is asked at the
   next `_safe_point()`. "yes" → merged, and the new skill dispatches on the
   following turn.
3. **Decline discards.** "no" → the `staging/` file and the unused
   `data/models/nlu/v<N>/` are both removed. `self.nlu` / `self.registry` are
   unchanged.
4. **Permissions before sandbox.** A manifest with `net` → the sandbox fake
   is **not** called until the grant resolves "yes". "no" → never called.
5. **Unclear answer stays pending.** "what time is it" → the question is still
   queued and re-asked at the next safe point. After 3 unclear replies it
   resolves as "no" with a notice.
6. **Before standby.** A job finishes while the follow-up window is open and
   the user stays silent → the keep question is asked before the standby line.
7. **Serialized jobs.** Two teaches back to back → the second waits. The final
   registry and model contain **both** skills.
8. **Duplicate refused.** A teach of a name already queued or running is
   refused in the dialog, and no second job starts.
9. **Failure is announced, not spoken mid-turn.** A `BuildError` in the job →
   a notice at the next safe point. The loop keeps running.
10. **Shutdown.** `stop()` with a job blocked in `generate` → the job is
    cancelled and staging is cleaned up. `run()` returns.
11. **Revert** runs as a background job with no keep question.
12. Existing M2 tests: fast/slow-split tests are rewritten to match decision
    6 (said explicitly per CLAUDE.md, since those assertions described the
    old spec). All other M2 tests pass unchanged.

**Dry run** (`python -m jarvis text`, fake Claude with an artificial delay at
the network boundary, as in M2's outcome): teach "flip a coin" → while it
builds, "search black holes" answers → at the next pause, "Shall I keep it?"
→ yes → "flip a coin" works on the next line. Repeat with a "no", and with a
permission-requesting skill.

## Open questions

- **Re-run an unclear reply as a command?** If the user answers "keep it?" with
  "search black holes", 2.5 drops that reply and asks again later. Classifying
  it and dispatching it if it's a confident, non-meta intent would feel more
  natural, but it adds a second path into `handle()`. Proposed: leave it for
  M4, which already reasons about unclear input.
- **Cap on queued jobs?** Proposed: no cap. Jobs are rare and user-initiated.
