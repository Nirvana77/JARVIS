# Milestone 8: JARVIS rewrites its own skills and redeploys without dropping the watch

**Asked for:** 2026-10-03, by the owner: *"This is what I want JARVIS to do.
Rewrite its own code and redeploy without breaking the connection to the
clients."* The three choices put to them:

| Question | Answer |
|---|---|
| How much of its own code may JARVIS rewrite? | **Skills + builtins.** The core (understanding, learning, connections) stays human-reviewed, so a bad self-edit cannot break the loop that would repair it. |
| When a self-change passes all checks? | **Deploy, tell me after.** It goes live, rolls itself back if it misbehaves, and JARVIS says what it changed. "Undo that" reverts it. |
| Where do its commits go? | **Its own branch, pushed:** `jarvis/self` on GitHub. `develop` stays the owner's to merge. |

## Where it stands (2026-10-03)

- **Learned skills** are already rewritten by JARVIS (M7: build, repair) and
  swapped in live at an idle point. No restart, no disconnect.
- **Builtins** are repo code. When one fails, the failure is written to
  `data/learning/builtin-failures.jsonl` for a person. JARVIS never touches
  them.
- **Every deploy of the core drops the watch.** The builder polls `develop`
  every 120 s, builds `jarvis:prod`, and rolls `jarvis-brain` with strategy
  `Recreate`: the old pod stops before the new one has loaded its models (and
  maybe retrained). The watch reconnects with backoff, after a minute or two.
  The dev brain coming or going also cuts sessions (HAProxy
  `shutdown-backup-sessions` / `shutdown-sessions`).

## Part A: JARVIS rewrites builtins, live

1. **Overrides, not edits in place.** JARVIS's version of a builtin lives in
   `data/skills/overrides/<name>.py`, next to the learned skills that both
   brains share. Discovery loads an override in place of the packaged builtin
   (`origin = "builtin"`, `overridden = True`), so it hot-swaps through the
   same staged registry and merge gate as a learned skill: no restart. Deleting
   the override restores the packaged one.
2. **What starts a rewrite:**
   - A builtin raises: the M7 repair job, now for builtins as well. It has
     the same daily cap and switches the skill off after repeated failures
     (back to the packaged version, not off).
   - A confusion corrected three times (`state.confusions`): the examples of
     the two skills are rewritten so they stop overlapping.
   - The owner says "change how <skill> works": the edit flow, now for
     builtins too.
3. **The gate, every time, in the sandbox:**
   - AST validation;
   - the generated tests;
   - the **repo's own tests for that builtin**, run against the override;
   - the call that failed, replayed;
   - **the builtin's recent `ok` calls from the interaction log**, replayed:
     each must still return a line and not raise.

   One failure and the change is not kept. The M6 repair loop, a continued
   conversation with Claude, is built here.
4. **After the swap:**
   - **A probation:** the first `probation_calls` (5) uses are watched, and an
     exception or a correction ("no, I meant …") reverts to the previous
     version on its own and says so.
   - **Telling the owner:** "I've rewritten 'search', sir: it now … . Say
     'undo that' to go back." "What did you change today?" lists every
     change.
5. **Git.** Each kept change is committed to a `jarvis/self` branch as an edit
   of `jarvis/skills/builtin/<name>.py` (with its test changes), author
   "JARVIS (on behalf of Kevin Lundell)", and pushed.
   - **Credential:** a fine-grained token with contents:write on
     `Nirvana77/JARVIS` only, in the `jarvis-env` Secret and the dev `.env`.
   - **Guard:** a repository ruleset that protects `develop` and `master`
     from that token, so it can only write `jarvis/self`.
   - **Getting it into the image:** the owner merges `jarvis/self` into
     `develop` when they like. Until then the override in `data/` is what
     runs.

## Part B: a deploy that does not drop the watch

1. **Gateway and brain, two processes in one pod.**
   - The **gateway** (`python -m jarvis gateway`) holds the edges' WebSockets,
     auth, pairing, the audio intake and speech out (the parts of
     `remote/server.py` that face the edge). It changes rarely.
   - The **brain** (orchestrator, NLU, skills, learning, factory) runs as a
     child. It talks to the gateway over a Unix socket, through the same four
     roles `RemoteLink` already gives it (wake / mic / stt / tts), proxied.
2. **The swap:**
   - The gateway starts a new brain from new code, which loads its models and
     retrains if it must, and says "ready".
   - At the next idle moment (no turn in flight, no dialog waiting for an
     answer), the gateway routes the next utterance to the new brain and stops
     the old one.
   - The conversation memory moves with it: the recent turns are in RAM, so
     they are handed over; the facts are on disk already.
   - A background learning job in the old brain is let finish or is
     cancelled and re-queued.
3. **Code without a new image.** The pod keeps a checkout of `develop` on a
   volume.
   - The builder, instead of rolling the deployment, tells the gateway
     "update to <sha>" through an admin endpoint on the pod network.
   - The gateway fetches the commit, starts the new brain from it, and swaps.
   - A full image rebuild and pod roll happen only when `requirements.lock`
     or the gateway's own code changed. That is the one case where the watch
     still reconnects, and the deployment moves to `RollingUpdate` with
     `maxSurge: 1` so the new pod is ready first.
4. **The dev brain** joins the same way: it can run as a second brain behind
   the pod's gateway instead of taking the watch over through HAProxy.
   Whether to keep HAProxy at all is decided in this part.

## Order

**A first**: it is what the owner asked for in substance, and it needs no new
process. **Then B.** Each is tests first, the full suite, a dry run and a
thread report, as CLAUDE.md asks.

## Verification

- **A.**
  - A builtin that raises is rewritten and swapped, with no restart and the
    socket untouched.
  - A rewrite that fails the repo's tests or the replay is not kept.
  - A kept rewrite that raises during probation reverts by itself.
  - "Undo that" reverts.
  - A commit lands on `jarvis/self`, never on `develop`.
  - Overrides are discovered by both brains.
- **B.**
  - A brain swap during a live edge connection loses no utterance and does
    not close the socket (a test edge stays connected across the swap).
  - A swap requested mid-turn waits for idle.
  - A new brain that fails to start leaves the old one serving.
  - Threads and processes go back to baseline after each swap.
  - Live: push a trivial change to `develop` while talking to the watch;
    the conversation continues.

## Needs the owner

- **A GitHub fine-grained token** for `jarvis/self`, and the ruleset on
  `develop`/`master`.
- **Permission to change** the `jarvis-brain` deployment (volume, strategy)
  and the builder's last step, in Part B.
