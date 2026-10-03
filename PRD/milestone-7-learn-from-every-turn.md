# Milestone 7: learn from every turn

**Asked for:** 2026-10-03, by the owner, right after the steady-ship merge:
*"The goal is to have JARVIS running and every interaction will add to his
learning base. Creating an ever growing LLM that can understand you and if it
does not understand it will learn and change its code to understand it
self."* The three choices put to them:

| Question | Answer |
|---|---|
| How much may JARVIS change itself without asking? | **Fully autonomous.** It learns phrasings, writes new skills and repairs broken ones on its own. |
| The "ever-growing LLM" | **Grow around the LLM.** The local model stays fixed. What grows is what it reads and what routes to it: the classifier, the skills, memory and the notes. A fine-tune is revisited once there are months of logged turns. |
| Ollama | **Use the one on the cluster** (`ollama` namespace, GPU node `kevin-ai`). |

## Where JARVIS stands today (2026-10-03)

JARVIS learns only when it is told to: "learn how to …" starts the teach
dialog. Everything else it hears is forgotten.

- **No turn is kept.** Recent turns live in RAM (`core/memory.py`) and expire.
  There is no interaction log.
- **The corpus has two sources**: the `intents.json` seeds and each skill's
  `MANIFEST.examples` (`nlu/corpus.py`). There is no slot for phrasings learned
  from use.
- **A confirmed guess teaches nothing.** When "Did you mean 'flip a coin'?" gets
  a yes, the guess is run. The words that were actually heard are not saved,
  so the same mishearing is guessed again next time.
- **Not understanding ends with "I didn't catch that".** JARVIS never offers to
  learn. `TeachFlow` can take a `seed_description`, but the orchestrator never
  passes it one.
- **No corrections.** "No, I meant …" is classified as a new turn.
- **A skill that raises** makes JARVIS say "Something went wrong, sir." The
  traceback goes to the log and nothing else happens. The factory's retry loop
  covers only building a skill, not a skill that breaks later.
- **The reasoner has never been on, on either brain.** Both `config.toml`s
  point at `http://localhost:11434`, where nothing listens. The pod logs
  `reasoner tier: none`. So M4's mishearing guess and M4.5's planning and
  answering have never run outside the tests.

## The loop this milestone builds

```
heard ─► classify ─┬─ sure ──────────────► run ─► log
                   ├─ unsure ─► confirm ─► run ─► log + learn the phrasing   (C)
                   └─ unknown ─► reasoner ┬ maps to a skill ► confirm ► run ► learn  (C)
                                          ├ an answer ► say it ► log
                                          └ a new capability ► build the skill (E)
skill raises ─► log ─► repair the skill in the background (F)
"no, I meant …" ─► undo what was learned, learn the right label (D)
```

Every arrow ends in the interaction log (B), which feeds the corpus.

## Parts, in build order

### A. The reasoner online

- **Address.** `[reasoner] base_url` becomes
  `http://ollama.ollama.svc.cluster.local:11434` in the pod's `jarvis-config`
  ConfigMap. The dev brain runs on the same node and uses the ClusterIP
  (`http://10.43.254.194:11434`; the service DNS name does not resolve from the
  host).
- **Model.** `qwen2.5:3b` is not pulled there. Use `qwen3:8b`, which is
  already pulled.
- **Code: turn qwen3's thinking off.** qwen3 thinks before it answers unless
  told not to. `Reasoner` (`core/reasoner.py`) sends `"think": false` when
  `[reasoner] think = false` (the new default). Otherwise every guess pays for
  hidden reasoning tokens and blows `guess_timeout_s`.
- **Shared GPU.** The cluster's Ollama keeps one model loaded at a time
  (`OLLAMA_MAX_LOADED_MODELS=1`). When another user has `qwen3-coder:30b` or
  `gpt-oss:20b` loaded, the first JARVIS call waits for a model swap, which can
  outrun the 8 s guess timeout. That is acceptable: the existing timeout
  already falls back to the plain line. The swaps are logged so their cost is
  visible. `keep_alive` stays at the server's 30 minutes.
- **Then do the manual checks M4 and M4.5 never had.** These are the misheard
  command, a plan ("wake me in five and beep the watch") and an answer, on the
  dev brain, with timings.

### B. The interaction log

- **What is written.** One JSON line per turn to
  `data/interactions/<device>/<YYYY-MM-DD>.jsonl`, with these fields:
  - `ts`, `device`, `heard`
  - `label` and `confidence` (with the runner-up)
  - `path`: one of `direct | compound | confirmed | mishear | plan | answer | unknown | teach | correction`
  - `skill`, `params`
  - `outcome`: one of `ok | error | declined | cancelled`
  - `said`
  - the model version
- **Where it is written.** An append-only writer in `core/interactions.py`.
  It is injected like memory (a fake in tests) and called from the one place
  a turn ends.
- **Retention.** Kept for `[learning] keep_days` (default 365). Rotated at
  startup and on the idle tick.
- **Forgetting.** "Forget everything I told you" also deletes that device's
  log.
- **Privacy.** The log is local, like memory. Audio is never logged, as
  before. The edge transcribes everything it hears, but only turns that
  reached the orchestrator, meaning they passed addressing, are logged.

### C. Learning phrasings from confirmed turns

- **A third corpus source**, `learned`, kept in
  `data/nlu/learned_examples.json`: (text, label, why, ts, from-log-line).
  `build_corpus` reads it, so `corpus_digest` (#15) changes and a restart
  retrains as well.
- **What becomes an example:**
  1. **A yes to "Did you mean …?"** The text that was *heard* becomes an
     example of the label that ran, e.g. "flip a corn" → `flip_a_coin`.
  2. **A plan with one step, confirmed.** The heard text becomes an example
     of that step's label. A plan with several steps teaches nothing: its
     phrase belongs to no single label.
  3. **An unsure turn that went unchallenged.** The confidence fell between
     the unknown cutoff and the accept threshold, the skill ran `ok`, and no
     correction (D) came in the follow-up window.
- **A new "unsure" confirm that needs no reasoner.** When the classifier's
  best label is under the accept threshold but within `[learning]
  confirm_margin` of it, JARVIS asks "Did you mean <that skill's first
  example>?". It is the same question as M4's, but it works with no LLM, and
  a yes feeds (1).
- **What is never learned:**
  - **Phrasings for the guarded meta-actions** (`teach`, `edit_skill`,
    `revert_skill`, `remove_skill`, `forget_memory`). A learned phrasing must
    never make a destructive action easier to trigger.
  - **Duplicates.**
  - **A phrasing that already classifies correctly** with margin, since it
    adds nothing.
  - **More than `[learning] max_per_label` (50) per label.** The oldest
    learned example goes first; seeds and manifest examples are never
    dropped.
- **Retraining is batched.** It starts when `[learning] retrain_after` (5)
  new examples have collected, or at the first idle tick after
  `retrain_idle_s`. It uses the existing worker, `_self_check` and idle-only
  swap, so a turn is never blocked.
- **The self-check gains a regression gate.** Before the swap, every seed
  probe and the last N `ok` turns from the log are reclassified. If the new
  model gets more of them wrong than the old one, it is discarded and the
  newest examples are quarantined. Learning cannot quietly make JARVIS
  worse.
- **Announcing.** With `[learning] announce = "idle"` (the default),
  JARVIS says once at the next idle point what it learned, e.g. "I've learned
  that 'flip a corn' means flip a coin, sir." `"never"` keeps it in the log
  only.

### D. Corrections

- **A seed intent, `correction`.** Examples: "no, I meant …", "that's not what
  I asked", "wrong one", "not that". It is only active inside the follow-up
  window after a skill ran.
- **What it does:**
  - It marks the previous log line `corrected`.
  - If that turn was learned (C), it removes the example.
  - If the correction names the right thing ("no, I meant set a timer"),
    JARVIS classifies that part, runs it, and learns the *original* heard
    text as an example of the right label.
- **Repeated mistakes are recorded.** A label pair that is corrected three
  times goes into `data/learning/confusions.json`, so the knowledge of what
  gets mixed up survives a retrain.

### E. Learning a new skill when nothing fits

- **The signal.** After M4/M4.5's path, the reasoner classifies the turn as
  one of:
  - `answer`: it answered.
  - `chat`: small talk, nothing to do.
  - `noise`: a fragment or misheard background.
  - `capability`: a request for something JARVIS cannot do yet.

  This is one more field in the existing `_think` reply. With no reasoner
  there is no signal, and nothing is built from unclassified noise.
- **Autonomous build** (the owner's choice). On `capability`:
  - JARVIS says "I can't do that yet, sir. I'll learn it."
  - It starts the background learning job (M2.5) with no dialog. The reasoner
    drafts the spec (description, 3–5 example phrasings, params) from the
    utterance plus the recent turns. That fills `seed_description` and the
    examples.
  - Claude builds the skill. It is validated and sandboxed exactly as today.
  - It is **kept without "Shall I keep it?"**, retrained and swapped in at
    idle, then announced: "I've learned to …, sir. Try it."
  - The original utterance is one of its examples, so saying it again now
    works.
- **Guardrails stay on even in autonomous mode:**
  - **No permission is granted without a yes.** A spec that needs network,
    files or edge tools stops and asks, as the teach flow does today. This is
    the one place the owner's "fully autonomous" is not taken literally.
    Granting network access to code nobody read is a decision for a person,
    and `[learning] auto_permissions = true` exists for the owner to make it.
  - **A spend cap.** At most `[learning] max_builds_per_day` (5)
    autonomous builds. Past that, the request is logged and JARVIS says it
    will learn it tomorrow.
  - **Deduplication.** A request similar (by embedding) to one already being
    built, or already failed today, is not built again.
  - **Undo is one sentence.** "Forget how to …" (remove) and "go back to the
    previous version" (revert) already exist.

### F. Repairing a skill that breaks

- **The trigger.** A learned skill raises, or returns a non-string, at run
  time. JARVIS says the `error` line and logs the traceback, the utterance and
  the params. In the background it then starts a **repair job** for that
  skill.
- **The repair job is M6's repair loop applied to a running skill.** Claude
  gets:
  - the current source and manifest;
  - the failing call, as data: utterance, params and traceback, truncated;
  - the previous failures, as a continued conversation.

  The repaired module passes the full gate: AST validation, generated tests,
  and the dry run. It must also pass the **failing call itself**, replayed in
  the sandbox, so a repair that does not fix the reported failure is not
  kept.
- **Kept, swapped and announced.** A kept repair is swapped in at idle and
  announced: "I've repaired …, sir." The old version stays revertible.
- **A circuit breaker.** At most `[learning] max_repairs_per_skill_per_day`
  (2). A skill that keeps failing after that is disabled until the owner says
  otherwise, and JARVIS says so instead of failing silently.
- **Builtins are not rewritten.** They are repo code under git, and an
  unreviewed edit there would be overwritten by the next deploy anyway. A
  builtin failure is logged to `data/learning/builtin-failures.jsonl` for a
  person to fix.

### G. Seeing what it learned

- **"What have you learned today?"** An inline answer from the log: phrasings,
  new skills, repairs and corrections.
- **`python -m jarvis learning log [--since 1d]`.** Prints the same.
- **`python -m jarvis learning undo <id>`.** Removes a learned example.

## The two brains and the one `data/`

The dev brain and the pod share `data/` (the hostPath in `jarvis-brain`).
Both learning at once would cause three problems:

- They would race on `learned_examples.json`.
- They would both retrain into `data/models/nlu/`.
- They would build skills into **different** directories. The pod keeps
  learned skills on the `jarvis-state` PVC, the dev brain in its checkout.

This milestone therefore adds `[learning] enabled`:

- **The pod learns.** It is the brain the watch talks to.
- **The dev brain does not, by default.** It still logs, to its own device
  names, and still loads whatever the pod trained.

Writes are atomic (temp file + rename) and guarded by a lock file, so a manual
`nlu rebuild` on the host cannot interleave with the pod. Unifying the two
learned-skill directories is out of scope. It goes into `known-issues.md`.

## Verification

Tests first, with injected fakes as in `tests/` (fake reasoner, fake Claude
client, fake sandbox, tmp `data_dir`).

- **A.** The `think` flag reaches the request body. An unavailable reasoner
  still means today's behaviour.
- **B.** One log line per turn on every path, with the right `path` and
  `outcome`. Forget deletes it. Rotation drops old days.
- **C.**
  - A confirmed "did you mean" adds the *heard* text.
  - A guarded meta-action is never added.
  - The cap, deduplication and the batch trigger work.
  - A model that loses accuracy on the regression gate is discarded and its
    examples are quarantined.
  - After a retrain, the misheard phrase classifies directly, with no
    confirm.
- **D.**
  - A correction inside the window undoes a learned example and learns the
    right one.
  - Outside the window, "no" is just "no".
- **E.**
  - `capability` builds and keeps the skill with no dialog.
  - A spec that needs a permission still asks.
  - The daily cap and deduplication work.
  - `noise`/`chat` build nothing.
  - With no reasoner, nothing is built.
- **F.**
  - A raising learned skill starts one repair.
  - A repair that does not pass the replayed call is rejected.
  - The breaker disables the skill after two failed repairs.
  - A raising builtin is logged and not repaired.
- **Thread counts.** Repair and build jobs reuse the M2.5 worker. The count
  returns to baseline after each one, in tests and in the dry run.
- **Dry run** (`python -m jarvis text`, copied `data/`, the cluster's
  Ollama):
  1. Say a misheard command, then confirm it.
  2. Say it again: it should now run with no confirm, after a retrain
     triggered by setting `retrain_after = 1` for the run.
  3. Ask for something new ("tell me a random number between 1 and 6"): it
     should build the skill, keep it, and the same sentence should then work.
  4. Break a learned skill's file by hand: it should repair it.
  5. "No, I meant …": the correction should undo what was learned and learn
     the right label.

## Out of scope

- **Fine-tuning the LLM.** It is revisited when the log holds enough
  confirmed turns to be worth it. The log's format is chosen so it can be
  exported as training pairs later.
- **#3, multi-device sessions.**
- **Unifying the pod's and the dev brain's learned-skill directories.**
