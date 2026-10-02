# Milestone 4.5 — reasoning and per-device memory: outcome

**Date:** 2026-10-01
**Branch:** `develop` working tree, on top of M4 (not yet committed)
**Result:** ✅ A sentence that is several commands runs as several, with no
model. A turn the NLU cannot place is, with Ollama up, re-said as commands
(confirmed, then run) or answered in the persona's voice. Every device has its
own memory — the recent conversation and what it was asked to remember — and
the reasoner is given both. Verified by tests and by conversing with the real
program in text mode with a **stand-in** for Ollama: this machine has none, so
what a real model makes of the prompt is the part still to be checked.

Plan and design decisions: `PRD/milestone-4.5-reasoning-and-memory.md`.

---

## What shipped

| Area | Module(s) | Notes |
|---|---|---|
| Memory | `jarvis/core/memory.py` (new) | `Memory` (one per brain) and `DeviceMemory` (a view for one device). Turns in RAM, capped and expiring; facts in `data/memory/<device>.json`, written atomically, de-duplicated, capped. `focus(device)` is the device whose turn it is. A device id that could not be a filename is refused. |
| Reasoning | `jarvis/core/reasoning.py` (new) | Pure: the system prompt (persona character + task), `build_prompt()` (commands, facts, turns, the time, the transcript), `parse()` → `Thought(answer | commands)`, `clip()`. |
| Compound | `jarvis/nlu/compound.py` (new) | `split()` and the two bars a clause must clear. |
| Orchestrator | `jarvis/core/orchestrator.py` | `handle()` now tries `_compound()` first, then `_handle_one()` (the old body). `_unclear()` goes M4 → `_think()`. `_run_steps()`, `_chainable()`. `_turn()` wraps a session's turn: sets the device, records what was heard and said, and absorbs a cancel. `_recall()` / `_forget()`. New `memory=` argument. |
| Reasoner | `jarvis/core/reasoner.py` | `generate(format=…)` — Ollama's JSON constraint. |
| Persona | `jarvis/core/persona.py`, `personas/*/responses.toml` | `character()`; lines `nothing_remembered`, `forget_confirm`, `forgotten`, `forget_kept`. |
| Skills | `jarvis/core/context.py`, `jarvis/skills/registry.py`, `jarvis/skills/builtin/note.py` | `ctx.memory`; the registry hands each skill the view for the turn's device; `note` files its text there as well as in `notes.txt`. |
| Seed intents | `intents.json` | `recall_memory`, `forget_memory`. The NLU retrains on the next start. |
| Remote | `jarvis/remote/server.py` | `RemoteLink.device_id`. |
| Wiring, config | `jarvis/app.py`, `jarvis/config.py`, `config.example.toml`, `config.toml` | One `Memory` per brain, shared by orchestrator and registry. `[nlu] compound`, `[reasoner] answer_questions` / `plan_commands` / `reason_timeout_s`, `[memory]`. |
| Docs | `README.md`, `CLAUDE.md` | "When it isn't sure what you said"; the new threads in step 5. |
| Tests | `tests/test_memory.py`, `tests/test_reasoning.py` (new) | 101 tests. |

## Test changes (said explicitly, per CLAUDE.md)

- `tests/test_config_example.py`: `"memory": "memory"` added to its table of
  config sections. The test exists to make that list complete; a new section
  has to be in it.
- `tests/test_misheard.py` (M4, written earlier the same day, uncommitted):
  `test_it_can_be_switched_off_with_ollama_still_running` asserted the
  reasoner was *never called* with `correct_misheard = false`. The reasoner
  now has a second job, so it asserts what it meant: no mishearing guess was
  asked for. Its other tests are unchanged and pass (it gained four for the review's first finding: 42 now).
- One thing was **not** done because a test said otherwise: the plan was to
  move "remember that …" to a new `remember` intent, but
  `tests/test_nlu.py` pins it to `note`. So `note` stayed the way a fact gets
  in, and the skill gained `ctx.memory` instead (decision 7).

## Verification

Automated, in a clean environment (`config.example.toml`, an empty data
directory — see "Shared data" below for why not in place):
```
python -m pytest     # 767 passed, 3 skipped, 0 failed
```
That is also the first run in which known issues #4 and #6 do not fire, which
confirms they are this machine's local state and nothing else.

`tests/test_reasoning.py` and `tests/test_memory.py` cover the plan's
Verification list item by item; beyond it, a loopback test over a real socket
shows a turn from an edge recorded under that edge's id and not under `local`.

**End-to-end dry run** (`python -m jarvis text`; real NLU trained with the new
intents, real registry with this machine's 20 skills, real persona, real
memory on disk; `open`/`play`/`search` avoided because they open a browser or
go to Wikipedia).

Without Ollama:
```
you> set a timer for five minutes and find my watch      intent unknown (0.32)
  steps   : "set a timer for five minutes" -> set_timer (0.90) · "find my watch" -> find_watch (0.85)
Jarvis: The watch isn't connected to me.                  <- both ran; no edge in text mode
Jarvis: The watch isn't connected to me.
you> remember that I parked on level two
Jarvis: Noted.
you> where did i park
Jarvis: I have no skill for that yet, sir.                <- no model: the plain line, as before
you> what do you remember
Jarvis: You asked me to remember: I parked on level two.
you> forget everything i told you
Jarvis: Forget everything you have asked me to remember, sir?
you> yes
Jarvis: Forgotten, sir.
you> what do you remember
Jarvis: You have not asked me to remember anything, sir.
```

With the stand-in on `localhost:11434` (its answers are canned, but it can
only give the right one by reading the memory out of the prompt):
```
you> where did i park
Jarvis: You told me: I parked on level two, sir.          <- from the fact
you> how many days are in a leap year
Jarvis: Three hundred and sixty-six, sir.
you> and a normal year
Jarvis: Three hundred and sixty-five, sir.                <- from the turn before
you> wake me in five and beep the watch
  plan    : "set a timer for 5 minutes" -> set_timer (0.92) · "find my watch" -> find_watch (0.85)
Jarvis: Did you mean 'set a timer for 5 minutes, then find my watch', sir?
you> yes                                                  <- both ran
you> wake me in five and beep the watch
Jarvis: Did you mean 'set a timer for 5 minutes, then find my watch', sir?
you> no
Jarvis: I'm afraid I didn't catch that, sir.              <- nothing ran
you> flip a corn
Jarvis: Did you mean 'flip a coin', sir?                  <- M4, unchanged
you> blah blah fishpaste
Jarvis: Could you rephrase that, sir?                     <- {"none": true}
you> forget everything i told you ... yes
you> where did i park
Jarvis: You have not told me, sir.                        <- the prompt had no facts and no turns left
```
The stand-in saw `format=json` on every reasoning call, a 6.0–6.9k-character
prompt, and the facts and turn counts it should have (2, 3, 4 … then 0 after
"forget").

**Thread monitoring (CLAUDE.md step 5):**
- **Tests:** the conftest guard passed on all 101 new tests.
- **Dry run:** 48 threads at startup, 64 after the first classification, then
  64–66 after every turn of both runs, 1 at exit.
- **84 turns** cycling through every new path, with the stand-in: one
  executor worker from first turn to last, no Python thread left over;
  OS-level 64–65, peaking at 67 while a `reason` / `compound` thread is
  alive.
- Without the stand-in the same script ended on 3 executor workers. That is
  not these features: the same drift shows at `HEAD` on turns that pre-date
  M4. Written up as known issue #10.

## Reviewed, and what the review changed

An independent review of the working tree (an Opus agent, given the code and
the two specs and not my reading of them) found eight things. Each of the six
fixed here has a test that failed first.

| Finding | What was done |
|---|---|
| The M4 and plan confirmations used `flows.ask_yes_no`, which matches yes-words *anywhere*: "I'm unsure" and "that's incorrect" ran the guess. That broke the one rule both milestones rest on. | `Orchestrator._confirmed` — a whole-word yes or nothing — for the guess, the plan and forgetting. The loose helper is still used by two M2 dialogs: known issue #13. |
| An interrupt during step one of a chain did not stop step two; an interrupt while the reasoner was thinking was lost, and the question was asked anyway. | `_run_steps` and `_unclear` check `_interrupted()` between steps and after each reasoner call. |
| An utterance queued by one edge was run — and would have been remembered — as the next edge to connect. | `RemoteLink.connect` drops what a *different* device left queued. The older loop behind it is known issue #11. |
| `{"commands": [], "answer": "…"}` (a model filling in every key) lost the answer. | An empty `commands` is no commands. |
| A memory file that did not parse was treated as empty and then written over by the next note. A forget whose delete failed was still called "Forgotten". | The damaged file is set aside (`<device>.json.unreadable-<stamp>`). `forget()` raises, keeps the facts, and JARVIS says the error line. |
| A reasoner that timed out on the guess was then asked to think: 8 s + 15 s of silence, against what `guess_timeout_s` says. | A failed guess ends the turn with the plain line. |
| `"watch\n"` passed as a device id. | `fullmatch` in `memory.py`. The same pattern in `protocol.py` pre-dates this: known issue #12. |
| **"Forget everything" leaves the notes in `notes.txt`.** | Not changed — it is a decision. The plan, the README and this document now say so plainly. See "Open questions". |

## Merged with M5 (the knowledge base)

M5 was built in parallel on `knowledge-base`, from the same `develop`, and
the two were merged on `reasoning-memory-knowledge`. Both kept what the
speaker said, in different places, so the merge had to decide who owns what:

| | Owner after the merge |
|---|---|
| *"Remember that I parked on level two"* | M5's `remember` skill → a fact in the knowledge base (shared by every device) **and** the device's memory, which keeps the fact's knowledge-base id. With the knowledge base off, the device's memory alone. |
| *"Note that …"* | `note` → `notes.txt`, M5's `dictated-notes.md` mirror, and the device's memory. |
| *"Where did I park?"*, *"what do my notes say about …"* | M5's `recall` — the NLU is sure of these, so the reasoner is never asked. |
| *"What do you remember?"* | M4.5's `recall_memory`: this device's list, read back. ("Read back my notes" moved to M5's `recall`; on the merged model it was `recall` 0.55 against `recall_memory` 0.34.) |
| *"Forget everything I told you"* | M4.5's `forget_memory`: the device's memory **and** the knowledge-base facts it put there (by id), or neither — a knowledge base that cannot remove one is the error line, not "Forgotten". Another device's facts stay. |

Tests: `tests/test_memory_knowledge.py` (6). The merged suite passes in full
(see the merge commit). Still not removed by "forget": the notes in
`notes.txt` and in `dictated-notes.md`, both files shared by every device —
the open question below now covers both.

The NLU, trained on the merged seed in isolation, keeps the new intents apart:
"remember that I parked on level two" → `remember` 0.92, "what do you
remember" → `recall_memory` 0.77, "where did I park" → `recall` 0.81, "forget
everything I told you" → `forget_memory` 0.80, "forget the coin skill" →
`remove_skill` 0.90.

## Found along the way

- **Known issue #5 is fixed here** (decision 11) and has left
  `PRD/known-issues.md`. It was: a cancel (Enter, or the edge's `interrupt`)
  during a question asked from *inside* a turn — an edge tool's "How long,
  sir?", a teach/edit/revert dialog — raised `_Cancelled` through `handle()`
  and out of `run()`, stopping the loop. `_session` only caught it around its
  own listen. `Orchestrator._turn` now catches it around the turn. Test:
  `test_a_cancel_during_a_question_inside_a_turn_does_not_stop_the_loop`,
  which reproduces the "How long, sir?" case.
- **Shared data.** On this machine the production pod's `/app/data` is the
  repo's `data/` (same directory), so a test run or dry run that retrains the
  NLU changes what the pod loads on its next restart. The suite and the dry
  runs for this milestone were therefore run from a scratch directory with
  their own `config.toml` and data dir; `data/` was not written to.
  **Restarting the dev brain on this tree will retrain the NLU with the two
  new intents**, and the pod — still on the old code — will then answer
  "what do you remember" with "I don't know how to do that yet" until it is
  rebuilt from a `develop` that has this.
- **Two more classes cost a little everywhere.** With `recall_memory` and
  `forget_memory` in the model, existing intents keep their confidence to
  within a point or two, with one exception worth knowing: "you can forget
  how to flip a coin" (a `remove_skill` seed pattern) went from 0.68 to 0.56,
  under the 0.6 bar for meta-actions. Rewording the new patterns did not
  bring it back; it is the price of two more classes, the same one every
  learned skill charges. On the other side: "forget everything" was
  `remove_skill` at 0.95 and "what do you remember" was `note` at 0.67 on the
  live model — both wrong, both fixed by the new intents.
- **Known issue #10** (the executor's occasional extra worker), from the
  thread monitoring above.

## Not verified here

- **A real Ollama model.** Whether `qwen2.5:3b` returns the JSON it is asked
  for, plans and answers sensibly, says `none` to room noise instead of
  answering it, and does all that inside `reason_timeout_s` on the brain's
  hardware. An unclear turn can now cost two model calls (the mishearing
  guess, then this). Things to watch: answers to things nobody asked — the
  conversation window after a reply is wide, and anything said in it that is
  not a command now goes to the reasoner; `answer_questions = false` turns
  that off and keeps the rest.
- **Voice and the remote path with real hardware.** The device id, the turn
  recording and the questions are covered over a loopback socket, not a
  watch.
- **Answers are a small local model's.** It will be wrong sometimes. The
  prompt tells it to say so when it does not know and never to invent facts
  about the speaker; nothing checks that it obeys.

## Open questions

- **Should `note` keep writing `notes.txt`** (and, since M5, mirror into
  `dictated-notes.md`)? Today a note goes to the
  device's memory *and* to the skill's shared log, and "forget everything"
  clears only the first, so JARVIS says "Forgotten" while the text is still
  on disk. Recommended: stop writing `notes.txt` when a memory is present
  (the memory file *is* the device's notes), keeping it only as the fallback
  without one. That changes what a file some people may read by hand
  contains, so it was left for the owner.
- Facts are only ever added whole or forgotten whole. "Forget where I
  parked" (one fact) is the obvious next step.
- "What is …" / "who is …" still classify as `search` and go to Wikipedia,
  not to the reasoner — the NLU is sure of them, so the reasoner is never
  asked. Whether the reasoner should get first refusal is a product
  decision.
- A confident whole that is secretly two commands of the *same* skill
  ("flip a coin and then flip three coins") stays one command (decision 4).
- M5's known issue #14 ("a question about the documents that does not sound
  like one is `unknown`") says its fix belongs with M4: ask the knowledge base
  before giving up on an `unknown`. `_think` is that place — the excerpts
  could go into the reasoning prompt. Not done in the merge.
