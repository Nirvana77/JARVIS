# Milestone 4 — misheard-command reasoning: outcome

**Date:** 2026-10-01
**Branch:** merged into `develop` (via `reasoning-memory-knowledge`)
**Result:** ✅ Where JARVIS used to say "didn't catch that" unconditionally, it
now asks the local reasoner whether the transcript is a mishearing of something
it knows, confirms the guess by voice (*"Did you mean 'flip a coin', sir?"*),
and only on a yes runs the corrected phrase through the real NLU and dispatches
it. With no Ollama running the step is skipped and behaviour is what it was.
Verified by tests and by conversing with the real program in text mode — with a
**stand-in** for Ollama at the HTTP boundary, because this machine has no
Ollama. The real model's guesses are the part still to be checked (see "Not
verified here").

Spec: `PRD/jarvis-2026-rebuild.md`, "Milestone 4". There is no separate plan
doc; the decisions that section left open are listed below.

---

## What shipped

| Area | Module(s) | Notes |
|---|---|---|
| Prompt and parsing | `jarvis/core/mishear.py` (new) | Pure functions. `vocabulary()` — one line per command: the seed intents' patterns, plus each registered skill's name and up to 6 of its examples (a builtin is both and gets one line). `build_prompt()`, `parse_guess()` (first line; strips quotes, a "Command:" label, a full stop; rejects "none", a rambling reply, and a reply that only repeats what was heard), `soundalike()`. |
| Orchestrator | `jarvis/core/orchestrator.py` | `_unclear(text)` replaces the two unconditional `persona.line("unknown")` calls — the `UNKNOWN` branch and the low-confidence meta-action branch. `_reason_about_unclear(text)` returns `(corrected, label, confidence)` or `None`. New `reasoner=` constructor argument. `_too_unsure_for_meta()` is the existing meta-action gate, now shared. |
| Reasoner | `jarvis/core/reasoner.py` | `generate()` takes `temperature=` and `timeout=` per call. The guess uses temperature 0 and its own timeout; the persona rewrite is unchanged (0.7, 30 s). |
| Wiring | `jarvis/app.py` | All three builders (voice, text, serve) pass the reasoner they already built. |
| Config | `jarvis/config.py`, `config.example.toml`, `config.toml` | `[reasoner] correct_misheard` (default true) and `guess_timeout_s` (default 8.0). |
| Personas | `personas/*/responses.toml` | A `did_you_mean` line with a `{guess}` slot, in both personas. |
| Shared helper | `jarvis/factory/jobs.py` | `run_detached()` takes a thread `name=`. |
| Tests | `tests/test_misheard.py` (new) | 38 tests. |

## Decisions the PRD left open

1. **The guess is classified before the question, not after.** The PRD has
   confirm → classify → dispatch. Here the corrected phrase goes through the
   real classifier *first*, and the question is asked only if the classifier
   recognises it (not `unknown`, and not a meta-action below
   `meta_action_threshold`). Dispatch still waits for the yes. Otherwise a
   guess the NLU cannot place gets *"Did you mean X?"* → "yes" → *"I didn't
   catch that"*, which is worse than not asking.
2. **A guess has to sound like what was heard** (`mishear.MIN_SOUNDALIKE =
   0.5`, a `difflib` ratio on letters and digits). This is not in the PRD. A
   small model would often rather answer than say "none", and without a floor
   every stray sentence becomes a question. In the pairs tried, real
   mishearings score 0.7–0.9 ("flip of corn"/"flip a coin" 0.74, "philip
   acorn"/"flip a coin" 0.70) and unrelated ones 0.24–0.42 ("what's the
   capital of france"/"search the news" 0.33). It only stops wild guesses:
   two commands that are phrased alike ("turn on the lights"/"turn off the
   mic", 0.64) still pass, and the question is what catches those. It is one
   constant and one `if`, easy to move or drop once the real model has been
   listened to.
3. **Nothing from the reasoner picks the skill.** It returns a phrase. The
   label, the confidence and the slots all come from the real NLU and slot
   filler run on that phrase — so a correction to a meta-action still has to
   clear `meta_action_threshold` on its own.
4. **One guess per turn.** A "no", silence, or an unclear answer gets the
   plain line. A cancel (Enter, or the edge's button) during the question
   drops the turn silently, the way a cancelled command is dropped.
5. **The guess runs on its own short-lived thread** (`mishear-guess`), not
   `asyncio.to_thread`. See "Thread monitoring".
6. **Off switch.** `[reasoner] correct_misheard = false` keeps Ollama for the
   persona rewrite and skips this step.

## Test changes (said explicitly, per CLAUDE.md)

No existing test was changed. `tests/test_orchestrator.py`'s unknown and
low-confidence-meta tests build the orchestrator without a reasoner and pass
as they were — which is the "reasoner absent → behaviour unchanged" case.

## Verification

Automated:
```
python -m pytest     # 657 passed, 3 skipped, 5 failed
```
The 5 failures are the same 5 as before this work (619 passed, 5 failed on a
clean `develop`), and all come from this machine's local state, not the code:
4 are known issue #4, the fifth is known issue #6.

`tests/test_misheard.py` against the PRD's Verification paragraph:
- reasoner absent, or present but unavailable → the plain line, nothing asked,
  the reasoner never called (both branches);
- a plausible correction, confirmed → the *cleaned* phrase is what the NLU
  sees and what is dispatched, with slots taken from it; the raw reply never
  reaches the NLU;
- declined → the plain line, nothing dispatched, exactly one reasoner call;
- "none" (in six spellings) → the plain line, no question;
- beyond the PRD: nothing is dispatched before the answer; the reasoner
  raising; a guess the NLU does not know; a guess that is an unsure
  meta-action; a guess that repeats the transcript; a guess that sounds
  nothing like it; a cancel during the question; the prompt's contents; the
  off switch; the full `run()` loop; `Reasoner.generate`'s new arguments; and
  that every persona has the line.

**End-to-end dry run** (`python -m jarvis text`, real NLU v23, real registry
with 20 skills, real persona):

Without Ollama — unchanged:
```
you> flip a corn          intent revert_skill (conf 0.54)   <- the low-confidence meta branch
Jarvis: I have no skill for that yet, sir.
you> flip three chords    intent unknown (conf 0.23)
Jarvis: I have no skill for that yet, sir.
```

With the stand-in Ollama on `localhost:11434` (answers `/api/tags` and
`/api/generate`; canned guesses, deliberately untidy ones):
```
you> flip a corn
  guess   : "flip a coin"   -> flip_a_coin (0.92)
Jarvis: Did you mean 'flip a coin', sir?
you> yes
Jarvis: Heads.
you> flip three chords                      reply was: "Flip three coins."  (quoted, full stop)
  guess   : "Flip three coins"   -> flip_a_coin (0.84)
Jarvis: Did you mean 'Flip three coins', sir?
you> yes please
Jarvis: Heads.
you> hats or tales                          reply was: Intended command: heads or tails
  guess   : "heads or tails"   -> flip_a_coin (0.78)
Jarvis: Did you mean 'heads or tails', sir?
you> no
Jarvis: I'm afraid I didn't catch that, sir.
you> make me a sandwich                     reply was: none
Jarvis: I'm afraid I didn't catch that, sir.
you> flip a corn
Jarvis: Did you mean 'flip a coin', sir?
you>                                        (no answer)
Jarvis: I'm afraid I didn't catch that, sir.
```
The stand-in saw `temperature 0.0` and a 5.6k-character prompt (30 commands)
on every guess. The driver and the stand-in were kept in the scratchpad, not
the repo.

**Thread monitoring (CLAUDE.md step 5):**
- **Tests:** the conftest guard passed on all 38 new tests — none leaves a
  thread behind.
- **Dry run:** `/proc/<pid>/task` sampled after every turn. 48 threads at
  startup, 64 after the first classification (ONNX's pool), then flat: 64–65
  through 13 turns with the stand-in, 64 without it, 1 at exit. An 80-turn
  run of misheard commands held at exactly one executor worker throughout.
- **A climb was found and fixed on the way.** The first version made the
  guess with `asyncio.to_thread`, and the thread count crept up — 2 executor
  workers became 9 over 80 turns. Cause: the loop's `ThreadPoolExecutor`
  starts a new worker when a job is submitted before the previous worker has
  marked itself idle, and "guess, then immediately speak" hits that window
  reliably. It is bounded (20 workers on this machine), not a leak, but it
  climbs turn after turn, which is what step 5 says to treat as a bug. The
  guess now runs on its own thread via `run_detached`, which is gone when the
  guess is; it also means a slow Ollama cannot hold up shutdown.

## Found along the way

Added to `PRD/known-issues.md`, not fixed here:
- **#5** — a cancel during a question asked from inside `handle()` (an edge
  tool's "How long, sir?"; by reading, the teach/edit/revert dialogs too)
  escapes `run()` and stops the loop. Pre-existing. *(Fixed the same day, in
  M4.5: `_session` now treats it as a cancelled turn, and the entry moved to
  that outcome doc.)*
- **#6** — `tests/test_skills.py::test_registry_discovers_the_builtins` reads
  the machine's real `data/remote/tools`.
- **#7** — a learned skill's params are never filled: "flip three coins" flips
  one. It is why the corrected "Flip three coins" above answered with a
  single "Heads." — saying it correctly does the same.

Also: `Persona.phrase()` calls the reasoner synchronously on the event loop
(and builds a new `TextEmbedding` per call). Pre-existing, only with Ollama
up, and untouched here — but it means a reply after a confirmed correction
waits on two reasoner calls, one of them blocking the loop.

## Not verified here

- **A real Ollama model.** Whether `qwen2.5:3b` actually answers "flip a coin"
  to "flip a corn", says "none" when it should, and does it inside
  `guess_timeout_s` on a CPU-only brain (the prompt is about 1,400 tokens;
  the list of commands is its unchanging prefix, so Ollama's prompt cache
  should cover most of it after the first call). This is the PRD's manual
  check and it is the one that matters most. Things to watch for: guesses
  rejected by `MIN_SOUNDALIKE` that should not have been (`-v` logs each
  one), and questions asked that should not have been.
- **The voice and remote paths.** The question goes through `_ask`, as the
  teach dialog's do, so on an edge in ByName the "yes" needs no name. Not
  exercised with a real mic or edge.

## Open questions

- An unclear reply to "did you mean" (neither yes nor no) is treated as no
  and dropped. If the user instead repeats the command, re-running that reply
  as a command would be the natural next step — the same question M2.5 left
  open for background questions.
- `MIN_SOUNDALIKE` and `EXAMPLES_PER_COMMAND` are module constants, not
  config. Promote them if they turn out to need tuning per install.
