# Milestone 7: learn from every turn — outcome

**Date:** 2026-10-03
**Branch:** `learn-always` (off `develop`, not merged)
**Result:** ✅ Built and dry-run end to end against the cluster's Ollama
(`qwen3:8b`) and the real Claude API:
- A misheard command, once confirmed, is understood directly after a retrain
  and a restart.
- A request nothing fits was built as a new skill, kept and used.
- A broken learned skill was repaired.
- A correction re-taught a phrasing.

Suite: 1119 passed, 2 skipped (1030 before M7).

Plan: `PRD/milestone-7-learn-from-every-turn.md`.

---

## What was built, part by part

| Part | Where | Notes |
|---|---|---|
| A. The reasoner online | `core/reasoner.py`, `[reasoner] think` | `"think": false` is sent to Ollama. Measured with `qwen3:8b`: 0.25 s a guess, against 12.5 s with thinking. The first call after a model swap on the shared GPU took 3.8 s |
| B. Interaction log | `learning/interactions.py`, `Orchestrator._record` | One line per turn: path, outcome, skill, params, runner-up, model version. "Forget everything I told you" deletes the device's log too |
| C. Learned phrasings | `learning/phrasings.py`, `Orchestrator._learn_phrasing`, `_retrain_phrasings`, `_regression_gate` | Third corpus source. Batches retrain at idle; a model doing worse on the seed probes plus recent sure turns is discarded and its phrasings rejected |
| D. Corrections | `Orchestrator._correction`, intent `correction` | Undoes the last turn's phrasing (rejects it for good), learns the heard words as what was meant, runs that, counts the confusion |
| E. Building what nothing fits | `reasoning.py` (`"learn"`), `Orchestrator._learn_capability` | Kept with no keep question. Daily cap, same-day dedup (text, description, or MiniLM cosine ≥ 0.85), permissions still asked unless `auto_permissions` |
| F. Repairs | `Orchestrator._after_failure`, `_start_repair`, `LearningJob` replay | The failing call is replayed in the sandbox and must pass. A skill past its daily repairs is switched off; builtins go to `data/learning/builtin-failures.jsonl` |
| G. Seeing it | intent `learned_today`, `python -m jarvis learning` | `status`, `log --since 2d`, `phrasings`, `undo ID`, `enable SKILL` |

## Where the build differs from the plan

- **Paths.** Phrasings live in `data/learning/phrasings.json` (the plan said
  `data/nlu/learned_examples.json`). Confusions, caps, events and disabled
  skills are in one `data/learning/state.json`, not `confusions.json`, so that
  everything learned sits in one folder.
- **The no-LLM "Did you mean …?"** asks about a turn that came out `unknown`
  but whose best label was within `confirm_margin` *under* `nlu.threshold`
  (`_offer_near_miss`). The plan placed it above the threshold. That would have
  put a question in front of turns that already work today. Instead, an unsure
  turn above the threshold is run and learned unless the next turn corrects it.
- **One-step plans nothing knows count as capabilities.** In the dry run,
  `qwen3:8b`, asked about "roll a twenty sided die", replied with that sentence
  as a command instead of `{"learn": …}`. A single planned step the classifier
  does not know is now treated as a request to learn. The prompt also says
  never to plan a command that is not listed.
- **Known issue #18 guard.** An unsure turn whose runner-up is a meta action is
  not learned. Otherwise "forget how to flip a coin", which runs the coin with
  `remove_skill` second, would be learned as a way to flip a coin.
- **Startup.** `ensure_nlu` already trains on every phrasing in the file (they
  are in the corpus digest), so `build_learning` marks them trained rather
  than retraining them again at the first idle.
- **Known issue #16 is half fixed.** With Ollama reachable, `Persona.phrase`
  rewrites every skill reply. It now builds one embedder and runs off the event
  loop. The double generation for knowledge answers is still open.
- **A `KeyError` raised inside a skill** used to be reported as "skill
  missing". It is now that skill's failure, so a learned one is repaired.
- **The M6 repair loop** (a continued conversation) was not built. Repairs
  reuse the factory's existing retry-with-feedback, as fresh prompts.

## Dry run

Text mode, worktree copy of `data/`, `[learning] retrain_after = 1`, the
cluster's Ollama by its ClusterIP, a deliberately broken learned skill
`count_sheep` (`len(sheep) // 0`).

- **Run 1.** "flip a corn" → `revert_skill` 0.51, held back by the meta-action
  bar → the guess said "none" → the plan offered "flip a coin" → yes → learned
  `flip a corn → flip_a_coin` (plan) → retrained, regression gate passed,
  swapped in, announced "I've learned that 'flip a corn' means …". Run 1 also
  showed the capability gap above, which was fixed before run 2.
- **Run 2, a fresh start on the same data:**
  - "flip a corn" → `flip_a_coin` 0.70 directly. The lesson survived the
    restart.
  - "roll a twenty sided die" → "I can't do that yet, sir. I shall learn it."
    → Claude built `roll_a_twenty_sided` → "I've learned to roll one or more
    twenty-sided dice …" → asked again: `roll_a_twenty_sided` 0.79, "Thirteen,
    sir."
  - "count some sheep" → ZeroDivisionError → "I have run into a problem, sir."
    → repaired in the background → "I've repaired 'count_sheep', sir." → asked
    again: "One, sir. Two, sir. Three, sir."
  - "flip two coins" (ran) → "no I meant roll a twenty sided die" → `correction`
    0.72 → learned `flip two coins → roll_a_twenty_sided` (correction) → rolled.
  - "what have you learned today" listed the two phrasings, the new skill, the
    repair and the correction.
- **Threads.**
  - Run 2: 48 at startup, 64–67 during turns, and a peak of 79 while a newly
    trained model was staged next to the live one (each brings its own
    onnxruntime pool). Back to 35–53 after each swap closed the old one, with
    no growth from turn to turn.
  - Run 1: 48 → 64–66 per turn → 80 while staged → 65 after the swap.

## Known issue #19, fixed after the dry run (owner's request)

The pod kept learned skills on its `jarvis-state` PVC and the dev brain in its
checkout, so a skill one brain learned was unknown to the other. They now live
in `data/skills/learned/` (`Config.learned_skills_dir`), which both brains
share. `Registry.discover` points the `jarvis.skills.learned` package there
(import names unchanged) and `flows.learned_source_path` follows it, so
promotion, edit, revert and repair all write to the shared copy. At startup
`app._discover` copies any skill still in the old place into the shared one
(`migrate_learned_skills`): it never overwrites and never deletes, and when the
two differ it logs which one it kept. Two `test_skills.py` discovery tests now
put their skill under `<data_dir>/skills/learned/` instead of redirecting the
package path by hand: the location was specified to change, not the test's
intent. Both brains had a `flip_a_coin`, and they differed (same behaviour,
different examples); the pod's, which the watch has been using, was put in the
shared folder at the merge.

## Found, not fixed (in `known-issues.md`)

- **#20.** A learned skill's number goes to the first number said: "roll a
  twenty sided die" once rolled twenty dice.
- **#16 (rest).** Every reply rewritten by the LLM can lose content ("1 sheep, 2
  sheep, 3 sheep" → "One, sir. Two, sir. Three, sir.") and adds 0.4–0.9 s.

## Not done here (owner's call)

- **Pointing the pod at the cluster's Ollama.** Set `[reasoner] base_url =
  "http://ollama.ollama.svc.cluster.local:11434"`, `model = "qwen3:8b"` in the
  `jarvis-config` ConfigMap. It only helps once the pod runs this branch: the
  `think` flag is new.
- **Pointing the dev brain at it.** On the host the ClusterIP
  (`http://10.43.254.194:11434`) works; the service DNS name does not resolve
  there.
- **`[learning] enabled = false` in the dev brain's `config.toml`**, so only
  the pod learns.
