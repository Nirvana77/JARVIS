# Milestone 4.5 — reasoning and per-device memory

**Asked for:** 2026-10-01, by the owner, right after M4: *"Can you add
reasoning too? And memory per device."* The three choices put to them:

| Question | Answer |
|---|---|
| What should "reasoning" do? | **Both**: answer open questions, and pick and chain skills. |
| What should JARVIS remember per device? | **Both**: the recent conversation, and lasting facts. |
| Which model reasons? | **Local Ollama only.** No Claude at run time, as the PRD already says. |

M4 gave the reasoner one job: guess what a misheard command was. This gives it
the rest of the turn the NLU could not place, and gives it something to reason
*with*.

## What it does

1. **Compound commands, with no model at all.** *"Set a timer for five
   minutes and find my watch"* is two commands, and today the classifier
   calls the whole sentence `unknown`. The sentence is split at
   "and" / "then" / a comma, each clause is classified on its own, and if
   every clause is a confident, distinct skill they are run in order. These
   are the speaker's own words classified by the real NLU, so nothing is
   guessed and nothing is asked.
2. **Planning, when the NLU is lost.** If the transcript is `unknown` and M4
   found no mishearing, the reasoner is asked what the speaker wants. It may
   answer with **commands** — the request re-said the way JARVIS's commands
   are said (*"wake me in five and beep the watch"* → `set a timer for 5
   minutes`, `find my watch`). Each is classified by the real NLU, confirmed
   by voice in one question, then run in order.
3. **Answering.** Or it may **answer** — one or two spoken sentences, in the
   persona's voice (*"what's the capital of France?"*). Or say there is
   nothing to do, and JARVIS says the plain line as before.
4. **Memory, per device.** Each device (`watch`, `livingroom`; `local` for
   the all-in-one and text modes) has:
   - its **recent conversation** — the last few turns, what was heard and
     what was said back — in RAM, forgotten after a quiet spell or a restart;
   - its **lasting facts** — what it was asked to note or remember — on disk
     in `data/memory/<device>.json`.
   Both go into the reasoner's prompt, so *"where did I park?"* can be
   answered from *"remember that I parked on level two"*, and *"and Spain?"*
   from the turn before. *"What do you remember?"* reads the facts back;
   *"forget everything I told you"* clears them, after a yes.

## Decisions

1. **The reasoner still never picks a skill.** As in M4: it returns
   *phrases*. The label, the confidence and the slots come from the real
   classifier and slot filler run on each phrase. A plan with a step the NLU
   does not recognise is dropped whole.
2. **A guess is confirmed; the speaker's own words are not.** The planner's
   commands are the model's reading of the request, so they are asked about
   (*"Did you mean 'set a timer for 5 minutes, then find my watch', sir?"*),
   once, and only a clear whole-word yes counts ("I'm unsure" is not one).
   The compound split (1) is not a guess and runs straight away. Either kind
   of chain stops at the next step if the speaker cancels.
3. **A chain is skills only.** No step of a chain may be a meta-action
   (teach, edit, revert, remove, forget), a session action (greeting,
   goodbye, shutdown) or dictation (`note`: *"note that buy milk and call
   mum"* is one note). A single planned command may be anything the NLU is
   sure of, as M4's correction may.
4. **The split is conservative.** Every clause must clear a confidence *and*
   a similarity bar well above the classifier's own (`0.6` / `0.6`: measured
   on the live model, real clauses score ≥ 0.77 / ≥ 0.86 and stray halves
   like "garfunkel", "30 seconds", "dogs" score ≤ 0.77 / ≤ 0.53, never both
   high), and neighbouring clauses must be different skills — *"a timer for 5
   minutes and 30 seconds"* and *"search for cats and dogs"* stay whole.
5. **Order on an unclear turn:** compound split → M4's mishearing guess (a
   "no" still ends the turn with the plain line) → plan or answer → the plain
   line. Two reasoner calls at most, and only on a turn the NLU gave up on.
6. **An answer is spoken as it comes.** The persona's character is in the
   prompt, so the answer is not sent through `persona.phrase()` for a third
   call. It is clipped to a few sentences: this is speech.
7. **`note` is how a fact gets in.** *"Remember that …"* already classifies
   as `note` and `tests/test_nlu.py` pins it, so that stays: the `note` skill
   also files the text under the device it was said to (`ctx.memory`), and
   still appends to `notes.txt` as before. No new "remember" intent.
8. **Reading back and forgetting are inline**, like greeting and goodbye:
   two new seed intents, `recall_memory` and `forget_memory`, no LLM needed —
   they work with Ollama off. Forgetting asks first and sits behind
   `meta_action_threshold`, because *"forget …"* is also how a skill is
   removed.
9. **Which device.** The orchestrator asks its mic (`device_id`): the
   `RemoteLink` answers with the connected edge's id; anything else is
   `local`. One turn at a time (known issue #3), so "the device of this turn"
   is well defined.
10. **Not M5.** The facts are a short list handed to the model whole — no
    embeddings, no retrieval, no documents. M5's knowledge base can ingest
    `data/memory/` later; nothing here stands in its way.
11. **A cancelled question no longer stops the loop** (known issue #5). This
    adds three more questions asked from inside a turn; rather than a
    fourth, fifth and sixth local catch, `_session` treats a cancel during a
    turn as the cancelled turn it is.

## Privacy, stated plainly

Facts are plain text on the brain's disk, one file per device, until they are
forgotten. **Forgetting clears the device's memory, not the `note` skill's own
`notes.txt`** — that file is the skill's log from before this milestone, is
shared by every device and is still appended to (decision 7), so a note said
before "forget everything" is still in it. Whether `note` should stop writing
it is an open question for the owner. The conversation is in RAM only. Both are sent to the reasoner —
the local Ollama, nowhere else. Transcripts are already printed on every
turn; nothing new is logged.

## Config

```toml
[nlu]
compound = true            # "…and…" / "…then…" run as separate commands when each half is one

[reasoner]
answer_questions = true    # no command fits: let the reasoner answer
plan_commands = true       # …or re-say the request as commands, confirmed before they run
reason_timeout_s = 15.0

[memory]
turns = 8                  # exchanges kept per device, in RAM
idle_forget_s = 900        # …and dropped after this long without one
max_facts = 200            # lasting facts kept per device, in data/memory/<device>.json
```

## Verification

Unit tests, fake reasoner, no Ollama:

- compound: two confident distinct skills run in order, each with its own
  slots; a weak half, a repeated skill, a `note`, a meta-action → not split;
  switched off → not split; works with no reasoner;
- planning: commands → classified → one confirm → run in order; declined →
  plain line, nothing run; an unrecognised step → plain line, no question; a
  single command → "did you mean"; `plan_commands = false` → ignored;
- answering: spoken as returned (not rephrased), clipped; "none", bad JSON,
  a reasoner error → plain line; `answer_questions = false` → plain line;
  no reasoner → unchanged from M4;
- M4 keeps its behaviour: a mishearing guess comes first, and its "no" does
  not fall through to an answer;
- the prompt carries the commands, the device's facts, its recent turns and
  the time; another device's facts and turns are not in it;
- memory: facts survive a new `Memory` on the same directory, are per device,
  de-duplicated and capped; turns are capped and expire on a fake clock; a
  bad device id cannot name a file;
- `note` files the fact under the turn's device and still writes `notes.txt`;
- `recall_memory` / `forget_memory`: read back, nothing to read, forget after
  yes, kept after no, low confidence → plain line;
- a turn in `run()` is recorded with what was said back, under the link's
  device id;
- a cancel during a question asked inside a turn leaves the loop running.

Then the full suite, a text-mode dry run with a stand-in Ollama (this machine
has none), and thread counts, per CLAUDE.md.
