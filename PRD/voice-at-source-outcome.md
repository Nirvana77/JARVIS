# Replies in character, written at the source: outcome

**Date:** 2026-10-03
**Branch:** `voice-at-source` (off `develop`)
**Asked for:** the owner, after M7 turned the cluster's Ollama on and every
skill reply started being rewritten by it. Character matters to them. They
chose **D**, writing the voice into the skills, with **C**, a fact-checked
rewrite, as the backup.
**Result:** ✅ Voiced skills are spoken as written (0.2 s for "Noted, sir.").
A rewrite that loses a fact is thrown away. The factory writes new skills in
the persona's voice. Suite: 1150 passed, 2 skipped.

## The problem (measured with `qwen3:8b` on the cluster)

`Persona.phrase` rewrote every dynamic reply, which cost 0.4–0.9 s each, and
it changed content:

- "1 sheep, 2 sheep, 3 sheep." → "One, sir. Two, sir. Three, sir."
- "Heads, Heads, Heads — 3 heads and 0 tails." → "Three, sir."

## D: the voice is written into the skill

- **`SkillManifest.voice`** names the persona a skill's lines are written in.
  When it is the active persona, the orchestrator speaks the line as written
  (`Orchestrator._phrase` / `_voiced`). With another persona active, the line
  gets the checked rewrite instead.
- **The 9 builtins** declare `voice="jarvis"`, and their lines were rewritten
  once by hand ("Noted, sir.", "According to Wikipedia, sir: …"). The
  brain-side fallbacks of edge tools ("The watch isn't connected to me, sir.")
  are in voice too.
- **The factory** gets a `VoiceGuide` (the persona's name, its character and
  up to 12 sample lines) and asks Claude to write every reply in that voice.
  Facts stay explicit, short replies vary their wording with `random.choice`,
  and the manifest declares `voice`. In a real call for `draw_a_card`, Claude
  produced `voice="jarvis"` and "The {card}, sir." / "You have drawn the
  {card}, sir." / "The deck offers the {card}, sir.".
- **Canned replies** to an `intents.json` intent use the persona's
  `reply_<tag>` line when there is one (`reply_thanks`); otherwise
  `intents.json`'s reply gets the checked rewrite.

## C: a rewrite may not change the facts

`jarvis/core/voice.py` keeps a rewrite only when it has exactly the same
numbers (digits or words: "5" = "five", "21" = "twenty-one") and every content
word of the original (four letters or more, not a function word, by crude
stem). Otherwise the original is spoken. A line of more than 30 words is never
rewritten. Strict on purpose: when in doubt it is plainer, never wrong.

In the dry run, the check rejected "3 heads and 0 tails" → "Three, sir." and,
before `reply_thanks` existed, "Happy to help!" → "A pleasure, sir.". That
second case is why canned replies are now voiced at the source.

## Known issue #16, closed

- **The embedder:** one per persona, not one per call (M7).
- **The event loop:** the rewrite runs in a thread (M7).
- **The knowledge answer's double generation:** gone. `recall` is a voiced
  skill, so its answer is not rewritten again.
- **The rewrite losing content:** it is now fact-checked (above).

## Tests changed, said plainly

Fourteen assertions in `test_knowledge_e2e.py`, `test_knowledge_skills.py`,
`test_memory.py`, `test_memory_knowledge.py` and `test_skills.py` pinned the
builtins' old wording ("Noted.", "I'll remember that.", "According to
Wikipedia: …", "Opening github.", "I had trouble reaching Wikipedia."). The
wording was what this change specified to change. Only the expected string
moved; what each test checks did not.

`VoiceGuide.from_persona` was made to tolerate a stub persona after
`test_serve_builds_an_orchestrator_that_cannot_be_shut_down` caught it
assuming a fully loaded one. That was a code fix; the test is unchanged.

## Dry run (text mode, the cluster's Ollama)

| Said | Spoken | Time |
|---|---|---|
| note that the gate code is 5512 | Noted, sir. | 0.20 s (voiced: no rewrite) |
| search python programming language | According to Wikipedia, sir: … | 0.80 s (the Wikipedia fetch) |
| flip three coins | Heads, Heads, Heads — 3 heads and 0 tails. | 0.40 s (rewrite rejected) |
| thanks | Happy to help! | 0.60 s (rewrite rejected; `reply_thanks` came after) |

## Left as it is

- **`flip_a_coin`**, learned before this, is not in voice. "Edit the flip a
  coin skill" now rewrites it in voice, through the factory's voice guide.
- **The watch's own replies** (`result.say` from the firmware) are written in
  the watch repo, so they get the checked rewrite. They could be voiced there.
- **`hours`, `payments`, `opentoday`, `weather`** in `intents.json` are 2024
  tutorial leftovers ("We accept VISA, Mastercard and AMEX"), unrelated to
  this change.
