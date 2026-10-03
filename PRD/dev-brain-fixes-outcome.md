# What the first session on the dev brain after M7 showed: outcome

**Date:** 2026-10-03
**Branch:** `dev-brain-fixes` (off `develop`)
**From:** the owner's session on the watch (`./jarvis-run -v serve`), and their
summary of it: *"The brain should understand that I meant something else and
try to solve it. I asked the same question two times but JARVIS did not
learn."*
**Result:** ✅ Suite 1178 passed, 2 skipped. The session's questions were
replayed in text mode against the cluster's Ollama, and each one now gets
somewhere.

| What happened | Why | Now |
|---|---|---|
| "What is today's date?" → "You have not asked me to remember anything." | There was no date or time skill; the nearest class was `recall_memory`'s "…today" | New builtin `clock`: "It's 19:34 on Saturday the 3rd of October, sir." |
| The same question twice more, the same wrong answer | Nothing treated asking again as "that was wrong" | **Asked again**, below |
| "Did you mean 'fetch the power log, then get the power data from the watch'?" | The reasoner planned one command twice | A plan keeps one step per command |
| "Exactly." → "I didn't catch that." | Not a yes-word | "exactly", "precisely", "spot on", "that's right", "right", "indeed", "that's it" (a negation, "not exactly", is still not a yes) |
| "Learn how to get the current date." → "I'm afraid I didn't catch that." | Learning was off on the dev brain, and JARVIS said so as if it had not heard | Both brains learn (below). With learning off it says so: "Learning is switched off on this brain, sir …" |
| Both machines tell the time in UTC | No time zone anywhere | `[general] timezone`, used by `clock`, the reasoner's "Now" and `ctx.now()`. The factory tells Claude to use `ctx.now()` |

## Asked again

A question asked again within `[learning] correction_window_s` (60 s), from
the same device, counts as "the last answer was wrong" when it is either the
same question in other words (content words, Jaccard ≥ 0.6: "what is today's
date" / "what is the date of today") or led by a "no" ("No. What is the
current date?"). Commands are not questions: "flip a coin" said twice flips
twice. Different subjects are different questions ("the weather in Paris" /
"… in London").

Then (`Orchestrator._retry_differently`):

1. Whatever answered last time is ruled out: its label and skill are avoided
   by the mishearing guess, the plan, and the classifier's near miss.
2. Whatever that turn taught is undone (its phrasing, or the unsure turn
   waiting to be learned).
3. The reasoner is told they are asking again, what it said, and not to say
   it again.
4. What it finds is learned for **both** wordings. If nothing fits and
   learning is on, it is learned as a new skill with both questions as
   examples.

In the replay, "What's the weather like today?" got intents.json's 2024 line
"The weather is nice". "No. What is the weather today?" then ruled `weather`
out, the reasoner said `learn: "check the weather forecast"`, and JARVIS
answered "I can't do that yet, sir. I shall learn it."

**Seen, and accepted for now.** "What is the date of today?" right after a
*correct* clock answer also counts as asked again, and the reasoner answers it
instead (correctly, in 2 s rather than 0.2 s). A question is rarely re-asked
in other words unless the answer did not land, and the cost is time, not a
wrong answer.

## Both brains learn

M7 had the dev brain log only (`[learning] enabled = false`), so that the pod
would be the one brain learning into the shared `data/`. That was wrong: while
the dev brain runs, HAProxy sends the watch to it, so it is the brain being
talked to. Both learn now. The race the setting avoided is closed a
different way: each brain retrains only on the phrasings it learned itself
(`_own_phrasings` / `_own_pending`), and gets the other's at its next start
through the corpus digest. One test, `test_learning_off_builds_nothing`, now
expects `<learning_off>` instead of `<unknown>`. That is a specification change
at the owner's report, said here as CLAUDE.md asks.

## Smaller

- `voice.content_words` read "today's" as nothing at all (the apostrophe); it
  is "today" now. That also sharpens the rewrite fact-check.
- The "learned skill differs between …" line is INFO, not WARNING: it repeats
  on every start and needs nobody.
- Threads in the replay: 48 at startup, 65–67 per turn, flat.

## Also done, at the owner's yes

- **Removed** `intents.json`'s 2024 tutorial intents `hours`, `payments`,
  `opentoday` and `weather`, which answered with made-up lines ("The weather is
  nice", "We accept VISA, Mastercard and AMEX"). "What's the weather like
  today?" is now `unknown` and goes to the reasoner, which can answer it or
  learn it. "Are you open today?" now lands on `open_app`, because of "open".
  That was a shop-hours question that only made sense for the tutorial bot.
- **`[general] timezone = "Europe/Stockholm"`** in the dev `config.toml` and
  the pod's `jarvis-config`. The dev `config.toml`'s `host = "00.0.0.0"` is
  now `0.0.0.0`, and its `[learning] enabled` is true again.
