# Steady the ship: outcome

**Date:** 2026-10-03
**Branch:** `steady-ship` (off `develop`, not merged)
**Result:** ✅ The suite passes on the owner's machine with its real
`config.toml` (1030 passed, 2 skipped; it was 5 failed). Seven known issues are
fixed: #4, #6, #7, #11, #12, #13 and #15. The 2024 code is deleted, and the
README, CLAUDE.md and the PRD describe what `jarvis/` actually is, including the
watch work that had no record.

This was not a PRD milestone. It came from a read-through of every doc in the
repo, and the owner chose to "steady first" and to delete the legacy code now.

---

## Fixed

| # | What was wrong | Fix |
|---|---|---|
| 4 | Four auth tests inherited the machine's `[server]` (`allow_insecure = true`) | `tests/remote_harness.make_config` builds a fresh `ServerConfig` |
| 6 | The builtin-roster test saw the watch's tools in `data/remote/tools` | It discovers from an empty `tmp_path` data dir |
| 13 | `flows.ask_yes_no` matched yes-words as substrings, so "I'm unsure" confirmed removing a skill | `ask_yes_no` is now `ask_yes_no_or_none(...) is True` (whole words). A negated yes-word ("not sure") is unclear, which also fixes it for background questions and `Orchestrator._confirmed`, where "not sure" counted as yes |
| 12 | `$` in validators accepted a trailing newline, so `"watch\n"` was a device id and a filename | `\Z` in every validator: device id, tool name, power-log file name, firmware version, strict base64, the `/power` day |
| 11 | Words queued by an edge that left woke an empty session about once a second | `RemoteLink.triggered()` is false while disconnected; the same device reconnecting still gets them |
| 15 | Editing `intents.json` or a skill's examples did not retrain | Each model version records `corpus_digest` (text/label pairs + embedding model) in `meta.json`. `ensure_nlu` retrains when it differs. A model from before the digest retrains once; one with no `meta.json` keeps the labels-only check |
| 7 | A learned skill's params were never filled ("flip three coins" flipped one) | `integer`/`number` params of a **learned** skill are read with `slots.extract_typed`. Strings and booleans are not guessed (see below) |

## Removed

`libs/`, `actions/`, `Movies/` (already copied to `personas/jarvis/style/`),
the legacy, NLTK and Keras sections of `check_setup.py`, and `*.keras` /
`*.pkl` in `.gitignore`. `main.py` is the thin shim the migration mapping
planned (`python main.py` = `python -m jarvis`). Comments in `jarvis/` that
say what a module replaced ("Replaces `libs/brain.py`") were kept as history.

## Docs

- **README**: rewritten around `jarvis/`. It drops the "Claude-backed Q&A
  fallback" claim (Claude teaches, never answers), adds a section on teaching
  a skill, introduces the watch as an edge, and lists every CLI command.
- **CLAUDE.md**: the legacy setup, architecture, intent table and retraining
  triggers are replaced with a `jarvis/` architecture map, the rules that
  matter when changing it, how to add a builtin / an intent / a watch tool,
  and the shared-`data/` warning.
- **PRD**: the resolved blockers are removed (the key resolves the model with
  no workspace id), the migration mapping is marked complete, the watch is in
  the topology, and a Milestone 3.5 section is added. M6 now says the factory
  already retries with feedback (5 attempts, as fresh prompts) and that the
  docs rewrite and the legacy removal are done.
- **M3.5**: the draft plan is brought over from the unmerged `esp32-edge`
  branch and marked as superseded in part. `PRD/milestone-3.5-esp32-edge-outcome.md`
  is written from the history.
- **Outcome headers** of M2–M5 now say merged, not "not yet committed".

## Test changes (said explicitly, per CLAUDE.md)

- `tests/remote_harness.py`, `tests/test_skills.py`: isolation only (#4, #6).
  No assertion changed.
- New tests:
  - `tests/test_flows.py`: #13, both helpers;
  - `tests/test_protocol.py`, `test_power_log.py`, `test_edge_tools.py`,
    `test_remote_firmware.py`: #12;
  - `tests/test_remote_link.py`: #11;
  - `tests/test_nlu_freshness.py`: #15;
  - `tests/test_learned_params.py`: #7.

  Each failed before its fix. The firmware one first passed for the wrong
  reason (no file existed to refuse), so it now stages a `watch\n.bin`.
- **One test of mine was changed before it ever passed.** For #7 I first
  mapped the factory's `string` type to text extraction, with a test expecting
  "add to the shopping list that we need eggs" → `item="we need eggs"`. It
  returned "the shopping list that we need eggs", because extraction takes
  what follows the first "to"/"that". A wrong string is worse than none: the
  skill can no longer use its default or ask. So strings are no longer
  extracted, and the test now asserts that they are left out. No PRD
  requirement was involved.

## Verification

- `python -m pytest` in the worktree, with the real `config.toml` linked:
  **1030 passed, 2 skipped, 0 failed** (baseline: 992 passed, 5 failed: #4 ×4
  and #6).
- `python check_setup.py`: all required checks pass; the model resolves.
  `python main.py --selftest` and `python -m jarvis --selftest`: PASS.
- **#15 on the real model** (a copy of `data/`): the first start retrained
  v28 (no digest) into v29; the second start kept v29.
- **Dry run** (`python -m jarvis -v text`, copied `data/`, the locally learned
  `flip_a_coin`, API key blocked):
  - "flip three coins" flipped 3, and "flip 10 coins" flipped 10;
  - "remove the flip a coin skill" → "flip a coin" → *"Remove … for good?"* →
    "i'm unsure" → *"Very well, I'll keep it."*; the same with "not sure";
  - the skill still answered afterwards.
- **Threads** (CLAUDE.md step 5): the conftest guard was green on all 1030
  tests. In the dry run, sampled every 0.5 s: 48 at startup, then 64–66 from
  the first classification on (the ONNX pool). No growth across turns, and 48
  at exit.

## Found along the way

- **Known issue #18 (new):** "forget how to flip a coin" is classified as
  `flip_a_coin` (0.43 vs `remove_skill` 0.22) and runs the skill.
- **The M6 repair loop is partly built.** The PRD still described a single
  pass. The PRD now says what is left.
- `"not sure"` counted as a **yes** in `ask_yes_no_or_none`, the helper used
  for "Shall I keep it?", for guesses and for forgetting. It is unclear now.

## Not done here

Known issues #3 (multi-device and non-blocking speech, a milestone of its own),
#8 and #9 (OTA), #10, #14, #16, #17 and #18. No real-Ollama verification yet.
`master` is still behind `develop`.
