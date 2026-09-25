# Known issues

Things that are wrong and not yet fixed, with enough written down that the fix
doesn't start from zero. Fixed entries move out of here into the milestone
outcome doc that fixed them.

---

## 1. The skill factory rejects what it builds: "sandbox tests failed", repeatedly

**Reported:** 2026-09-25, from live use (`python -m jarvis serve` + edge).
**Status:** open, not investigated.
**Severity:** M2's headline feature — `teach` — does not produce a skill.
**Owner:** unassigned.

### What was seen

Teaching a new skill fails at the sandbox stage over and over: Claude generates
the skill, and its tests fail, so the job sets it aside and nothing is learned.
Spoken, that comes out as *"'<name>' failed its own tests, sir. I've set it
aside."* (`jarvis/factory/jobs.py`).

### Where it fails

`LearningJob._generate_validate_sandbox()` in `jarvis/factory/jobs.py`:

```
build (Claude) → validate (AST denylist) → permission → sandbox.run_tests   ← here
                                                      → sandbox.dry_run
```

The failing branch is the `test_result.ok is False` one, which already logs the
subprocess's stderr at WARNING:

```
log.warning("sandbox tests failed for %s:\n%s", manifest.name, test_result.stderr)
```

So **the diagnosis is almost certainly already in the terminal scrollback of a
failing run** — that stderr is the first thing to read, and nothing below is
worth guessing about until it has been.

### Confound to rule out first

M3 fixed a bug (outcome doc, "Found by the first live run") where `_ask` timed
out during a `teach` dialog and aborted it with *"Never mind, then."* That
looked like the factory failing and was not. **Re-run a `teach` dialog on the
fixed code before investigating anything here**; the symptom may be different
or gone.

### Candidates, in the order worth checking

1. **`RLIMIT_AS` = `[factory] sandbox_mem_mb` (512 MB).** `config.toml` already
   carries the note that "pytest's plugin-rewrite discovery alone needs >256MB
   in a venv this size (scans every installed dist-info)". This venv has since
   grown faster-whisper, ONNX Runtime, fastembed, websockets and more, so 512 MB
   may no longer be enough for pytest to *start*. A `MemoryError` or a bare
   non-zero exit with no test output points straight here. Cheap to test: raise
   `sandbox_mem_mb` to 1024 and see whether the same skill passes.
2. **`RLIMIT_CPU` = `sandbox_cpu_s` (5 s) / `sandbox_timeout_s` (10 s).** Same
   shape of problem — pytest's collection on a large venv is not free. A
   `SandboxError` (timeout) rather than a test failure points here.
3. **The generated tests themselves.** Claude may be writing tests that need
   the network (`unshare` is unavailable on this box — `--selftest` reports
   `net-isolation no`, so a network test would *pass* locally and fail on a box
   where isolation works, or vice versa), or that import something `python -I`
   cannot see. Read the generated test file: failures are kept, not deleted —
   see `config.skill_quarantine_dir` (`data/skills/_quarantine/`).
4. **The prompt.** If the tests are wrong in a consistent way, the fix is in
   `jarvis/factory/claude_client.py`'s prompt rather than in the sandbox.

### What to capture next time it happens

- the `sandbox tests failed for <name>` WARNING and the stderr under it;
- the quarantined skill and its test file from `data/skills/_quarantine/`;
- whether `python -m jarvis --selftest` reports `net-isolation yes` or `no`;
- the skill name and the words used to teach it.

### Why the test suite doesn't catch it

`tests/test_factory_*.py` exercise the factory against a **fake Claude** and a
real sandbox with small, hand-written skills, and they pass. Whatever is wrong
is in the live path: either real generated code, or resource limits that the
small test skills never reach.
