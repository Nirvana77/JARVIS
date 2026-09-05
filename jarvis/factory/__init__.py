"""The skill factory (M2) — see PRD/milestone-2-skill-factory.md.

Claude runs on the host (needs network + key); everything it generates is
validated (`validate.py`) and executed only inside a sandbox (`sandbox.py`)
before it is ever trusted. `build.py` is the pure orchestration between the
two; `flows.py` is the voice dialog (`teach` / `edit_skill` / `revert_skill`)
that drives them. Nothing here is on JARVIS's hot path — it fires only when a
skill is being taught or changed.
"""
