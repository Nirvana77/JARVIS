"""The only place in the factory that talks to the Anthropic API.

Reads the key the way ``.env`` has always held it (``ANTHROPIC_API_KEY`` / the
older ``api_key``, ``ANTHROPIC_MODEL`` override, ``ANTHROPIC_WORKSPACE_ID``
→ the ``anthropic-workspace-id`` header) and is scoped to one job: turn a
:class:`~jarvis.factory.spec.SkillSpec` into a skill module + test source.
Everything downstream (``validate.py``, ``sandbox.py``) treats what comes back
as untrusted text — this client's only responsibility is getting a plausible
module out of Claude and handing it over as plain strings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from jarvis.factory.spec import SkillSpec

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"

#: shared by adaptive thinking and both modules. 8000 truncated real skills
#: mid-test; kept under the SDK's non-streaming ceiling (~21k).
MAX_TOKENS = 16000

_CONTRACT = '''\
A JARVIS skill is a single Python module. It must define exactly two things
at module scope:

1. `MANIFEST = SkillManifest(...)` — keyword arguments only, every value a
   Python *literal* (strings, lists, dicts, sets — no function calls, no
   f-strings, no variables):
   - `name`: the skill's identifier, matching the module's own name.
   - `description`: a short human-readable sentence.
   - `examples`: a list of 3-6 example phrases a user might say.
   - `params`: a dict describing keyword params `run()` accepts, e.g.
     `{"seconds": {"type": "integer", "required": True}}`. Use `{}` if none.
   - `permissions`: a *set literal* (not `frozenset(...)`) drawn only from
     {"pure", "notify", "fs_read", "fs_write", "net", "shell"}. Request the
     minimum the skill actually needs — most skills need only `{"pure"}`.
   - `voice`: only when the request names a voice to write in (see "Voice"
     below): that persona's name, e.g. `voice="jarvis"`. Otherwise leave it out.

2. `def run(ctx, **params) -> str:` — does the work and returns the line
   JARVIS should speak. `ctx` exposes `ctx.say(text)` (speak a progress line
   immediately), `ctx.data_dir` (a `pathlib.Path` scratch directory, created
   on first access), and `ctx.llm` (an optional reasoner, may be `None`).
   `run` must always return a `str`, never raise for expected input.

Rules:
   - Only import from the Python standard library or `ctx`. Never import
     `socket`, `http`, `urllib`, `requests`, `httpx`, `subprocess`, `ctypes`,
     and never call `eval`, `exec`, `__import__`, `os.system`, `os.popen`.
   - Never call the builtin `open(...)` directly — use `ctx.data_dir` for any
     file scratch space.
   - No network calls, no shelling out, no filesystem access outside
     `ctx.data_dir`. If the task is genuinely impossible under `{"pure"}`,
     say so in one sentence before the code and use the narrowest permission
     that unblocks it.

Worked example (a real JARVIS skill, for shape only — do not copy verbatim):

```python
from __future__ import annotations

from datetime import datetime

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="note",
    description="Append a timestamped note to your notes file.",
    examples=[
        "note that the wifi code is 1234",
        "write down buy milk on the way home",
        "make a note that the meeting moved to friday",
    ],
    params={"text": {"type": "string", "required": False}},
    permissions={"fs_write"},
)


def run(ctx, text: str = "") -> str:
    text = (text or "").strip()
    if not text:
        return "What would you like me to note?"
    path = ctx.data_dir / "notes.txt"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')}  {text}\\n")
    return "Noted."
```

The test module (pytest, no fixtures beyond a plain stand-in `ctx` object you
construct inline) must import the skill module and exercise `run()` for at
least the happy path.

Keep both modules compact: do what was asked, without extra features, and a
handful of focused tests rather than an exhaustive suite.
'''

_RESPONSE_FORMAT = """\
Respond with exactly two fenced code blocks, in this order, and nothing else:

```python skill
<the full skill module source>
```

```python test
<the full pytest test module source>
```
"""

_BLOCK_RE = re.compile(
    r"```python\s+skill\s*\n(?P<skill>.*?)```.*?```python\s+test\s*\n(?P<test>.*?)```",
    re.DOTALL,
)


@dataclass(frozen=True)
class GeneratedSkill:
    name: str
    module_source: str
    test_source: str


@dataclass(frozen=True)
class VoiceGuide:
    """The persona a generated skill should speak as: its replies are written
    in that voice (``SkillManifest.voice``) instead of being rewritten by the
    local LLM every time they are spoken."""

    name: str
    character: str
    samples: tuple[str, ...] = ()

    @classmethod
    def from_persona(cls, persona, max_samples: int = 12) -> "VoiceGuide | None":
        """``None`` for a persona without a name or a character to write in:
        then skills are written plain and rewritten at run time, as before."""
        name = getattr(persona, "name", None)
        character = getattr(persona, "character", None)
        character = character() if callable(character) else None
        if not isinstance(name, str) or not name or not isinstance(character, str):
            return None
        style = getattr(persona, "style_lines", ())
        lines = [
            l for l in (style if isinstance(style, (list, tuple)) else ())
            if isinstance(l, str) and 3 <= len(l.split()) <= 14
        ]
        step = max(1, len(lines) // max_samples)
        return cls(name=name, character=character, samples=tuple(lines[::step][:max_samples]))


class ClaudeClientError(RuntimeError):
    """Claude was unreachable, mis-configured, or returned an unparsable reply."""


class ClaudeClient:
    """Constructed once, best-effort, in `jarvis/app.py`. ``available`` mirrors
    the ``Reasoner``'s capability-probed pattern — a missing key degrades
    `teach` to "I can't learn new skills right now, sir," never a hard crash.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        workspace_id: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.model = model
        self.available = bool(api_key)
        self._client = None
        #: set by the app once the persona is loaded: skills are written in it
        self.voice: VoiceGuide | None = None
        if not api_key:
            log.info("no Anthropic API key configured — skill factory disabled")
            return
        import anthropic

        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        kwargs = {"api_key": api_key}
        if headers:
            kwargs["default_headers"] = headers
        self._client = anthropic.Anthropic(**kwargs)

    @classmethod
    def from_config(cls, config) -> "ClaudeClient":
        return cls(
            api_key=config.anthropic_api_key,
            workspace_id=config.anthropic_workspace_id,
            model=config.factory.model,
        )

    def generate_skill(
        self,
        spec: SkillSpec,
        existing_source: str | None = None,
        feedback: str | None = None,
    ) -> "GeneratedSkill":
        if not self.available:
            raise ClaudeClientError("no Anthropic API key configured")

        prompt = self._build_prompt(spec, existing_source, feedback, voice=self.voice)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=_CONTRACT + "\n" + _RESPONSE_FORMAT,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 — network/API failures degrade, not crash
            raise ClaudeClientError(f"Claude request failed: {exc}") from exc

        text = "".join(b.text for b in response.content if b.type == "text")
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise ClaudeClientError(
                f"the reply was cut off at {MAX_TOKENS} tokens before both code "
                "blocks were complete — write a smaller skill and fewer tests"
            )
        match = _BLOCK_RE.search(text)
        if not match:
            raise ClaudeClientError("Claude's reply didn't contain the two expected code blocks")
        return GeneratedSkill(
            name=spec.name,
            module_source=match.group("skill").strip() + "\n",
            test_source=match.group("test").strip() + "\n",
        )

    @staticmethod
    def _build_prompt(
        spec: SkillSpec,
        existing_source: str | None,
        feedback: str | None = None,
        *,
        voice: VoiceGuide | None = None,
    ) -> str:
        lines = [
            f"Skill name: {spec.name}",
            f"Description: {spec.description}",
        ]
        if spec.examples:
            lines.append("Example phrases a user might say:")
            lines.extend(f"  - {ex}" for ex in spec.examples)
        if existing_source:
            lines.append(
                "\nThis skill already exists. Here is its current source — "
                "modify it to satisfy the description above, keeping the "
                "existing behavior that isn't part of the requested change:\n"
                f"```python\n{existing_source}\n```"
            )
        else:
            lines.append("\nThis is a brand new skill — write it from scratch.")
        lines.append(
            f"\nThe skill is saved as `{spec.name}.py` and the test runs from the "
            f"same directory, so the test must import it as a top-level module: "
            f"`import {spec.name}` (or `from {spec.name} import run, MANIFEST`). "
            f"It is not inside any package — `from jarvis.skills import {spec.name}` "
            f"will fail."
        )
        if voice is not None:
            lines.append(
                "\nVoice. Every line `run()` returns is spoken aloud by this character, "
                "exactly as written — nothing rewrites it afterwards — so write each one "
                f"in this voice:\n{voice.character}"
            )
            if voice.samples:
                lines.append("Lines in that voice:")
                lines.extend(f'  - "{s}"' for s in voice.samples)
            lines.append(
                "Keep every fact explicit in the line (numbers, names, results): the "
                "voice is in the wording, never instead of the content. A short reply "
                "may pick from two or three phrasings with `random.choice`, so it does "
                f'not sound canned. Declare it in the manifest: `voice="{voice.name}"`.'
            )
        if feedback:
            lines.append(
                "\nA previous attempt at this exact request was rejected. "
                "Read the feedback below carefully and fix the underlying "
                "problem — don't just repeat the same code:\n"
                f"{feedback}"
            )
        return "\n".join(lines)


def default_generate(
    spec: SkillSpec,
    existing_source: str | None,
    feedback: str | None = None,
    *,
    client: ClaudeClient,
) -> GeneratedSkill:
    """The default `generate` callable `build.build()` takes — a thin bind of
    a constructed `ClaudeClient` so `build()` itself never imports `anthropic`."""
    return client.generate_skill(spec, existing_source, feedback)
