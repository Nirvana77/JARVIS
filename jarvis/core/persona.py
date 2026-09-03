"""Personas — swappable, data-only.

A persona is a folder under ``personas/`` (never code):

  persona.toml   display name, style description + rules, address term, Piper voice
  style/*.txt    in-character transcripts that condition the LLM style-rewrite
  responses.toml canned lines for fixed events, spoken verbatim (in voice already)

Selection order: ``JARVIS_PERSONA`` env -> ``[persona].active`` in config.toml ->
``"jarvis"`` (resolved in :mod:`jarvis.config`). Fixed events use ``line()`` and
never need an LLM, so they stay in character on a pure-CPU box. Dynamic lines go
through ``phrase()``, which rewrites via the local reasoner when one is present
and otherwise returns the text unchanged.
"""

from __future__ import annotations

import logging
import random
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jarvis.core.reasoner import Reasoner

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PERSONAS_DIR = _REPO_ROOT / "personas"

_QUOTE = re.compile(r"[“”\"]")
_SECTION = re.compile(r"^\s*(spoken by j\.?a\.?r\.?v\.?i\.?s|spoken about|dialogue)", re.I)
_FOOTER_PREFIX = ("―", "—", "-")  # ―, —, -
_JARVIS_NAME = re.compile(r"j\.?a\.?r\.?v\.?i\.?s\.?", re.I)
_ADDRESSED = re.compile(r"^\s*j\.?a\.?r\.?v\.?i\.?s\.?\b", re.I)
# "sir" as a discussed word rather than a vocative ("call him a 'sir'")
_SIR_AS_WORD = re.compile(r"""['"“”]\s*sir|\bcall (?:him|me|you|it) """, re.I)


class PersonaNotFound(Exception):
    pass


@dataclass
class Persona:
    name: str
    display_name: str
    style_description: str
    rules: list[str]
    address_term: str
    voice: str | None
    style_lines: list[str] = field(default_factory=list)
    style_line_sources: dict[str, int] = field(default_factory=dict)
    _responses: dict[str, list[str]] = field(default_factory=dict)
    _reasoner: Reasoner | None = None
    _embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    _style_vecs: np.ndarray | None = field(default=None, repr=False)

    # -- construction ------------------------------------------------------

    @classmethod
    def load(
        cls,
        name: str,
        config,
        reasoner: Reasoner | None = None,
        personas_dir: Path | None = None,
    ) -> "Persona":
        root = (personas_dir or PERSONAS_DIR) / name
        if not root.is_dir():
            available = sorted(
                p.name for p in (personas_dir or PERSONAS_DIR).glob("*") if p.is_dir()
            )
            raise PersonaNotFound(
                f"persona {name!r} not found under {root.parent} "
                f"(available: {', '.join(available) or 'none'})"
            )

        meta_path = root / "persona.toml"
        if not meta_path.is_file():
            raise PersonaNotFound(f"{root} is missing persona.toml")
        meta = tomllib.loads(meta_path.read_text(encoding="utf-8"))

        responses: dict[str, list[str]] = {}
        resp_path = root / "responses.toml"
        if resp_path.is_file():
            raw = tomllib.loads(resp_path.read_text(encoding="utf-8"))
            for event, value in raw.items():
                responses[event] = [value] if isinstance(value, str) else list(value)

        address_term = str(meta.get("address_term", ""))
        style_lines: list[str] = []
        sources: dict[str, int] = {}
        style_dir = root / "style"
        if style_dir.is_dir():
            for txt in sorted(style_dir.glob("*.txt")):
                extracted = _extract_jarvis_lines(
                    txt.read_text(encoding="utf-8"), address_term
                )
                if extracted:
                    sources[txt.stem] = len(extracted)
                    style_lines.extend(extracted)
        # dedupe, keep order
        seen: set[str] = set()
        style_lines = [
            s for s in style_lines if not (s.lower() in seen or seen.add(s.lower()))
        ]

        return cls(
            name=name,
            display_name=str(meta.get("display_name", name.title())),
            style_description=str(meta.get("style", "")).strip(),
            rules=[str(r) for r in meta.get("rules", [])],
            address_term=address_term,
            voice=meta.get("voice") or None,
            style_lines=style_lines,
            style_line_sources=sources,
            _responses=responses,
            _reasoner=reasoner,
            _embedding_model=config.nlu.embedding_model,
        )

    # -- fixed events ---------------------------------------------------------

    def line(self, event: str, default: str = "") -> str:
        """A canned, in-voice line for a fixed event. Spoken verbatim."""
        choices = self._responses.get(event)
        return random.choice(choices) if choices else default

    # -- dynamic phrasing ---------------------------------------------------

    def phrase(self, text: str) -> str:
        """Rewrite a dynamic line into the persona's voice, if a reasoner is up."""
        text = text.strip()
        if not text:
            return text
        if not (self._reasoner and self._reasoner.available and self.style_lines):
            return text
        try:
            shots = self._nearest_style_lines(text, k=6)
            system = self._system_prompt(shots)
            out = self._reasoner.generate(
                system,
                f"Rewrite this line in character, keeping the meaning and any "
                f"facts exactly. Reply with only the rewritten line.\n\n{text}",
            )
            return _strip_wrapping_quotes(out) or text
        except Exception as exc:  # noqa: BLE001
            log.warning("persona rewrite failed, using plain text: %s", exc)
            return text

    def _system_prompt(self, shots: list[str]) -> str:
        parts = [self.style_description or f"You are {self.display_name}."]
        if self.rules:
            parts.append("Rules:\n" + "\n".join(f"- {r}" for r in self.rules))
        if shots:
            parts.append(
                "Lines in the right voice:\n" + "\n".join(f'"{s}"' for s in shots)
            )
        return "\n\n".join(parts)

    def _nearest_style_lines(self, text: str, k: int) -> list[str]:
        if len(self.style_lines) <= k:
            return list(self.style_lines)
        if self._style_vecs is None:
            from fastembed import TextEmbedding

            embedder = TextEmbedding(model_name=self._embedding_model)
            self._style_vecs = np.asarray(
                list(embedder.embed(self.style_lines)), dtype=np.float32
            )
        from fastembed import TextEmbedding

        embedder = TextEmbedding(model_name=self._embedding_model)
        q = np.asarray(next(iter(embedder.embed([text]))), dtype=np.float32)
        sims = self._style_vecs @ q
        top = np.argsort(sims)[::-1][:k]
        return [self.style_lines[i] for i in top]


# -- transcript parsing -------------------------------------------------------

def _strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and _QUOTE.match(s[0]) and _QUOTE.match(s[-1]):
        s = s[1:-1].strip()
    return s


def _extract_jarvis_lines(raw: str, address_term: str) -> list[str]:
    """Pull the lines J.A.R.V.I.S. himself speaks out of a wiki-style transcript.

    Format: ``Spoken by J.A.R.V.I.S:`` / ``Spoken about J.A.R.V.I.S:`` /
    ``Dialogue:`` section headers, ``"quoted"`` lines, and ``―Speakers[src]``
    footers listing who spoke, in turn order.
    """
    term = (address_term or "sir").lower()
    section = ""
    block: list[str] = []
    out: list[str] = []

    def flush(footer: str) -> None:
        if not block:
            return
        speakers = [s.strip() for s in re.split(r",| and ", footer) if s.strip()]
        jarvis_first = bool(speakers) and _JARVIS_NAME.search(speakers[0])
        two_party = len(speakers) == 2 and any(_JARVIS_NAME.search(s) for s in speakers)
        for i, quote in enumerate(block):
            low = quote.lower()
            is_jarvis = False
            if section.startswith("spoken by"):
                is_jarvis = True
            elif (
                re.search(rf"\b{re.escape(term)}\b", low)
                and not _ADDRESSED.match(quote)
                and not _SIR_AS_WORD.search(quote)
            ):
                is_jarvis = True  # only J.A.R.V.I.S. addresses the user as "sir"
            elif section == "dialogue" and two_party:
                # strict alternation from whoever the footer names first
                is_jarvis = (i % 2 == 0) if jarvis_first else (i % 2 == 1)
            if is_jarvis and not _ADDRESSED.match(quote) and len(quote) >= 6:
                out.append(quote)
        block.clear()

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _SECTION.match(stripped):
            flush("")
            m = _SECTION.match(stripped)
            head = m.group(1).lower()
            section = "spoken by" if head.startswith("spoken by") else (
                "spoken about" if head.startswith("spoken about") else "dialogue"
            )
            continue
        if stripped[0] in _FOOTER_PREFIX and "[src]" in stripped:
            flush(stripped.lstrip("".join(_FOOTER_PREFIX)).replace("[src]", ""))
            continue
        if _QUOTE.match(stripped[0]):
            if section == "spoken about":
                continue
            block.append(_strip_wrapping_quotes(stripped))
    flush("")
    return out
