"""Persona loading, selection order, and the JARVIS-line extractor."""

from __future__ import annotations

import textwrap

import pytest

from jarvis.config import load_config
from jarvis.core.persona import Persona, PersonaNotFound, _extract_jarvis_lines


def test_load_jarvis_persona(config):
    p = Persona.load("jarvis", config)
    assert p.display_name == "JARVIS"
    assert p.address_term == "sir"
    assert p.voice == "en_GB-alan-medium"
    assert p.rules and p.style_description


def test_jarvis_style_pool_spans_multiple_films(config):
    p = Persona.load("jarvis", config)
    assert len(p.style_line_sources) >= 2, p.style_line_sources
    assert len(p.style_lines) >= 30
    # most lines are in-voice (contain the address term or are declarative asides)
    with_sir = sum(1 for s in p.style_lines if "sir" in s.lower())
    assert with_sir >= 15


def test_extractor_excludes_non_jarvis_lines(config):
    p = Persona.load("jarvis", config)
    joined = "\n".join(p.style_lines).lower()
    # lines spoken *to* / *about* JARVIS must not leak in
    assert "grow a spine" not in joined              # Tony, The Avengers
    assert 'call him a "sir"' not in joined          # Ultron, Age of Ultron
    assert not any(s.lower().startswith("j.a.r.v.i.s") for s in p.style_lines)


def test_extractor_on_a_small_fixture():
    raw = textwrap.dedent(
        """\
        Spoken by J.A.R.V.I.S:
        "We are now running on emergency backup power."
        ―J.A.R.V.I.S.[src]

        Dialogue:
        "J.A.R.V.I.S., you there?"
        "At your service, sir."
        ―Tony Stark and J.A.R.V.I.S.[src]

        Spoken about J.A.R.V.I.S:
        "Grow a spine, J.A.R.V.I.S.. I got a date."
        ―Tony Stark[src]
        """
    )
    lines = _extract_jarvis_lines(raw, "sir")
    assert "We are now running on emergency backup power." in lines
    assert "At your service, sir." in lines
    assert all("grow a spine" not in s.lower() for s in lines)
    assert all(not s.lower().startswith("j.a.r.v.i.s") for s in lines)


def test_load_plain_persona_has_no_style(config):
    p = Persona.load("plain", config)
    assert p.style_lines == []
    assert p.address_term == ""


def test_canned_line_is_verbatim_from_the_list(config):
    p = Persona.load("jarvis", config)
    for _ in range(10):
        assert p.line("greeting") in {
            "For you, sir, always.",
            "At your service, sir.",
            "Online and ready, sir.",
        }
    assert p.line("no_such_event") == ""
    assert p.line("no_such_event", "fallback") == "fallback"


def test_phrase_is_identity_without_a_reasoner(config):
    p = Persona.load("jarvis", config)
    assert p.phrase("The weather in Malibu is 72 degrees.") == (
        "The weather in Malibu is 72 degrees."
    )


def test_env_var_beats_config_active(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[persona]\nactive = "jarvis"\n', encoding="utf-8")

    monkeypatch.delenv("JARVIS_PERSONA", raising=False)
    assert load_config(cfg_file).persona.active == "jarvis"

    monkeypatch.setenv("JARVIS_PERSONA", "plain")
    assert load_config(cfg_file).persona.active == "plain"


def test_missing_persona_raises_with_a_helpful_message(config):
    with pytest.raises(PersonaNotFound) as exc:
        Persona.load("does-not-exist", config)
    msg = str(exc.value)
    assert "does-not-exist" in msg
    assert "jarvis" in msg and "plain" in msg  # lists what's available
