"""Replies in character without a rewrite at run time (D), and a rewrite that
may not change the facts when there is one (C).

Owner's choice, 2026-10-03: character matters. Measured with the cluster's
qwen3:8b, the run-time rewrite cost 0.4–0.9 s per reply and could drop
content ("1 sheep, 2 sheep, 3 sheep." -> "One, sir. Two, sir. Three, sir.").
"""

from __future__ import annotations

import asyncio
import importlib
import pkgutil

import pytest

import jarvis.skills.builtin as builtin_pkg
from jarvis.core import voice
from jarvis.core.persona import Persona
from jarvis.factory.claude_client import ClaudeClient, VoiceGuide
from jarvis.factory.spec import SkillSpec
from jarvis.factory.validate import validate
from jarvis.skills.contract import SkillManifest
from tests.test_orchestrator import FakeMic, FakeNLU, FakeRegistry, FakeWake


# -- C: what a rewrite must keep ----------------------------------------------------

@pytest.mark.parametrize("original, rewrite", [
    ("Heads.", "Sir. Heads."),
    ("Timer set for 5 minutes.", "Sir, the timer is set for five minutes."),
    ("Opening Spotify.", "Opening Spotify for you, sir."),
    ("Noted.", "Noted, sir."),
    ("It is 21 degrees.", "Twenty-one degrees, sir."),
    ("You have three new messages.", "Three new messages await you, sir."),
])
def test_a_rewrite_that_keeps_the_facts_is_kept(original, rewrite):
    assert voice.facts_kept(original, rewrite)


@pytest.mark.parametrize("original, rewrite", [
    ("1 sheep, 2 sheep, 3 sheep.", "One, sir. Two, sir. Three, sir."),   # the sheep went
    ("You rolled 13.", "You rolled 31, sir."),                            # a number changed
    ("Timer set for 5 minutes.", "Timer set, sir."),                      # a number went
    ("Opening Spotify.", "Opening the music app, sir."),                  # a name went
    ("The door code is 4821.", "The door code is 4812, sir."),
])
def test_a_rewrite_that_changes_a_fact_is_not(original, rewrite):
    assert not voice.facts_kept(original, rewrite)


def test_numbers_are_read_as_digits_and_as_words():
    assert voice.numbers("Set 2 timers for twenty five minutes and a hundred seconds") == {2, 25, 100}
    assert voice.numbers("It is 3.5 degrees at 7:45") == {3.5, 7, 45}


class Ollama:
    available = True

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, system, prompt, **kw):
        self.calls += 1
        return self.reply


def _persona(config, reply):
    p = Persona.load("jarvis", config, Ollama(reply))
    p._style_vecs = None
    p._nearest_style_lines = lambda text, k: []
    return p


def test_phrase_says_the_original_when_the_rewrite_lost_a_fact(config):
    p = _persona(config, "One, sir. Two, sir. Three, sir.")
    assert p.phrase("1 sheep, 2 sheep, 3 sheep.") == "1 sheep, 2 sheep, 3 sheep."


def test_phrase_keeps_a_faithful_rewrite(config):
    p = _persona(config, "Sir. Heads.")
    assert p.phrase("Heads.") == "Sir. Heads."


def test_a_long_line_is_never_rewritten(config):
    p = _persona(config, "Short, sir.")
    long_line = "According to Wikipedia: " + "word " * 40
    assert p.phrase(long_line) == long_line.strip()
    assert p._reasoner.calls == 0


# -- D: a skill whose replies are already in voice is spoken as written --------------

class PersonaSpy:
    name = "jarvis"

    def __init__(self):
        self.phrased = []

    def line(self, event, default=""):
        return f"<{event}>"

    def phrase(self, text):
        self.phrased.append(text)
        return f"rewritten: {text}"


class TTS:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


def _orch(config, manifest):
    from jarvis.core.orchestrator import Orchestrator
    from jarvis.nlu.corpus import intent_meta

    persona, tts = PersonaSpy(), TTS()
    registry = FakeRegistry(manifests=[manifest])
    registry.dispatch = lambda label, params: "Heads, sir."
    o = Orchestrator(
        config=config, wake=FakeWake(), mic=FakeMic(), stt=None, tts=tts,
        nlu=FakeNLU({}), persona=persona, registry=registry, intent_meta=intent_meta(),
    )
    o.standby = False
    return o, persona, tts


def test_a_voiced_skill_is_not_rewritten(config):
    coin = SkillManifest(name="flip_a_coin", description="Flip a coin.", voice="jarvis", origin="learned")
    o, persona, tts = _orch(config, coin)
    asyncio.run(o.handle("flip_a_coin", "flip a coin", 0.9))
    assert tts.said == ["Heads, sir."] and persona.phrased == []


def test_a_skill_in_another_personas_voice_is_rewritten(config):
    coin = SkillManifest(name="flip_a_coin", description="Flip a coin.", voice="plain", origin="learned")
    o, persona, tts = _orch(config, coin)
    asyncio.run(o.handle("flip_a_coin", "flip a coin", 0.9))
    assert persona.phrased == ["Heads, sir."]


def test_an_unvoiced_skill_is_rewritten(config):
    coin = SkillManifest(name="flip_a_coin", description="Flip a coin.", origin="learned")
    o, persona, tts = _orch(config, coin)
    asyncio.run(o.handle("flip_a_coin", "flip a coin", 0.9))
    assert persona.phrased == ["Heads, sir."]


def test_every_builtin_speaks_in_jarvis_voice():
    for info in pkgutil.iter_modules(builtin_pkg.__path__):
        mod = importlib.import_module(f"jarvis.skills.builtin.{info.name}")
        manifest = getattr(mod, "MANIFEST", None)
        if manifest is not None:
            assert manifest.voice == "jarvis", info.name


# -- D: the factory asks for the voice --------------------------------------------------

def test_a_generated_manifest_may_declare_its_voice():
    source = (
        "from jarvis.skills.contract import SkillManifest\n"
        "MANIFEST = SkillManifest(name='roll_a_die', description='Roll a die.', "
        "examples=['roll a die'], voice='jarvis')\n"
        "def run(ctx, **params):\n    return 'A four, sir.'\n"
    )
    assert validate(source, frozenset()).voice == "jarvis"


def test_the_factory_prompt_carries_the_voice(config):
    guide = VoiceGuide(
        name="jarvis", character="You are JARVIS, dry and precise.",
        samples=("Will that be all, sir?", "As you wish, sir."),
    )
    prompt = ClaudeClient._build_prompt(
        SkillSpec(name="roll_a_die", description="roll a die"), None, voice=guide
    )
    assert "You are JARVIS, dry and precise." in prompt
    assert "As you wish, sir." in prompt
    assert 'voice="jarvis"' in prompt


def test_without_a_voice_the_prompt_asks_for_none(config):
    prompt = ClaudeClient._build_prompt(SkillSpec(name="roll_a_die", description="roll a die"), None)
    assert 'voice="' not in prompt


def test_the_voice_guide_comes_from_the_persona(config):
    guide = VoiceGuide.from_persona(Persona.load("jarvis", config))
    assert guide.name == "jarvis"
    assert guide.character and 3 <= len(guide.samples) <= 12


# -- D for canned replies: the persona's own line for an intent ------------------------

class CannedPersona(PersonaSpy):
    def __init__(self, lines):
        super().__init__()
        self.lines = lines

    def line(self, event, default=""):
        return self.lines.get(event, default)


def _canned(config, lines):
    from jarvis.core.orchestrator import Orchestrator
    from jarvis.nlu.corpus import intent_meta

    persona, tts = CannedPersona(lines), TTS()
    o = Orchestrator(
        config=config, wake=FakeWake(), mic=FakeMic(), stt=None, tts=tts,
        nlu=FakeNLU({}), persona=persona, registry=FakeRegistry(), intent_meta=intent_meta(),
    )
    o.standby = False
    return o, persona, tts


def test_a_canned_reply_in_the_personas_own_words_is_spoken_as_written(config):
    o, persona, tts = _canned(config, {"reply_thanks": "A pleasure, sir."})
    asyncio.run(o.handle("thanks", "thanks", 0.9))
    assert tts.said == ["A pleasure, sir."] and persona.phrased == []


def test_without_one_the_seed_reply_gets_the_checked_rewrite(config):
    o, persona, tts = _canned(config, {})
    asyncio.run(o.handle("thanks", "thanks", 0.9))
    assert len(persona.phrased) == 1  # one of intents.json's replies


def test_the_jarvis_persona_thanks_in_its_own_words(config):
    assert "sir" in Persona.load("jarvis", config).line("reply_thanks", "").lower()
