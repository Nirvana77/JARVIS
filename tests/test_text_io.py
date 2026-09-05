"""Text-mode audio stand-ins (`jarvis/audio/text_io.py`) and the `text`
subcommand's wiring (`jarvis.app.build_text_orchestrator`) — the rig used to
drive the real pipeline (real NLU/registry/persona/skills) with scripted text
instead of a mic, for end-to-end scenario testing without audio hardware."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from jarvis.audio.text_io import TextIO, TextTTS, TextWake


def test_text_wake_always_triggers():
    wake = TextWake()
    wake.reset()
    assert wake.triggered(None) is True


def test_text_io_round_trips_a_line():
    io = TextIO(["search black holes"])
    audio = io.record_utterance(8.0, 1.0, 2.0)
    assert len(audio) > 0
    assert io.transcribe(audio) == "search black holes"
    # consumed — a second transcribe() with no new record_utterance is empty
    assert io.transcribe(audio) == ""


def test_text_io_exhaustion_calls_on_exhausted():
    calls = []
    io = TextIO(["only line"], on_exhausted=lambda: calls.append(True))
    io.record_utterance(8.0, 1.0, 2.0)  # consumes the only line
    audio = io.record_utterance(8.0, 1.0, 2.0)  # now exhausted
    assert len(audio) == 0
    assert calls == [True]


def test_text_io_empty_deque_reads_as_silence_without_lines():
    io = TextIO([])
    audio = io.record_utterance(8.0, 1.0, 2.0)
    assert len(audio) == 0


def test_text_tts_prints(capsys):
    TextTTS().say("hello, sir")
    assert "Jarvis: hello, sir" in capsys.readouterr().out


# -- full pipeline, real NLU/registry/persona, scripted text -------------------

def test_build_text_orchestrator_runs_a_scripted_conversation(config, capsys):
    from jarvis.app import build_text_orchestrator

    orch = build_text_orchestrator(config, lines=["search black holes", "shut down"])
    asyncio.run(orch.run())

    out = capsys.readouterr().out
    assert 'heard   : "search black holes"' in out
    assert "intent  : search" in out
    assert "Shutting down" in out or "Powering down" in out
    assert orch.running is False


def test_build_text_orchestrator_self_terminates_without_shutdown(config):
    from jarvis.app import build_text_orchestrator

    orch = build_text_orchestrator(config, lines=["search cats"])
    # no "shut down" in the script — must still terminate via on_exhausted,
    # not hang forever re-waking (TextWake fires instantly every time)
    asyncio.run(asyncio.wait_for(orch.run(), timeout=15.0))
    assert orch.running is False
