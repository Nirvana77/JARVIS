""""What should I remember today?" — from the watch, 2026-10-02.

"What should I remember today?" and "Is there anything that I should remember
today?" were heard as `remember` and *stored*; "What are the to-dos for
today?" was `unknown`. They are questions about what the speaker asked JARVIS
to remember, so they are `recall_memory` — and "today" reads back what was
remembered today, or everything when nothing was.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import pytest

from jarvis.core.memory import Memory
from jarvis.nlu import slots
from jarvis.nlu.corpus import intent_meta
from tests.test_reasoning import make_orch


@pytest.mark.parametrize(
    "text",
    ["What should I remember today?", "Is there anything that I should remember today?",
     "What are the to-dos for today?"],
)
def test_a_question_about_remembering_is_never_stored(tmp_path, text):
    assert not slots.has_lead_in("remember", text)
    memory = Memory(tmp_path)
    orch = make_orch(memory=memory)
    asyncio.run(orch.handle("remember", text, 0.6))
    assert memory.device("local").facts() == []


def _memory_with(tmp_path, entries):
    """``entries``: (text, days ago)."""
    now = datetime.now()
    body = {"facts": [{"text": t, "at": (now - timedelta(days=d)).isoformat(timespec="seconds")}
                      for t, d in entries]}
    (tmp_path / "local.json").write_text(json.dumps(body), encoding="utf-8")
    return Memory(tmp_path)


def test_today_reads_back_what_was_remembered_today(tmp_path):
    memory = _memory_with(tmp_path, [("the gate code is 7731", 3),
                                     ("I need to pick up Alva at 2", 0),
                                     ("buy milk", 0)])
    orch = make_orch(memory=memory)
    asyncio.run(orch.handle("recall_memory", "What should I remember today?", 0.9))
    (said,) = orch._persona.spoken
    assert "pick up Alva at 2" in said and "buy milk" in said
    assert "gate code" not in said
    assert said.lower().startswith("today")


def test_today_with_nothing_from_today_reads_back_everything(tmp_path):
    memory = _memory_with(tmp_path, [("the gate code is 7731", 3)])
    orch = make_orch(memory=memory)
    asyncio.run(orch.handle("recall_memory", "What are the to-dos for today?", 0.9))
    (said,) = orch._persona.spoken
    assert "gate code" in said
    assert "nothing from today" in said.lower()


def test_without_today_it_is_everything_as_before(tmp_path):
    memory = _memory_with(tmp_path, [("the gate code is 7731", 3), ("buy milk", 0)])
    orch = make_orch(memory=memory)
    asyncio.run(orch.handle("recall_memory", "what do you remember", 0.9))
    (said,) = orch._persona.spoken
    assert "gate code" in said and "buy milk" in said


def test_the_questions_are_in_the_seed():
    patterns = [p.lower() for p in intent_meta()["recall_memory"].patterns]
    for p in ("what should i remember today", "is there anything i should remember",
              "what are the to-dos for today"):
        assert p in patterns, p
