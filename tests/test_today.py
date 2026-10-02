"""More from the watch, 2026-10-02 (second session).

- "Do you have any reminders for me today?", "Do I have anything today at
  2?", "What is the note for today?" were unknown or went to the knowledge
  base, which has no dates: they are `recall_memory`, about today.
- "Remind me about the album at 2 today." was unknown: it is `remember`.
- "Remove our notes from today." found nothing and "Forget everything about
  today" matched only facts with the *word* "today": "today" means what was
  remembered today.
- "Remove the parking note." offered three of four matches and left one.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import pytest

from jarvis.core import forgetting
from jarvis.core.memory import Memory
from jarvis.nlu import slots
from jarvis.nlu.corpus import intent_meta
from tests.test_reasoning import make_orch


def _memory_with(tmp_path, entries):
    now = datetime.now()
    body = {"facts": [{"text": t, "at": (now - timedelta(days=d)).isoformat(timespec="seconds")}
                      for t, d in entries]}
    (tmp_path / "local.json").write_text(json.dumps(body), encoding="utf-8")
    return Memory(tmp_path)


@pytest.mark.parametrize("text", ["Remove our notes from today.", "Forget everything about today",
                                  "forget today's notes", "delete what I told you today"])
def test_forgetting_today_means_what_was_remembered_today(tmp_path, text):
    memory = _memory_with(tmp_path, [("the gate code is 7731", 3),
                                     ("I need to pick up Alva at 2", 0), ("buy milk", 0)])
    orch = make_orch(memory=memory, answers=["yes"])
    asyncio.run(orch.handle("forget_fact", text, 0.6))
    assert orch._persona.spoken[0] == "<forget_fact_confirm I need to pick up Alva at 2; buy milk>"
    assert memory.device("local").facts() == ["the gate code is 7731"]


def test_today_and_a_subject_narrows_to_today_s_facts_about_it(tmp_path):
    memory = _memory_with(tmp_path, [("buy milk", 3), ("I need to pick up Alva at 2", 0),
                                     ("buy milk", 0)])
    orch = make_orch(memory=memory, answers=["no"])
    asyncio.run(orch.handle("forget_fact", "forget the milk note from today", 0.9))
    assert orch._persona.spoken[0] == "<forget_fact_confirm buy milk>"


def test_more_than_three_matches_are_offered():
    facts = ["I parked on level 2", "Remove the park note", "Forget about the park",
             "Forget everything about the parking"]
    assert forgetting.matching("the parking", facts) == facts


@pytest.mark.parametrize("text", ["remind me about the album at 2 today",
                                  "remind me to call the dentist", "remind me that the bins go out"])
def test_remind_me_is_a_lead_in(text):
    assert slots.has_lead_in("remember", text)


def test_remind_me_keeps_what_follows():
    assert slots.extract("remember", "Remind me about the album at 2 today.") == {
        "text": "the album at 2 today"}
    assert slots.extract("remember", "remind me to call the dentist") == {
        "text": "call the dentist"}


def test_the_new_questions_are_in_the_seed():
    recall = [p.lower() for p in intent_meta()["recall_memory"].patterns]
    for p in ("do you have any reminders for me today", "do i have anything today",
              "what is the note for today", "is there any note for today"):
        assert p in recall, p
    remember = [p.lower() for p in intent_meta()["remember"].patterns]
    assert "remind me about {intent}" in remember
    # not "remind me to {intent}": the watch's set_timer owns "remind me to
    # check the oven in 15 minutes"
    assert "remind me to {intent}" not in remember
