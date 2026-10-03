"""M4.5: the turn the NLU could not place — plan it, answer it, or leave it.

M4 (``mishear.py``) asks the reasoner one narrow thing: is this transcript a
mishearing of a known command? When it is not, this asks the wider question —
what does the speaker want? — and takes one of three replies:

  ``{"commands": [...]}``  the request, re-said the way JARVIS's commands are
                           said, in order ("wake me in five and beep the
                           watch" -> "set a timer for 5 minutes", "find my
                           watch")
  ``{"answer": "..."}``    something it can simply say, in character
  ``{"learn": "...", "examples": [...]}``
                           (M7) a request for something JARVIS cannot do yet:
                           what it should learn, and other ways to ask for it
  ``{"none": true}``       nothing to do

This module is the pure part: the prompt, and what is accepted back. As in
M4, nothing here acts. The orchestrator classifies every command with the
real NLU and confirms a plan by voice before any of it runs
(``Orchestrator._think``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

#: steps in one request; a longer list is a model rambling, not a plan
MAX_COMMANDS = 4
#: this is speech: a couple of sentences, not a page
MAX_ANSWER_CHARS = 320
#: how much of the device's memory goes into one prompt, newest kept
MAX_FACTS_IN_PROMPT = 30
#: M7: a skill's description is a short instruction, not a specification
MAX_LEARN_CHARS = 160
#: ... with a few more ways of asking for it
MAX_LEARN_EXAMPLES = 4

_TASK = (
    "You are answering by voice. The speech recogniser heard something that "
    "matched none of your commands. Work out what the speaker wants and reply "
    "with one JSON object and nothing else:\n"
    '{"commands": ["...", "..."]} when they are asking for things your known '
    "commands do. Write each one the way that command is normally said, in the "
    "order to do them, keeping the speaker's own details (names, numbers, "
    "durations). Only when the known commands really cover the request: never "
    "a command that is not in the list.\n"
    '{"answer": "..."} for a question or a remark you can answer yourself: one '
    "or two short spoken sentences, in character, using what you have been "
    "asked to remember and the conversation so far when they help. If you do "
    "not know, say so plainly. Never invent facts about the speaker.\n"
    '{"learn": "...", "examples": ["...", "..."]} when they ask you to DO '
    "something (an action, a calculation, a lookup, a device to control) that "
    "none of your known commands can do and that you cannot simply answer from "
    "what you know: describe the ability as a short instruction starting with "
    'a verb ("roll a die with a given number of sides"), with two or three '
    "other ways a person might ask for it. Not for questions you can answer.\n"
    '{"none": true} if it is noise, half a sentence, or you cannot tell what '
    "they want."
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Thought:
    """What the reasoner made of an unclear turn: exactly one of the two."""

    answer: str = ""
    commands: tuple[str, ...] = ()
    #: M7: what JARVIS should learn to do, and more ways to ask for it
    learn: str = ""
    examples: tuple[str, ...] = ()


def system(character: str = "") -> str:
    """The persona's character (so an answer is in voice already), then the task."""
    return f"{character.strip()}\n\n{_TASK}" if character.strip() else _TASK


def build_prompt(
    heard: str,
    known: list[str],
    facts: list[str],
    turns: list[tuple[str, str]],
    now: datetime,
    retry: tuple[str, str] | None = None,
) -> str:
    parts = [
        "Known commands (name: ways of saying it):\n" + "\n".join(f"- {line}" for line in known)
    ]
    if facts:
        parts.append(
            "What you have been asked to remember:\n"
            + "\n".join(f"- {fact}" for fact in facts[-MAX_FACTS_IN_PROMPT:])
        )
    if turns:
        parts.append(
            "Earlier in this conversation:\n"
            + "\n".join(f"Speaker: {heard_}\nYou: {said or '(nothing)'}" for heard_, said in turns)
        )
    parts.append(f"Now: {now.strftime('%A')} {now.day} {now.strftime('%B %Y, %H:%M')}")
    if retry is not None:
        before, said = retry
        parts.append(
            f'The speaker is asking again. They asked "{before.strip()}" and you '
            f'answered "{said.strip()}", which was not what they wanted. Do not give '
            "that answer again: answer the question itself, or say what you should "
            "learn to do."
        )
    parts.append(f'The recogniser heard: "{heard.strip()}"')
    return "\n\n".join(parts)


def clip(text: str) -> str:
    """Whole sentences up to ``MAX_ANSWER_CHARS``; a single overlong one is
    cut at a word."""
    text = " ".join(text.split())
    if len(text) <= MAX_ANSWER_CHARS:
        return text
    kept = ""
    for sentence in _SENTENCE_END.split(text):
        if len(kept) + len(sentence) + 1 > MAX_ANSWER_CHARS:
            break
        kept = f"{kept} {sentence}".strip()
    if kept:
        return kept
    return text[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:") + "."


def _json_object(reply: str):
    """The first JSON object in the reply — a small model likes to wrap it in
    a sentence or a code fence even when told not to."""
    start = (reply or "").find("{")
    if start < 0:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(reply[start:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def parse(reply: str) -> Thought | None:
    """The reasoner's reply as a :class:`Thought`, or ``None`` when it
    declined or the reply is not something to act on."""
    obj = _json_object(reply)
    if obj is None:
        return None
    commands = obj.get("commands")
    if commands:  # a model that fills in every key sends [] beside its answer
        if (
            not isinstance(commands, list)
            or not 1 <= len(commands) <= MAX_COMMANDS
            or not all(isinstance(c, str) for c in commands)
        ):
            return None
        cleaned = tuple(" ".join(c.split()).strip(" .!?\"'") for c in commands)
        return Thought(commands=cleaned) if all(cleaned) else None
    answer = obj.get("answer")
    if isinstance(answer, str) and answer.strip():
        return Thought(answer=clip(answer))
    learn = obj.get("learn")
    if isinstance(learn, str) and learn.strip():
        learn = " ".join(learn.split()).strip(" .!?\"'")[:MAX_LEARN_CHARS]
        examples = obj.get("examples")
        examples = examples if isinstance(examples, list) else []
        cleaned = tuple(
            " ".join(e.split()).strip(" .!?\"'")
            for e in examples
            if isinstance(e, str) and e.strip()
        )[:MAX_LEARN_EXAMPLES]
        return Thought(learn=learn, examples=cleaned) if learn else None
    return None
