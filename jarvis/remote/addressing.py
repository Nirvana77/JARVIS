"""Addressing and the hold window (M3 decisions 5 and 6).

On the remote path there is no wake word: *the addressing modes are the wake
word*. Every transcript is classified here before anything else sees it, and the
classification is a rule rather than a model call — a classifier in this path
would add latency to every turn and be wrong occasionally, which is worse than a
rule that is wrong never and that the user controls by how they speak.

Two layers, in this order, and **the order is the point**:

1. **The mode commands, matched in every mode including Ignore, before the
   gate.** This is a safety property, not a convenience: there must be no state
   the user can reach from which they cannot speak their way out.
2. The mode gate.

=============  ==========================================================
mode           what reaches JARVIS
=============  ==========================================================
``byname``     only utterances starting with "Jarvis" (the name is
               stripped), plus anything inside a conversation window
``always``     everything said
``pushtotalk`` only what is said while the button is held — the hold *is*
               the address, and the mic is off the rest of the time
``ignore``     nothing (input paused); the edge still listens and discards
=============  ==========================================================

The hold window (decision 6) is the other half: the segmenter closes after
700 ms of silence, which is right for "has the person stopped talking" and wrong
for "has the person finished the thought". So addressed fragments are held for
``hold_ms`` and merged — and the countdown is **silence, not clock**.

Ported from Mike's ``src/routing.js`` and the hold window in ``src/handler.js``,
with the Swedish dropped and the worker routing replaced by JARVIS's single
destination.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


class Mode:
    IGNORE = "ignore"
    BYNAME = "byname"
    ALWAYS = "always"
    PUSHTOTALK = "pushtotalk"


MODES = (Mode.IGNORE, Mode.BYNAME, Mode.ALWAYS, Mode.PUSHTOTALK)
DEFAULT_MODE = Mode.BYNAME

#: How the user hears each mode named back. Short — it is spoken.
MODE_LABEL = {
    Mode.IGNORE: "paused",
    Mode.BYNAME: "by name",
    Mode.ALWAYS: "always",
    # Two words, because "PTT" spoken aloud is a thing nobody has ever
    # understood the first time.
    Mode.PUSHTOTALK: "hold to talk",
}

#: The names JARVIS answers to. ``[addressing] names`` overrides this.
NAMES = ("jarvis", "hey jarvis")


def is_mode(value) -> bool:
    return value in MODES


# -- folding ---------------------------------------------------------------
#
# Matching happens on a folded form (casing, accents and punctuation gone) but
# the *stripped* text has to come back in the user's own spelling, so the fold
# carries an index from each folded character back to the original offset.


def _fold_with_index(raw: str) -> tuple[str, list[int], str]:
    src = str(raw or "")
    folded: list[str] = []
    index: list[int] = []
    pending_space = False
    for i, char in enumerate(src):
        # Per character, so one source offset maps to one folded character:
        # normalising the whole string first would shift every later index.
        plain = "".join(
            c for c in unicodedata.normalize("NFD", char)
            if not unicodedata.combining(c)
        ).lower()
        if plain and plain.isalnum() and plain.isascii():
            if pending_space and folded:
                folded.append(" ")
                index.append(i)
            pending_space = False
            for c in plain:
                folded.append(c)
                index.append(i)
        else:
            pending_space = True
    return "".join(folded), index, src


def _original_offset(index: list[int], src: str, n: int) -> int:
    """Where in the original text the folded prefix of length ``n`` ends."""
    return len(src) if n >= len(index) else index[n]


def fold(raw: str) -> str:
    return _fold_with_index(raw)[0]


#: Leading words dictation and politeness add before a name.
_GREETING = r"(?:hey|hi|hello|ok|okay|yo|so)"
_LEADING_PUNCT = re.compile(r"^[\s,.:;!?—–-]+")


def strip_address(text: str, names=NAMES) -> str | None:
    """If ``text`` is addressed to one of ``names``, return it with the address
    removed; ``None`` when it is not addressed at all.

    Dictation variants are the whole job: "Jarvis." / "Jarvis," / "jarvis" /
    "Hey Jarvis" / "JARVIS:" all count, because the check runs on the folded
    form and the answer is mapped back to the original text.
    """
    folded, index, src = _fold_with_index(text)
    if not folded:
        return None
    for name in sorted((fold(n) for n in names), key=len, reverse=True):
        if not name:
            continue
        # The name must be followed by a word boundary, or "jarvis" would
        # swallow the first half of "jarvison".
        match = re.match(rf"(?:{_GREETING} )?{re.escape(name)}(?= |$)", folded)
        if not match:
            continue
        rest = src[_original_offset(index, src, len(match.group(0))):]
        return _LEADING_PUNCT.sub("", rest).strip()
    return None


def is_addressed(text: str, names=NAMES) -> bool:
    return strip_address(text, names) is not None


# -- the mode commands -----------------------------------------------------
#
# Matched on the folded text, so casing and trailing full stops are already
# gone. Anchored end to end: a command is the whole utterance, never a phrase
# inside a sentence, or discussing the feature would trigger it.

_VERB = r"(?:change|set|switch|put)"
# The words for the thing being changed and the words for what to change it to,
# as two lists rather than a dozen literal sentences — the sentences are the
# cross product, and a user who says "change mode to always" instead of "change
# input to always" is not making a mistake.
_NOUN = r"(?:input|listening|mode)"
_TO = r"(?:to)"
_GO = rf"(?:{_VERB}|go)"

_TARGETS = {
    # A trailing particle is allowed, and only these: "always ON" is how the
    # mode is actually said out loud. Allowing any trailing word would throw
    # away the anchoring that keeps "switch to always when you're debugging"
    # out of the command path, and that anchoring is worth more than reach.
    Mode.ALWAYS: r"(?:always)(?: on)?",
    Mode.BYNAME: r"(?:by ?name)",
    Mode.PUSHTOTALK: r"(?:push ?to ?talk|hold to talk|press to talk)",
}

#: Every way of saying "make it X" that is still unambiguous. A bare target is
#: deliberately *not* one of them: "always" and "by name" are ordinary words,
#: and these are matched before the gate, so a false positive costs the user a
#: sentence they have to say again. Either a verb or the noun has to be there —
#: someone who says only "always" gets their word passed through, which is the
#: safe way to be wrong.
_MODE_COMMANDS: list[tuple[str, re.Pattern]] = [
    (Mode.IGNORE, re.compile(rf"^(?:{_VERB} )?(?:pause|stop|mute) (?:the )?{_NOUN}$")),
    (Mode.IGNORE, re.compile(r"^(?:stop|quit) listening$")),
    ("previous", re.compile(rf"^(?:{_VERB} )?(?:continue|resume|unpause) (?:the )?{_NOUN}$")),
    ("previous", re.compile(r"^(?:start|continue) listening$")),
]
for _mode, _target in _TARGETS.items():
    _MODE_COMMANDS += [
        # "change the input to always", "switch to always", "go to by name"
        (_mode, re.compile(rf"^{_GO} (?:the )?(?:{_NOUN} )?(?:{_TO} )?{_target}$")),
        # "always mode" — the target naming the noun directly
        (_mode, re.compile(rf"^{_target} ?{_NOUN}$")),
    ]

#: Turning the microphone on and off by voice.
#:
#: The microphone switch is not the addressing mode — it is whether there is a
#: microphone for a mode to listen with — but it needs the same guarantee, for a
#: sharper reason: "turn off the mic" leaves the user where nothing they say can
#: be heard at all, so the way back has to be one they can reach. The noun list
#: is deliberately only the microphone's names; a broader word would catch a
#: sentence meant for a skill, and these are matched before everything.
_MIC = r"(?:mic|mike|microphone)"
_MIC_COMMANDS: list[tuple[bool, re.Pattern]] = [
    (True, re.compile(rf"^(?:{_VERB} |turn )?on (?:the |my )?{_MIC}$")),
    (True, re.compile(rf"^(?:turn )?(?:the |my )?{_MIC} on$")),
    (False, re.compile(rf"^(?:{_VERB} |turn )?off (?:the |my )?{_MIC}$")),
    (False, re.compile(rf"^(?:turn )?(?:the |my )?{_MIC} off$")),
]


def _candidates(text: str, names) -> list[str]:
    """The utterance, and the utterance with the name taken off the front.

    Every command below is written as "Jarvis, ..." in the PRD, but a bare
    "pause input" has to work too: the user who is paused has just learned that
    nothing they say is getting through and will start dropping words.
    """
    out = [text]
    bare = strip_address(text, names)
    if bare is not None:
        out.append(bare)
    return out


def match_mode_command(text: str, names=NAMES) -> str | None:
    """A mode, or ``"previous"``, or ``None``."""
    for candidate in _candidates(text, names):
        folded = fold(candidate)
        if not folded:
            continue
        for to, pattern in _MODE_COMMANDS:
            if pattern.match(folded):
                return to
    return None


def match_mic_command(text: str, names=NAMES) -> dict | None:
    """``{"on": bool}`` or ``None``."""
    for candidate in _candidates(text, names):
        folded = fold(candidate)
        if not folded:
            continue
        for on, pattern in _MIC_COMMANDS:
            if pattern.match(folded):
                return {"on": on}
    return None


def apply_mode_command(to: str, mode: str, previous_mode: str | None) -> tuple[str, str | None]:
    """Apply a mode command. Returns ``(mode, previous_mode)``.

    "Continue" restores what was in use before the pause rather than a fixed
    default: pausing while in push-to-talk and resuming into ByName would
    silently open a microphone the user had closed.
    """
    if to == "previous":
        back = previous_mode if is_mode(previous_mode) and previous_mode != Mode.IGNORE else DEFAULT_MODE
        return back, None
    if not is_mode(to):
        return mode, previous_mode
    if to != Mode.IGNORE:
        # Nothing to come back to from a mode the user chose on purpose.
        return to, None
    # Only a pause remembers where it came from — and pausing twice must not
    # forget the first answer.
    return Mode.IGNORE, previous_mode if mode == Mode.IGNORE else mode


# -- routing ---------------------------------------------------------------


@dataclass(frozen=True)
class Routed:
    """What one utterance turned out to be.

    ``kind`` is one of ``empty``, ``mode``, ``mic``, ``jarvis`` or ``dropped``.
    Nothing here mutates anything: the caller owns the device's state, and a
    classifier that changed state would be impossible to test one utterance at
    a time.
    """

    kind: str
    text: str = ""
    #: kind="jarvis": whether the user actually said the name
    addressed: bool = False
    #: kind="jarvis": the name with nothing after it — a call, not a command
    bare: bool = False
    #: kind="mode": a mode, or "previous"
    to: str | None = None
    #: kind="mic"
    on: bool | None = None
    #: kind="dropped": "paused" or "unaddressed"
    reason: str | None = None


def route(
    text: str,
    *,
    mode: str = DEFAULT_MODE,
    names=NAMES,
    conversation: bool = False,
) -> Routed:
    """Classify one transcript. ``conversation`` is the orchestrator's follow-up
    window (or an ``_ask`` prompt) being open, which lets ByName accept the next
    utterance without the name — it never applies in Ignore or PushToTalk."""
    raw = (text or "").strip()
    if not raw:
        return Routed("empty")

    # 1. The mode commands, before the gate, in every mode.
    to = match_mode_command(raw, names)
    if to is not None:
        return Routed("mode", to=to)

    # The microphone switch, on the same footing and for the same reason: the
    # state it can put the user in is one they must be able to speak out of.
    mic = match_mic_command(raw, names)
    if mic is not None:
        return Routed("mic", on=mic["on"])

    # 2. The gate.
    if mode == Mode.IGNORE:
        return Routed("dropped", reason="paused")

    stripped = strip_address(raw, names)
    if stripped is not None:
        # The name on its own is a call with nothing after it. Marked as such so
        # the caller can answer at once ("yes?") rather than send an empty
        # command down the NLU.
        return Routed("jarvis", text=stripped, addressed=True, bare=not stripped)

    if mode == Mode.BYNAME and not conversation:
        return Routed("dropped", reason="unaddressed")

    # Always, PushToTalk (the hold is the address), or a conversation window.
    return Routed("jarvis", text=raw, addressed=False)


# -- the hold window -------------------------------------------------------

#: Long enough to cross a thinking pause, short enough not to feel like a hang.
DEFAULT_HOLD_MS = 2000

#: How long a window may stay open past the last words that *actually arrived*,
#: whatever the `speaking` signal says. Mike's first version re-armed the cap on
#: every flip of the signal, and a detector that flipped every fifteen seconds —
#: open on room tone to its maximum, closed for a frame, open again — kept
#: "still listening" on the display for as long as it liked. Measured from the
#: last real fragment instead, the cap is a promise: nothing said is held longer
#: than this. The edge's own maximum segment is 15 s, so a genuine long
#: continuation still fits.
SPEAKING_CAP_MS = 20_000


class HoldWindow:
    """Merge the fragments of one thought into one utterance.

    The countdown is **silence, not clock**. While the edge reports
    ``speaking:on`` the window does not count down at all; when the next
    transcript arrives, the full ``hold_ms`` restarts. What is measured is the
    pause between the user's words — not the time Whisper spent on the first
    half of the sentence, which is what the first version accidentally measured
    (3 merges in 40 turns, and the one that merged had 25 ms to spare).

    Deliberately not asyncio-aware: it answers "how long until this is due"
    (:meth:`due_in`) and the caller owns the waiting. That makes the whole thing
    testable with a fake clock, and keeps one timer in one place in the server.
    """

    def __init__(
        self,
        hold_ms: int = DEFAULT_HOLD_MS,
        *,
        cap_ms: int = SPEAKING_CAP_MS,
        clock=time.monotonic,
    ) -> None:
        self.hold_ms = hold_ms
        self.cap_ms = cap_ms
        self._clock = clock
        self._parts: list[str] = []
        #: when what is held is due, and the wall past which it is due whatever
        #: the `speaking` signal says
        self._due = 0.0
        self._cap_due = 0.0
        #: the edge's detector is open right now. Kept outside the window
        #: because the signal usually arrives *before* there is one: the user
        #: starts the second half while the first is still in Whisper.
        self._speaking = False

    @property
    def held(self) -> bool:
        return bool(self._parts)

    @property
    def parts(self) -> list[str]:
        return list(self._parts)

    def add(self, text: str) -> None:
        """A fragment arrived. Restarts the countdown and re-arms the cap."""
        text = (text or "").strip()
        if not text:
            return
        self._parts.append(text)
        self._cap_due = self._clock() + self.cap_ms / 1000
        self._arm()

    def speaking(self, on: bool) -> None:
        """The edge's detector opened or closed a segment. Recorded whether or
        not a window is open, because the signal comes first and the fragment it
        belongs to comes a transcription later."""
        self._speaking = bool(on)
        if self._parts:
            self._arm()

    def _arm(self) -> None:
        """Start, or restart, the countdown — unless the user is talking, in
        which case only the cap runs."""
        self._due = self._cap_due if self._speaking else self._clock() + self.hold_ms / 1000

    def due_in(self) -> float | None:
        """Seconds until what is held should be released; ``None`` when there is
        nothing held, ``0`` when it is due now."""
        if not self._parts:
            return None
        return max(0.0, min(self._due, self._cap_due) - self._clock())

    def release(self) -> str | None:
        """The window closed: the fragments become one utterance.

        One space, and nothing cleverer. They are separate because the speaker
        paused, not because they are separate sentences, and punctuation
        invented here is punctuation the NLU reads as meaning something.
        """
        if not self._parts:
            return None
        text = " ".join(self._parts)
        self._parts = []
        return text

    def clear(self) -> None:
        """A disconnect. Drops what was held *and* the speaking flag — a closed
        socket cannot report that it stopped talking."""
        self._parts = []
        self._speaking = False


# -- the mode is per device, and it survives a restart --------------------


@dataclass(frozen=True)
class DeviceState:
    mode: str
    previous_mode: str | None = None


class DeviceModes:
    """``data/remote/<device_id>.json`` — one small file per edge device.

    The mode is per device, not per process: an edge left in push-to-talk stays
    in push-to-talk across a brain restart, because the user set it in a room
    they may not be in now.
    """

    def __init__(self, directory: str | Path, default: str = DEFAULT_MODE) -> None:
        self.dir = Path(directory)
        self.default = default if is_mode(default) else DEFAULT_MODE

    def _path(self, device_id: str) -> Path:
        # device_id is validated by the protocol before it ever gets here
        # (`_DEVICE_ID_RE`), so it cannot escape this directory.
        return self.dir / f"{device_id}.json"

    def get(self, device_id: str) -> DeviceState:
        path = self._path(device_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            mode = raw.get("mode")
            previous = raw.get("previous_mode")
        except (OSError, ValueError, AttributeError):
            return DeviceState(self.default)
        return DeviceState(
            mode if is_mode(mode) else self.default,
            previous if is_mode(previous) else None,
        )

    def set(self, device_id: str, mode: str, previous_mode: str | None = None) -> DeviceState:
        state = DeviceState(mode if is_mode(mode) else self.default, previous_mode)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._path(device_id).write_text(
                json.dumps({"mode": state.mode, "previous_mode": state.previous_mode}),
                encoding="utf-8",
            )
        except OSError as exc:  # a read-only data dir must not end the session
            log.warning("could not save the mode for %s: %s", device_id, exc)
        return state
