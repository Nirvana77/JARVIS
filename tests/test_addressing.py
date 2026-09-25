"""Addressing (M3 decisions 5 and 6) — what reaches JARVIS on the remote path,
and when it is considered finished being said.

There is no wake word on the remote path: the addressing modes are the wake
word. Two properties are load-bearing and have their own tests below:

* the mode commands are matched in **every** mode, including Ignore, and
  **before** the gate — so there is no state the user can reach that they
  cannot speak their way out of;
* the hold window counts **silence**, not clock.
"""

from __future__ import annotations

import json

import pytest

from jarvis.remote.addressing import (
    DEFAULT_MODE,
    MODE_LABEL,
    NAMES,
    SPEAKING_CAP_MS,
    DeviceModes,
    HoldWindow,
    Mode,
    apply_mode_command,
    is_addressed,
    match_mic_command,
    match_mode_command,
    route,
    strip_address,
)


class FakeClock:
    """Decision 13: the hold window is tested with a fake clock, never a sleep."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# -- the name --------------------------------------------------------------


@pytest.mark.parametrize(
    "said,rest",
    [
        ("Jarvis, search black holes", "search black holes"),
        ("jarvis search black holes", "search black holes"),
        ("Hey Jarvis, search black holes", "search black holes"),
        ("hey jarvis. search black holes", "search black holes"),
        ("JARVIS — search black holes", "search black holes"),
        ("Jarvis: search black holes", "search black holes"),
        ("  Jarvis,   search black holes  ", "search black holes"),
        ("Jarvis", ""),               # a call, with nothing after it
        ("Jarvis?", ""),
    ],
)
def test_the_name_is_recognised_through_whatever_the_stt_wrote(said, rest):
    assert strip_address(said, NAMES) == rest
    assert is_addressed(said, NAMES) is True


@pytest.mark.parametrize(
    "said",
    [
        "search black holes",
        "tell jarvis about it",          # the name, but not as an address
        "jarvison is a name",            # word boundary: no half-name matches
        "",
        "   ",
    ],
)
def test_an_utterance_that_does_not_open_with_the_name_is_not_addressed(said):
    assert strip_address(said, NAMES) is None
    assert is_addressed(said, NAMES) is False


def test_stripping_keeps_the_original_casing_and_punctuation_of_the_rest():
    assert strip_address("Jarvis, open GitHub — the PR page", NAMES) == "open GitHub — the PR page"


# -- the gate --------------------------------------------------------------


def test_byname_admits_only_what_names_jarvis():
    named = route("Jarvis, search black holes", mode=Mode.BYNAME)
    assert named.kind == "jarvis"
    assert named.text == "search black holes"
    assert named.addressed is True

    ambient = route("so then I told him it was fine", mode=Mode.BYNAME)
    assert ambient.kind == "dropped"
    assert ambient.reason == "unaddressed"


def test_byname_admits_anything_inside_the_conversation_window():
    """After a reply, and during an `_ask` prompt, the next utterance needs no
    name — the orchestrator's existing follow-up window, driven by transcripts."""
    out = route("open github", mode=Mode.BYNAME, conversation=True)
    assert out.kind == "jarvis" and out.text == "open github"
    assert out.addressed is False


def test_always_admits_everything():
    out = route("open github", mode=Mode.ALWAYS)
    assert out.kind == "jarvis" and out.text == "open github" and out.addressed is False
    # and the name is still stripped when it is used
    assert route("Jarvis, open github", mode=Mode.ALWAYS).text == "open github"


def test_pushtotalk_admits_what_is_said_while_the_button_is_held():
    """The hold *is* the address, so what reaches the gate during a hold is
    admitted exactly as Always admits it — the mic being shut the rest of the
    time is the edge's job, and nothing here may second-guess it."""
    out = route("open github", mode=Mode.PUSHTOTALK)
    assert out.kind == "jarvis" and out.text == "open github"
    assert route("Jarvis, open github", mode=Mode.PUSHTOTALK).text == "open github"


def test_ignore_admits_nothing():
    assert route("Jarvis, open github", mode=Mode.IGNORE).kind == "dropped"
    assert route("open github", mode=Mode.IGNORE).reason == "paused"


def test_a_bare_call_is_marked_as_one():
    out = route("Jarvis", mode=Mode.BYNAME)
    assert out.kind == "jarvis" and out.bare is True
    assert out.text == ""


def test_empty_text_is_neither_admitted_nor_dropped():
    assert route("", mode=Mode.ALWAYS).kind == "empty"
    assert route("   ", mode=Mode.IGNORE).kind == "empty"


# -- the mode commands -----------------------------------------------------


@pytest.mark.parametrize(
    "said,to",
    [
        ("Jarvis, pause input", Mode.IGNORE),
        ("pause input", Mode.IGNORE),
        ("Jarvis, pause the input.", Mode.IGNORE),
        ("stop listening", Mode.IGNORE),
        ("Jarvis, continue input", "previous"),
        ("continue the input", "previous"),
        ("resume input", "previous"),
        ("start listening", "previous"),
        ("Jarvis, change input to always", Mode.ALWAYS),
        ("change the input to always", Mode.ALWAYS),
        ("switch to always", Mode.ALWAYS),
        ("change input to by name", Mode.BYNAME),
        ("change mode to byname", Mode.BYNAME),
        ("Jarvis, change input to push to talk", Mode.PUSHTOTALK),
        ("change input to push-to-talk", Mode.PUSHTOTALK),
        ("always mode", Mode.ALWAYS),
    ],
)
def test_the_mode_commands_are_heard_however_they_are_said(said, to):
    assert match_mode_command(said) == to


@pytest.mark.parametrize(
    "said",
    [
        "always",                                   # a bare target is an ordinary word
        "by name",
        "tell me about push to talk",               # discussing it is not doing it
        "change input to always when you can",      # a command is the whole utterance
        "search for pause input",
    ],
)
def test_a_sentence_about_the_modes_is_not_a_mode_command(said):
    assert match_mode_command(said) is None


@pytest.mark.parametrize("mode", [Mode.IGNORE, Mode.BYNAME, Mode.ALWAYS, Mode.PUSHTOTALK])
def test_mode_commands_are_matched_before_the_gate_in_every_mode(mode):
    """A safety property, not a convenience: there must be no state the user
    can reach from which they cannot speak their way out."""
    out = route("Jarvis, change input to always", mode=mode)
    assert out.kind == "mode"
    assert out.to == Mode.ALWAYS
    # and, crucially, while paused
    assert route("continue input", mode=Mode.IGNORE).kind == "mode"


def test_pause_then_continue_restores_push_to_talk():
    """Resuming must never open a mic that push-to-talk had closed."""
    mode, previous = Mode.PUSHTOTALK, None
    mode, previous = apply_mode_command(Mode.IGNORE, mode, previous)
    assert mode == Mode.IGNORE and previous == Mode.PUSHTOTALK
    mode, previous = apply_mode_command("previous", mode, previous)
    assert mode == Mode.PUSHTOTALK
    assert previous is None


def test_continue_from_a_pause_that_was_never_set_falls_back_to_the_default():
    mode, previous = apply_mode_command("previous", Mode.IGNORE, None)
    assert mode == DEFAULT_MODE


def test_pausing_twice_still_remembers_where_it_came_from():
    mode, previous = apply_mode_command(Mode.IGNORE, Mode.ALWAYS, None)
    mode, previous = apply_mode_command(Mode.IGNORE, mode, previous)
    assert previous == Mode.ALWAYS
    assert apply_mode_command("previous", mode, previous)[0] == Mode.ALWAYS


def test_choosing_a_mode_on_purpose_leaves_nothing_to_come_back_to():
    mode, previous = apply_mode_command(Mode.ALWAYS, Mode.BYNAME, None)
    assert (mode, previous) == (Mode.ALWAYS, None)


def test_every_mode_has_a_short_label_the_user_hears_back():
    assert set(MODE_LABEL) == {Mode.IGNORE, Mode.BYNAME, Mode.ALWAYS, Mode.PUSHTOTALK}
    assert MODE_LABEL[Mode.IGNORE] == "paused"
    assert all(label and len(label) < 20 for label in MODE_LABEL.values())


# -- the microphone switch -------------------------------------------------


@pytest.mark.parametrize(
    "said,on",
    [
        ("turn on the mic", True),
        ("Jarvis, turn the microphone on", True),
        ("mic on", True),
        ("turn off the mic", False),
        ("turn the microphone off", False),
        ("mic off", False),
    ],
)
def test_the_microphone_switch(said, on):
    assert match_mic_command(said) == {"on": on}


def test_the_microphone_switch_also_works_from_every_mode():
    out = route("turn on the mic", mode=Mode.IGNORE)
    assert out.kind == "mic" and out.on is True


def test_an_ordinary_sentence_is_not_the_microphone_switch():
    assert match_mic_command("the mic is on the desk") is None
    assert match_mic_command("search for microphones") is None


# -- the hold window -------------------------------------------------------


def test_two_fragments_within_the_window_become_one_utterance():
    clock = FakeClock()
    window = HoldWindow(hold_ms=2000, clock=clock)

    window.add("search black holes")
    clock.advance(1.0)
    assert window.due_in() == pytest.approx(1.0)
    window.add("and neutron stars")
    clock.advance(1.0)
    assert window.due_in() == pytest.approx(1.0), "the second fragment restarts the full window"
    clock.advance(1.0)
    assert window.due_in() == 0
    assert window.release() == "search black holes and neutron stars"
    assert window.held is False
    assert window.due_in() is None


def test_a_fragment_after_the_window_stands_on_its_own():
    clock = FakeClock()
    window = HoldWindow(hold_ms=2000, clock=clock)
    window.add("search black holes")
    clock.advance(2.5)
    assert window.due_in() == 0
    assert window.release() == "search black holes"
    window.add("open github")
    assert window.release() == "open github"


def test_speaking_pauses_the_countdown():
    """While the edge reports `speaking:on` the hold doesn't count down: what is
    measured is the pause between the user's words, not the time whisper spent
    on the first half of the sentence."""
    clock = FakeClock()
    window = HoldWindow(hold_ms=2000, clock=clock)
    window.add("search black holes")
    window.speaking(True)
    clock.advance(5.0)
    assert window.due_in() > 0, "still talking — nothing is released"
    window.speaking(False)
    assert window.due_in() == pytest.approx(2.0), "the full window restarts when they stop"
    clock.advance(2.0)
    assert window.release() == "search black holes"


def test_the_speaking_flag_is_remembered_before_there_is_a_window():
    """It usually arrives *before* the window exists: the user starts the second
    half while the first is still in Whisper."""
    clock = FakeClock()
    window = HoldWindow(hold_ms=2000, clock=clock)
    window.speaking(True)
    window.add("search black holes")
    clock.advance(4.0)
    assert window.due_in() > 0


def test_a_detector_stuck_open_cannot_hold_a_turn_for_ever():
    clock = FakeClock()
    window = HoldWindow(hold_ms=2000, clock=clock)
    window.add("search black holes")
    window.speaking(True)
    clock.advance(SPEAKING_CAP_MS / 1000 + 1)
    assert window.due_in() == 0
    assert window.release() == "search black holes"


def test_the_cap_runs_from_the_last_words_that_actually_arrived():
    clock = FakeClock()
    window = HoldWindow(hold_ms=2000, clock=clock)
    window.speaking(True)
    window.add("first")
    clock.advance(15.0)
    window.add("second")           # real words: the cap re-arms from here
    clock.advance(10.0)
    # 25 s since the first fragment, but only 10 since the last: still held
    assert window.due_in() > 0
    clock.advance(11.0)
    assert window.due_in() == 0
    assert window.release() == "first second"


def test_a_disconnect_clears_what_was_held_and_the_flag():
    clock = FakeClock()
    window = HoldWindow(hold_ms=2000, clock=clock)
    window.add("search black holes")
    window.speaking(True)
    window.clear()
    assert window.held is False
    assert window.due_in() is None
    assert window.release() is None
    # the flag went with it: the next fragment counts down normally
    window.add("open github")
    assert window.due_in() == pytest.approx(2.0)


def test_a_window_of_zero_turns_holding_off():
    clock = FakeClock()
    window = HoldWindow(hold_ms=0, clock=clock)
    window.add("search black holes")
    assert window.due_in() == 0
    assert window.release() == "search black holes"


def test_fragments_are_joined_with_one_space_and_nothing_cleverer():
    clock = FakeClock()
    window = HoldWindow(hold_ms=2000, clock=clock)
    window.add("  search black holes ")
    window.add("and neutron stars")
    assert window.release() == "search black holes and neutron stars"


# -- the mode is per device, and it survives a restart ---------------------


def test_the_mode_is_remembered_per_device(tmp_path):
    modes = DeviceModes(tmp_path, default=Mode.BYNAME)
    assert modes.get("livingroom").mode == Mode.BYNAME

    modes.set("livingroom", Mode.ALWAYS, None)
    modes.set("kitchen", Mode.PUSHTOTALK, None)
    assert modes.get("livingroom").mode == Mode.ALWAYS
    assert modes.get("kitchen").mode == Mode.PUSHTOTALK

    # a fresh brain, same directory
    again = DeviceModes(tmp_path, default=Mode.BYNAME)
    assert again.get("livingroom").mode == Mode.ALWAYS
    assert again.get("kitchen").mode == Mode.PUSHTOTALK
    assert json.loads((tmp_path / "livingroom.json").read_text())["mode"] == Mode.ALWAYS


def test_a_pause_that_outlives_a_restart_still_knows_where_to_go_back_to(tmp_path):
    modes = DeviceModes(tmp_path, default=Mode.BYNAME)
    modes.set("livingroom", Mode.IGNORE, Mode.PUSHTOTALK)
    state = DeviceModes(tmp_path, default=Mode.BYNAME).get("livingroom")
    assert state.mode == Mode.IGNORE
    assert state.previous_mode == Mode.PUSHTOTALK


def test_a_corrupt_or_hostile_device_file_falls_back_to_the_default(tmp_path):
    (tmp_path / "livingroom.json").write_text("{not json")
    (tmp_path / "kitchen.json").write_text('{"mode": "rm -rf /"}')
    modes = DeviceModes(tmp_path, default=Mode.BYNAME)
    assert modes.get("livingroom").mode == Mode.BYNAME
    assert modes.get("kitchen").mode == Mode.BYNAME
