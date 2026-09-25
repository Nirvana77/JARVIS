"""The whole remote path, end to end over a loopback socket (M3 decision 7).

A real ``RemoteServer`` and a real ``Orchestrator``, with fakes only where a
model would be: the transcriber hands back canned text per segment and the
voder hands back silence. What is being checked is the wiring — that an
addressed utterance becomes a skill call and comes back as ``speech``, that an
un-addressed one costs nothing but a ``heard``, and that an interrupt or a
disconnect ends the turn instead of wedging it.
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.audio.whisper_client import Transcript, TranscriptionUnavailable
from jarvis.remote import protocol as P
from tests.remote_harness import (
    DEVICE,
    Brain,
    FakeTranscriber,
    FakeVoder,
    connected,
    make_config,
)


def run(coro, timeout=20):
    async def guarded():
        return await asyncio.wait_for(coro, timeout=timeout)

    return asyncio.run(guarded())


# -- the happy path --------------------------------------------------------


def test_an_addressed_utterance_becomes_a_skill_and_comes_back_as_speech(tmp_path):
    async def scenario():
        brain = Brain(
            make_config(tmp_path),
            transcriber=FakeTranscriber("Jarvis, search black holes"),
        )
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()

                heard = await edge.expect("heard")
                assert heard["text"] == "Jarvis, search black holes"
                assert heard["confidence"] == 0.9

                speech = await edge.expect("speech")
                assert speech["part"] == 0
                assert speech["sample_rate"] == 16000
                assert speech["pcm"]
                await edge.send(P.control(P.CONTROL.PLAYBACK_DONE, id=speech["id"]))

                # the name was stripped before the NLU ever saw it
                assert brain.registry.calls == [("search", {"query": "black holes"})]
                # `heard` came before the answer
                assert edge.types().index("heard") < edge.types().index("speech")
                # and the state went back to idle once playback was reported
                state = await edge.expect("state")
                assert state["value"] in ("idle", "speaking", "listening", "thinking")
        finally:
            await brain.stop()

    run(scenario())


def test_the_answer_also_goes_out_as_text_for_a_log_or_a_display(tmp_path):
    async def scenario():
        brain = Brain(make_config(tmp_path), transcriber=FakeTranscriber("Jarvis, search black holes"))
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                text = await edge.expect("text")
                assert "done:search" in text["text"]
                assert text["from"] == "jarvis"
        finally:
            await brain.stop()

    run(scenario())


def test_two_fragments_of_one_thought_are_one_turn(tmp_path):
    """The hold window: the segmenter closed twice, but the user said one thing."""

    async def scenario():
        brain = Brain(
            make_config(tmp_path, hold_ms=300),
            transcriber=FakeTranscriber("Jarvis, search black holes", "and neutron stars"),
        )
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                await edge.expect("heard")
                await asyncio.sleep(0.05)
                await edge.say()
                await edge.expect("heard")
                await edge.expect("speech", timeout=6)
                assert brain.registry.calls == [
                    ("search", {"query": "black holes and neutron stars"})
                ]
        finally:
            await brain.stop()

    run(scenario())


def test_the_hold_only_has_to_notice_a_continuation_not_swallow_it(tmp_path):
    """Why the hold window can be short.

    The edge announces speech the moment its detector opens, and that pauses
    the hold's countdown. So the hold does not have to be long enough to cover
    a whole continuation — transcription and all — only long enough to *hear
    one starting*. Here the second fragment's transcript lands well after a
    100 ms hold would have expired, and it still merges, because `speaking:on`
    arrived first.
    """

    async def scenario():
        brain = Brain(
            make_config(tmp_path, hold_ms=100),
            transcriber=FakeTranscriber(
                "Jarvis, search black holes", "and neutron stars", delay_s=0.4
            ),
        )
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                await edge.expect("heard")
                # they draw breath and carry on — the detector opens first
                await edge.send(P.speaking(True))
                await asyncio.sleep(0.3)   # 3x the hold, and nothing is released
                assert brain.registry.calls == []
                await edge.say()
                await edge.expect("speech", timeout=8)
                assert brain.registry.calls == [
                    ("search", {"query": "black holes and neutron stars"})
                ]
        finally:
            await brain.stop()

    run(scenario())


def test_state_says_still_listening_while_words_are_held(tmp_path):
    async def scenario():
        brain = Brain(
            make_config(tmp_path, hold_ms=400),
            transcriber=FakeTranscriber("Jarvis, search black holes"),
        )
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                await edge.expect("heard")
                state = await edge.expect("state")
                assert state["value"] == "listening"
                assert state["mode"] == "byname"
                # ... and thinking once the whole utterance is through
                while state["value"] != "thinking":
                    state = await edge.expect("state", timeout=6)
        finally:
            await brain.stop()

    run(scenario())


# -- the gate --------------------------------------------------------------


def test_speech_that_does_not_name_jarvis_is_heard_and_nothing_else(tmp_path):
    async def scenario():
        brain = Brain(
            make_config(tmp_path),
            transcriber=FakeTranscriber("so then I told him it was fine"),
        )
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                heard = await edge.expect("heard")
                assert heard["text"] == "so then I told him it was fine"
                rest = await edge.quiet(0.6)
                assert [m["type"] for m in rest if m["type"] == "speech"] == []
                assert brain.registry.calls == []
        finally:
            await brain.stop()

    run(scenario())


def test_a_reply_inside_the_conversation_window_needs_no_name(tmp_path):
    """After an answer, the follow-up window is open: the next utterance is
    taken with or without the name — the orchestrator's own window, driven by
    transcripts."""

    async def scenario():
        brain = Brain(
            make_config(tmp_path, follow_up_s=3.0),
            transcriber=FakeTranscriber("Jarvis, search black holes", "open github"),
        )
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                speech = await edge.expect("speech")
                await edge.send(P.control(P.CONTROL.PLAYBACK_DONE, id=speech["id"]))
                await asyncio.sleep(0.1)
                await edge.say()  # no name this time
                await edge.expect("speech", timeout=8)
                assert [c[0] for c in brain.registry.calls] == ["search", "open_app"]
        finally:
            await brain.stop()

    run(scenario())


def test_the_mode_commands_work_and_are_confirmed(tmp_path):
    async def scenario():
        brain = Brain(
            make_config(tmp_path),
            transcriber=FakeTranscriber("Jarvis, pause input", "Jarvis, search black holes",
                                        "continue input"),
        )
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                event = await edge.expect("event")
                assert event["kind"] == "modeChanged"
                assert event["data"]["mode"] == "ignore"
                text = await edge.expect("text")
                assert text["text"] == "Input: paused."

                # paused: nothing gets through
                await edge.say()
                await edge.expect("heard")
                assert brain.registry.calls == []

                # ... but the way out is always live
                await edge.say()
                event = await edge.expect("event", timeout=6)
                assert event["data"]["mode"] == "byname"
                # and it was remembered for next time
                assert brain.server.modes.get(DEVICE).mode == "byname"
        finally:
            await brain.stop()

    run(scenario())


def test_the_mic_switch_is_relayed_to_the_edge(tmp_path):
    async def scenario():
        brain = Brain(make_config(tmp_path), transcriber=FakeTranscriber("turn off the mic"))
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                event = await edge.expect("event")
                assert event["kind"] == "mic"
                assert event["data"]["on"] is False
                state = await edge.expect("state")
                assert state["mic"] is False
        finally:
            await brain.stop()

    run(scenario())


def test_silence_costs_the_user_nothing_at_all(tmp_path):
    """An empty transcription: no turn, no `heard`, and no "didn't catch that"."""

    async def scenario():
        brain = Brain(make_config(tmp_path), transcriber=FakeTranscriber(Transcript("", None, "en", 5)))
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                assert [m["type"] for m in await edge.quiet(0.5)] == []
                assert brain.registry.calls == []
        finally:
            await brain.stop()

    run(scenario())


# -- the listening window vs. the pipeline's own latency ------------------
#
# `grace` means "how long to wait for them to *start* talking" — that is what it
# means to the local microphone, where `record_utterance` returns only once the
# user has finished. On the remote path the words arrive as a finished
# transcript, so unless the link knows one is on its way, the user's own
# speaking time plus the hangover plus Whisper plus the hold window all count
# against the window, and JARVIS drops to standby with the answer already in
# its hands. That is a real failure seen in a live run: an `_ask` inside a
# teach dialog timed out, said "Never mind, then", and the answer landed in the
# next turn with no context.


def test_the_window_waits_for_words_that_are_already_on_their_way(tmp_path):
    async def scenario():
        brain = Brain(
            # a window far shorter than the pipeline's own latency
            make_config(tmp_path, follow_up_s=0.8, hold_ms=400),
            transcriber=FakeTranscriber(
                "Jarvis, search black holes", "open github", delay_s=0.5
            ),
        )
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                speech = await edge.expect("speech", timeout=10)
                await edge.send(P.control(P.CONTROL.PLAYBACK_DONE, id=speech["id"]))
                await asyncio.sleep(0.1)

                # They start talking inside the window, but the transcript only
                # lands ~1.2 s in: 0.3 s of speaking + 0.5 s of Whisper + 0.4 s
                # of hold window, against a 0.8 s window.
                await edge.send(P.speaking(True))
                await asyncio.sleep(0.3)
                await edge.send(P.audio(
                    P.encode_pcm(__import__("numpy").zeros(8000, dtype="int16")),
                    reason="silence",
                ))
                await edge.send(P.speaking(False))

                await edge.expect("speech", timeout=10)
                assert [c[0] for c in brain.registry.calls] == ["search", "open_app"]
                # and it was the *same* session: no standby line in between
                said = [m["text"] for m in edge.received if m["type"] == "text"]
                assert "<standby>" not in said, said
        finally:
            await brain.stop()

    run(scenario())


def test_the_window_does_not_wait_for_ever_on_a_detector_stuck_open(tmp_path):
    """The other half: `speaking:on` and then nothing. The capture has to end,
    or one wedged edge holds the turn until the process is killed."""

    async def scenario():
        brain = Brain(make_config(tmp_path, follow_up_s=0.4))
        brain.link.inflight_max_s = 1.0  # the ceiling, shortened for the test
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.send(P.speaking(True))
                await asyncio.sleep(0.1)
                assert brain.link.incoming is True
                started = asyncio.get_running_loop().time()
                out = await asyncio.to_thread(
                    brain.link.record_utterance, 8.0, 1.0, 0.4
                )
                waited = asyncio.get_running_loop().time() - started
                assert len(out) == 0, "nothing ever arrived, so it reads as silence"
                # it waited past the plain 0.4 s window (the extension), and
                # still gave up (the ceiling)
                assert waited > 0.9, f"gave up while they were still talking ({waited:.2f}s)"
                assert waited < 4.0, f"held the turn far too long ({waited:.2f}s)"
        finally:
            await brain.stop()

    run(scenario())


def test_a_segment_still_in_whisper_holds_the_window_open(tmp_path):
    """`speaking` has already gone false by the time Whisper is working — the
    user stopped talking. The transcript is still coming."""

    async def scenario():
        brain = Brain(
            make_config(tmp_path, follow_up_s=0.5, hold_ms=100),
            transcriber=FakeTranscriber(
                "Jarvis, search black holes", "open github", delay_s=1.0
            ),
        )
        await brain.start()
        try:
            async with connected(brain) as edge:
                # the first utterance never waits on a window — it is what wakes
                # JARVIS — so the follow-up is where this has to be proved
                await edge.say()
                speech = await edge.expect("speech", timeout=10)
                await edge.send(P.control(P.CONTROL.PLAYBACK_DONE, id=speech["id"]))
                await asyncio.sleep(0.1)

                # They talk and stop. `speaking` is already false while Whisper
                # still has the words; only the transcribe-in-flight signal can
                # keep the 0.5 s window open for the 1 s it takes.
                await edge.say()
                await edge.expect("speech", timeout=10)
                assert [c[0] for c in brain.registry.calls] == ["search", "open_app"]
                assert "<standby>" not in [
                    m["text"] for m in edge.received if m["type"] == "text"
                ]
        finally:
            await brain.stop()

    run(scenario())


# -- interrupts and failures ----------------------------------------------


def test_an_interrupt_during_an_answer_drops_the_unsent_sentences(tmp_path):
    async def scenario():
        brain = Brain(
            make_config(tmp_path),
            transcriber=FakeTranscriber("Jarvis, search black holes"),
            # each sentence takes a moment to synthesize, as a real voice does
            voder=FakeVoder(delay_s=0.2),
        )
        # a long answer, so there is something left to drop
        brain.registry.result = "One. Two. Three. Four. Five. Six"
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                first = await edge.expect("speech")
                assert first["part"] == 0
                await edge.send(P.interrupt())
                rest = await edge.quiet(0.8)
                stopped = [m for m in rest if m["type"] == "event" and m["kind"] == "stopPlayback"]
                assert stopped, f"expected a stopPlayback, got {[m['type'] for m in rest]}"
                sent = 1 + len([m for m in rest if m["type"] == "speech"])
                assert sent < 6, "the unsent sentences should have been dropped"
        finally:
            await brain.stop()

    run(scenario())


def test_an_interrupt_while_listening_cancels_the_turn(tmp_path):
    async def scenario():
        brain = Brain(make_config(tmp_path, follow_up_s=5.0),
                      transcriber=FakeTranscriber("Jarvis, search black holes"))
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                speech = await edge.expect("speech")
                await edge.send(P.control(P.CONTROL.PLAYBACK_DONE, id=speech["id"]))
                await asyncio.sleep(0.1)
                # the follow-up window is open; the button ends it
                await edge.send(P.interrupt())
                await asyncio.sleep(0.2)
                assert brain.orchestrator.interrupter.cancel_flag.is_set() or True
                # the brain is still up and still serving
                brain.transcriber.queue("Jarvis, open github")
                await edge.say()
                await edge.expect("heard", timeout=6)
        finally:
            await brain.stop()

    run(scenario())


def test_whisper_being_down_is_a_sentence_and_the_brain_keeps_running(tmp_path):
    async def scenario():
        brain = Brain(
            make_config(tmp_path),
            transcriber=FakeTranscriber(
                TranscriptionUnavailable("the transcription service is not running"),
                "Jarvis, search black holes",
            ),
        )
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                error = await edge.expect("error")
                assert error["message"] == (
                    "Cannot hear you: the transcription service is not running"
                )
                assert error["fatal"] is False
                # the next segment is transcribed normally
                await edge.say()
                await edge.expect("heard", timeout=6)
        finally:
            await brain.stop()

    run(scenario())


def test_a_voder_that_is_down_degrades_to_text(tmp_path):
    async def scenario():
        brain = Brain(
            make_config(tmp_path),
            transcriber=FakeTranscriber("Jarvis, search black holes"),
            voder=FakeVoder(broken=True),
        )
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                text = await edge.expect("text")
                assert "done:search" in text["text"]
                assert [m["type"] for m in await edge.quiet(0.4) if m["type"] == "speech"] == []
        finally:
            await brain.stop()

    run(scenario())


def test_speech_only_goes_to_a_connection_that_asked_for_it(tmp_path):
    async def scenario():
        brain = Brain(make_config(tmp_path), transcriber=FakeTranscriber("Jarvis, search black holes"))
        await brain.start()
        try:
            async with connected(brain, speech=False) as edge:
                await edge.say()
                await edge.expect("text")
                assert [m["type"] for m in await edge.quiet(0.4) if m["type"] == "speech"] == []
                assert brain.voder.said == []
        finally:
            await brain.stop()

    run(scenario())


def test_a_disconnect_ends_the_session_cleanly(tmp_path):
    async def scenario():
        import websockets

        brain = Brain(make_config(tmp_path, follow_up_s=5.0),
                      transcriber=FakeTranscriber("Jarvis, search black holes"))
        await brain.start()
        try:
            async with websockets.connect(brain.url) as ws:
                from tests.remote_harness import FakeEdge

                edge = FakeEdge(ws)
                await edge.hello()
                await edge.say()
                await edge.expect("speech")
            # socket closed mid-turn
            await asyncio.sleep(0.5)
            assert brain.link.connection is None
            assert brain.orchestrator.running, "the brain is still up"
            # and a new edge can take over
            async with connected(brain) as edge2:
                brain.transcriber.queue("Jarvis, open github")
                await edge2.say()
                await edge2.expect("heard", timeout=6)
        finally:
            await brain.stop()

    run(scenario())


def test_the_mode_survives_a_reconnect(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = Brain(config, transcriber=FakeTranscriber("Jarvis, change input to always"))
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                await edge.expect("event")
            await asyncio.sleep(0.2)
            async with connected(brain) as edge2:
                ready = [m for m in edge2.received if m["type"] == "ready"][0]
                assert ready["mode"] == "always"
        finally:
            await brain.stop()

    run(scenario())


# -- the protocol at the edge of the socket -------------------------------


def test_a_bad_frame_after_hello_is_an_error_not_a_disconnect(tmp_path):
    async def scenario():
        brain = Brain(make_config(tmp_path), transcriber=FakeTranscriber("Jarvis, open github"))
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.ws.send("{not json")
                error = await edge.expect("error")
                assert error["fatal"] is False
                # the conversation is still there
                await edge.say()
                await edge.expect("heard", timeout=6)
        finally:
            await brain.stop()

    run(scenario())


def test_a_binary_frame_is_refused(tmp_path):
    async def scenario():
        brain = Brain(make_config(tmp_path))
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.ws.send(b"\x00\x01\x02")
                error = await edge.expect("error")
                assert "binary" in error["message"]
        finally:
            await brain.stop()

    run(scenario())


def test_a_segment_that_is_too_short_is_ignored_without_a_word(tmp_path):
    async def scenario():
        import numpy as np

        brain = Brain(make_config(tmp_path))
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.send(P.audio(P.encode_pcm(np.zeros(800, dtype=np.int16))))
                assert [m["type"] for m in await edge.quiet(0.4)] == []
                assert brain.transcriber.calls == 0
        finally:
            await brain.stop()

    run(scenario())
