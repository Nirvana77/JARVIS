"""The shelved-extra command-interrupt utility (jarvis/core/interrupt.py)."""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from jarvis.config import load_config
from jarvis.core.interrupt import Cancelled, Interrupter


class FakeMic:
    """`.read()` returns loud frames for the first `talking_reads` calls, then
    silence. `record_utterance` returns a fixed 'rest' clip."""

    def __init__(self, talking_reads=0, rest=b""):
        self._talking_reads = talking_reads
        self._reads = 0
        self._rest = rest

    def read(self, timeout=None):
        self._reads += 1
        if self._reads <= self._talking_reads:
            return np.full(1280, 6000, dtype=np.int16)
        return np.zeros(1280, dtype=np.int16)

    def record_utterance(self, *a, **k):
        return (
            np.ones(8000, dtype=np.float32) * 0.1
            if self._rest == "speech"
            else np.zeros(0, dtype=np.float32)
        )


@pytest.fixture
def interrupter():
    it = Interrupter(load_config())
    it._interrupt = asyncio.Event()  # normally set by install()
    return it


def test_passthrough_when_nothing_is_enabled(interrupter):
    interrupter.enabled_enter = False
    interrupter.enabled_barge = False
    interrupter._interrupt = None

    async def scenario():
        work = asyncio.ensure_future(_immediately("hello world"))
        kind, value = await interrupter.guard_transcription(work, threading.Event(), None)
        assert (kind, value) == ("done", "hello world")

    asyncio.run(scenario())


def test_enter_raises_cancelled_and_stashes_the_decode(interrupter):
    interrupter.enabled_barge = False

    async def scenario():
        cancel = threading.Event()
        work = asyncio.ensure_future(_blocked(cancel))
        task = asyncio.ensure_future(
            interrupter.guard_transcription(work, cancel, FakeMic())
        )
        await asyncio.sleep(0.02)
        interrupter._interrupt.set()  # "Enter"
        with pytest.raises(Cancelled):
            await task
        assert cancel.is_set()                    # the decode was told to stop
        assert len(interrupter._draining) == 1
        await interrupter.shutdown()

    asyncio.run(scenario())


def test_barge_in_returns_new_audio_and_abandons_the_decode(interrupter):
    interrupter.enabled_enter = False
    interrupter._interrupt = None
    interrupter.enabled_barge = True
    mic = FakeMic(talking_reads=40, rest="")  # sustained speech -> onset fires

    async def scenario():
        cancel = threading.Event()
        work = asyncio.ensure_future(_blocked(cancel))
        kind, value = await interrupter.guard_transcription(work, cancel, mic)
        assert kind == "barge"
        assert isinstance(value, np.ndarray) and len(value) > 0
        assert cancel.is_set()
        assert len(interrupter._draining) == 1
        await interrupter.shutdown()

    asyncio.run(scenario())


def test_onset_worker_ignores_silence(interrupter):
    interrupter._barge_stop.clear()
    mic = FakeMic(talking_reads=0)

    def stop_soon():
        import time

        time.sleep(0.1)
        interrupter._barge_stop.set()

    threading.Thread(target=stop_soon).start()
    assert interrupter._onset_worker(mic) is None


# -- helpers ---------------------------------------------------------------

async def _immediately(text):
    return text


async def _blocked(cancel: threading.Event):
    for _ in range(50):
        if cancel.is_set():
            return "partial"
        await asyncio.sleep(0.02)
    return "done"
