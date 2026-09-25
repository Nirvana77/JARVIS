"""A brain and an edge on a loopback socket, with fakes where the models are.

Shared by ``test_remote_link.py`` and ``test_remote_auth.py``. Nothing here
touches a real network interface, microphone, GPU or speaker: the server binds
127.0.0.1:0, the transcriber hands back canned text per segment, and the voder
hands back a short buffer of silence.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import replace

import numpy as np

from jarvis.audio.whisper_client import Transcript, TranscriptionUnavailable
from jarvis.config import load_config
from jarvis.core.orchestrator import Orchestrator
from jarvis.nlu.corpus import intent_meta
from jarvis.remote import protocol as P
from jarvis.remote.addressing import DeviceModes
from jarvis.remote.server import RemoteLink, RemoteServer

TOKEN = "s3cret"
DEVICE = "livingroom"


def make_config(tmp_path, **over):
    """A config with everything sized for a test: loopback, short windows."""
    config = load_config()
    config = replace(
        config,
        data_dir=tmp_path,
        server=replace(config.server, host="127.0.0.1", port=0, hello_timeout_s=1.0),
        addressing=replace(config.addressing, hold_ms=over.pop("hold_ms", 80)),
        capture=replace(
            config.capture,
            window_timeout_s=over.pop("window_timeout_s", 0.6),
            follow_up_s=over.pop("follow_up_s", 0.6),
            barge_in=False,
        ),
        edge_tokens={DEVICE: TOKEN},
        **over,
    )
    return config


class FakeTranscriber:
    """Canned text per segment, in the order the segments arrive. An entry may
    be an exception (the service being down)."""

    def __init__(self, *results, delay_s=0.0):
        self.results = list(results)
        self.calls = 0
        #: real transcription costs about a second, and that second is the
        #: whole reason the listening window has to know a segment is coming
        self.delay_s = delay_s

    def queue(self, *results):
        self.results.extend(results)

    def transcribe_full(self, pcm):
        if self.delay_s:
            import time

            time.sleep(self.delay_s)
        self.calls += 1
        out = self.results.pop(0) if self.results else Transcript("", None, "en", 1)
        if isinstance(out, Exception):
            raise out
        if isinstance(out, str):
            out = Transcript(out, 0.9, "en", 10)
        return out

    def transcribe(self, pcm, cancel=None):
        return self.transcribe_full(pcm).text

    def health(self):
        return {"ok": True, "model": "fake", "device": "cpu", "warm": True}


class FakeVoder:
    """A tenth of a second of silence per sentence, and a record of what it was
    asked to say."""

    available = True
    voice = None

    def __init__(self, *, broken=False, delay_s=0.0):
        self.said: list[str] = []
        self.broken = broken
        #: real synthesis costs something per sentence, which is the whole
        #: reason an answer can be interrupted part-way through it
        self.delay_s = delay_s

    def speak(self, text, voice=None):
        if self.delay_s:
            import time

            time.sleep(self.delay_s)
        self.said.append(text)
        if self.broken:
            from jarvis.audio.voder_client import SynthesisUnavailable

            raise SynthesisUnavailable("the speech service is not running")
        return np.zeros(1600, dtype=np.int16), 16000

    def speak_or_none(self, text, voice=None):
        try:
            return self.speak(text, voice)
        except Exception:  # noqa: BLE001
            return None

    def health(self):
        return {"ok": True, "voice": "fake", "sample_rate": 16000}


class FakePersona:
    def __init__(self):
        self.spoken: list[str] = []

    def line(self, event, default=""):
        return f"<{event}>"

    def phrase(self, text):
        return text


class FakeNLU:
    def __init__(self, mapping):
        self.mapping = mapping

    def predict(self, text):
        return self.mapping.get(text, ("unknown", 0.1))

    def explain(self, text):
        from jarvis.nlu.classifier import Prediction

        label, conf = self.mapping.get(text, ("unknown", 0.1))
        return Prediction(label, conf, [(label, conf)], 1.0)


class FakeRegistry:
    def __init__(self, result="done"):
        self.calls: list[tuple] = []
        self.result = result

    def dispatch(self, label, params):
        self.calls.append((label, params))
        return f"{self.result}:{label}"

    def names(self):
        return []

    def manifests(self):
        return []


class Brain:
    """The server, the link and a real orchestrator with fake NLU/skills."""

    def __init__(self, config, *, transcriber=None, voder=None, nlu=None):
        self.config = config
        self.voder = voder or FakeVoder()
        self.transcriber = transcriber or FakeTranscriber()
        self.link = RemoteLink(config, voder=self.voder)
        self.server = RemoteServer(
            config,
            self.link,
            transcriber=self.transcriber,
            modes=DeviceModes(config.remote_dir, config.addressing.default_mode),
        )
        self.persona = FakePersona()
        self.registry = FakeRegistry()
        self.orchestrator = Orchestrator(
            config=config,
            wake=self.link,
            mic=self.link,
            stt=self.link,
            tts=self.link,
            nlu=nlu or FakeNLU({
                "search black holes": ("search", 0.9),
                "search black holes and neutron stars": ("search", 0.9),
                "open github": ("open_app", 0.9),
                "yes": ("yes", 0.9),
                "go to sleep": ("goodbye", 0.9),
            }),
            persona=self.persona,
            registry=self.registry,
            intent_meta=intent_meta(),
        )
        self.link.attach(self.orchestrator)
        self.ws_server = None
        self.task: asyncio.Task | None = None

    @property
    def port(self) -> int:
        return self.ws_server.sockets[0].getsockname()[1]

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    async def start(self, *, run_orchestrator=True):
        self.link.bind_loop()
        self.ws_server = await self.server.serve()
        if run_orchestrator:
            self.task = asyncio.ensure_future(self.orchestrator.run())
            await asyncio.sleep(0)  # let it reach its first await
        return self

    async def stop(self):
        self.orchestrator.stop()
        self.link.submit("")  # nudge a blocked read
        self.link.disconnect(self.link.connection) if self.link.connection else None
        if self.task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self.task, timeout=5)
        self.ws_server.close()
        await self.ws_server.wait_closed()


class FakeEdge:
    """The edge side of the socket, driven a message at a time."""

    def __init__(self, ws):
        self.ws = ws
        self.received: list[dict] = []

    async def send(self, msg):
        await self.ws.send(json.dumps(msg))

    async def hello(self, token=TOKEN, device_id=DEVICE, speech=True):
        await self.send(P.hello(token, device_id))
        ready = await self.expect("ready")
        if speech:
            await self.send(P.control(P.CONTROL.SET_SPEECH, on=True, sample_rate=16000))
            await self.expect("ready")
        return ready

    async def say(self, ms=800):
        """One segment of (fake) speech; what it 'says' is the transcriber's
        next canned line."""
        pcm = P.encode_pcm(np.zeros(int(16000 * ms / 1000), dtype=np.int16))
        await self.send(P.speaking(True))
        await self.send(P.audio(pcm, reason="silence", floor_db=-60, peak_db=-20))
        await self.send(P.speaking(False))

    async def recv(self, timeout=5.0) -> dict:
        raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        msg = json.loads(raw)
        self.received.append(msg)
        return msg

    async def expect(self, kind, timeout=5.0) -> dict:
        """The next message of this type, skipping the ones in between."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            left = deadline - asyncio.get_running_loop().time()
            if left <= 0:
                raise AssertionError(
                    f"no {kind!r} within {timeout}s; got "
                    f"{[m['type'] for m in self.received]}"
                )
            msg = await self.recv(timeout=left)
            if msg["type"] == kind:
                return msg

    async def quiet(self, seconds=0.4) -> list[dict]:
        """Everything that arrives in the next `seconds`, which is usually the
        point: nothing should."""
        out: list[dict] = []
        try:
            while True:
                out.append(await self.recv(timeout=seconds))
        except (asyncio.TimeoutError, TimeoutError):
            pass
        return out

    def types(self) -> list[str]:
        return [m["type"] for m in self.received]


@contextlib.asynccontextmanager
async def connected(brain, **kw):
    import websockets

    async with websockets.connect(brain.url) as ws:
        edge = FakeEdge(ws)
        await edge.hello(**kw)
        yield edge
