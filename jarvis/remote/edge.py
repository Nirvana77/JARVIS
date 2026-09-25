"""The audio satellite — ``python -m jarvis edge`` (M3 decisions 1, 2, 10).

Runs on a Raspberry Pi-class box in the room. It owns the microphone, the
speaker and (optionally) a push-to-talk button, and **nothing else**: no wake
word, no models, no ONNX. Its whole install is::

    pip install numpy sounddevice websockets       # + gpiozero for a button
    sudo apt install libportaudio2

What it does, in one loop: cut what it hears into speech segments
(``jarvis/audio/segment.py``), tell the brain the moment its detector opens or
closes one, send the segment, and play back the sentences that come the other
way. Everything else — transcription, addressing, skills, persona, speech
synthesis — happens on the brain.

Echo, stated plainly: while it is playing an answer the edge does not feed its
segmenter at all, and it keeps ignoring input for a short tail afterwards. So
the Pi cannot hear itself — and it cannot hear you over itself either. Voice
barge-in *over* playback needs echo cancellation; the protocol already carries
everything that would need.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import replace

import numpy as np

from jarvis.audio.player import Player
from jarvis.audio.segment import Segmenter, float_to_pcm16, resample_to
from jarvis.remote import protocol as P

log = logging.getLogger(__name__)

#: Played when something was said (or the button pressed) with no brain to send
#: it to. Two short notes, so "it isn't listening" is never silent.
_EARCON_HZ = (660, 440)
_EARCON_MS = 90

#: After playback, keep ignoring the microphone this long: the tail of the
#: speaker, and the room's own reverberation of it.
_ECHO_TAIL_S = 0.2

#: Between sentences of one answer, so it doesn't run together.
_PART_GAP_S = 0.15


def _earcon(sample_rate: int = 16_000) -> np.ndarray:
    parts = []
    for hz in _EARCON_HZ:
        n = int(sample_rate * _EARCON_MS / 1000)
        t = np.arange(n, dtype=np.float32) / sample_rate
        envelope = np.minimum(1.0, np.minimum(t * 40, (n / sample_rate - t) * 40))
        parts.append(0.2 * envelope * np.sin(2 * math.pi * hz * t, dtype=np.float32))
    return float_to_pcm16(np.concatenate(parts))


class Microphone:
    """Raw frames off the sound card, resampled to the one rate on the wire.

    Deliberately not ``jarvis.audio.capture.Microphone``: that one calibrates a
    noise floor and decides where an utterance ends, and on this path the
    segmenter does both — better, and identically on every machine.
    """

    def __init__(self, on_frames, *, device=None, pipewire_node: str = "", sample_rate: int = 16_000):
        self.on_frames = on_frames
        self.device = device
        self.pipewire_node = pipewire_node
        self.sample_rate = sample_rate
        self._stream = None
        self.native_rate = sample_rate

    @property
    def open(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        from jarvis.audio.pipewire import pipewire_target

        def callback(indata, frames, time_info, status):  # sound-card thread
            if status:
                log.debug("input status: %s", status)
            self.on_frames(np.frombuffer(bytes(indata), dtype=np.int16).copy())

        with pipewire_target(self.pipewire_node) as target:
            device = self.device if target is None else target
            try:
                self._stream = sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=int(self.sample_rate * 0.02),
                    dtype="int16",
                    channels=1,
                    device=device,
                    callback=callback,
                )
                self.native_rate = self.sample_rate
            except Exception:  # noqa: BLE001 — a card that will not open at 16 kHz
                info = sd.query_devices(device, "input")
                self.native_rate = int(info["default_samplerate"])
                log.info("sound card wants %d Hz — resampling on the edge", self.native_rate)
                self._stream = sd.RawInputStream(
                    samplerate=self.native_rate,
                    blocksize=int(self.native_rate * 0.02),
                    dtype="int16",
                    channels=1,
                    device=device,
                    callback=callback,
                )
        self._stream.start()

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("closing the mic: %s", exc)

    def to_wire(self, frames: np.ndarray) -> np.ndarray:
        if self.native_rate == self.sample_rate:
            return frames
        return float_to_pcm16(
            resample_to(frames.astype(np.float32) / 32768.0, self.native_rate, self.sample_rate)
        )


class Edge:
    """One edge device: connect, listen, speak, reconnect."""

    def __init__(self, config, *, connect=None) -> None:
        self.config = config
        self.edge = config.edge
        self.options = self.edge.segmenter_options()
        self.segmenter = Segmenter(self.options)
        self.player = Player(
            device=self.edge.output_device, pipewire_node=self.edge.output_pipewire_node
        )
        self.mic = Microphone(
            self._on_frames,
            device=self.edge.input_device,
            pipewire_node=self.edge.input_pipewire_node,
            sample_rate=self.options.sample_rate,
        )
        self._connect = connect  # injected in tests; defaults to websockets.connect

        self.mode = config.addressing.default_mode
        self.mic_wanted = True          # the brain's `mic{on}` switch
        self.holding = False            # the push-to-talk button is down
        self.running = False
        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frames: asyncio.Queue | None = None
        self._speech: asyncio.Queue | None = None
        self._speaking_sent = False
        self._muted_until = 0.0         # echo: ignore the mic until this time
        self._playing_id: str | None = None
        self._button = None
        self._failures = 0
        self._reported_offline_at = 0.0

    # -- the sound card ------------------------------------------------

    def _on_frames(self, frames: np.ndarray) -> None:
        """Sound-card thread. Hand the frames to the loop and return: anything
        slower than that here is a dropout."""
        loop, q = self._loop, self._frames
        if loop is None or q is None:
            return
        try:
            loop.call_soon_threadsafe(q.put_nowait, frames)
        except RuntimeError:
            pass

    @property
    def ptt(self) -> bool:
        return self.mode == "pushtotalk"

    def mic_should_be_open(self) -> bool:
        if not self.mic_wanted:
            return False
        if self.ptt:
            # In push-to-talk the microphone is *actually* off until the button
            # is held. That is the whole reason the mode exists.
            return self.holding
        return True

    def apply_mic_state(self) -> None:
        if self.mic_should_be_open():
            self.mic.start()
        elif self.mic.open:
            self.mic.stop()
            self.segmenter.reset()

    # -- segmentation --------------------------------------------------

    async def _pump(self) -> None:
        """Frames in, segments out. One task, one segmenter, no locks."""
        assert self._frames is not None
        while self.running:
            frames = await self._frames.get()
            if time.monotonic() < self._muted_until:
                continue  # our own voice, or its tail
            segments = self.segmenter.push(self.mic.to_wire(frames))
            await self._report_speaking()
            for segment in segments:
                await self._send_segment(segment)

    async def _report_speaking(self) -> None:
        """Sent the moment the detector flips, and *before* the audio: the
        brain's hold window needs to know somebody is still talking while the
        first half of their sentence is still in Whisper."""
        if self.segmenter.speaking == self._speaking_sent:
            return
        self._speaking_sent = self.segmenter.speaking
        await self._send(P.speaking(self._speaking_sent))

    async def _send_segment(self, segment) -> None:
        if self._ws is None:
            # Recorded with nowhere to send it. Dropped, not queued: a command
            # replayed minutes late is worse than a lost one.
            await self._offline_earcon()
            return
        await self._send(
            P.audio(
                P.encode_pcm(segment.pcm),
                final=True,
                reason=segment.reason,
                floor_db=segment.floor_db,
                peak_db=segment.peak_db,
            )
        )
        log.info(
            "segment %dms %s floor %d peak %d",
            segment.duration_ms, segment.reason, segment.floor_db, segment.peak_db,
        )

    # -- the button ----------------------------------------------------

    def _setup_button(self) -> None:
        if not self.edge.ptt_gpio:
            return
        try:
            from gpiozero import Button
        except Exception as exc:  # noqa: BLE001
            log.warning("no push-to-talk button (%s) — gpiozero not available", exc)
            return
        self._button = Button(self.edge.ptt_gpio, hold_time=0.05)
        # The button must never be conditional on the connection or anything
        # else: in push-to-talk it is the only way in.
        self._button.when_pressed = lambda: self._from_button(self.on_press)
        self._button.when_released = lambda: self._from_button(self.on_release)
        log.info("push-to-talk button on GPIO %d", self.edge.ptt_gpio)

    def _from_button(self, coro_fn) -> None:
        loop = self._loop
        if loop is not None:
            asyncio.run_coroutine_threadsafe(coro_fn(), loop)

    async def on_press(self) -> None:
        """A press stops a spoken answer at once — without waiting for the
        brain — and then opens the microphone."""
        if self._playing_id is not None:
            self.player.stop()
            self._drain_speech()
            self._playing_id = None
            await self._send(P.interrupt())
        self.holding = True
        # In hold mode silence no longer closes a segment — the release does —
        # but the detector still runs, so the silence at both ends is trimmed
        # before Whisper ever sees it.
        self.segmenter = Segmenter(replace(self.options, hold=True))
        self._speaking_sent = False
        self.apply_mic_state()
        if self._ws is None:
            await self._offline_earcon()

    async def on_release(self) -> None:
        self.holding = False
        segment = self.segmenter.flush("release")
        if segment is not None:
            await self._send_segment(segment)
        self.segmenter = Segmenter(self.options)
        self._speaking_sent = False
        await self._report_speaking()
        self.apply_mic_state()

    # -- playback ------------------------------------------------------

    def _drain_speech(self) -> None:
        if self._speech is None:
            return
        while not self._speech.empty():
            self._speech.get_nowait()

    async def _play_loop(self) -> None:
        """Play the sentences of one answer in order, then say so."""
        assert self._speech is not None
        while self.running:
            msg = await self._speech.get()
            pcm = P.decode_pcm(msg["pcm"])
            rate = int(msg["sample_rate"])
            self._playing_id = msg["id"]
            # Half-duplex: the mic is ignored while this plays, and for a tail
            # afterwards, so the Pi cannot hear itself.
            self._muted_until = time.monotonic() + len(pcm) / max(1, rate) + _ECHO_TAIL_S
            finished = await asyncio.to_thread(self.player.play, pcm, rate)
            self._muted_until = time.monotonic() + _ECHO_TAIL_S
            self.segmenter.reset()
            self._speaking_sent = False
            if not finished:
                self._playing_id = None
                continue
            if msg.get("final"):
                await self._send(P.control(P.CONTROL.PLAYBACK_DONE, id=msg["id"]))
                self._playing_id = None
            else:
                await asyncio.sleep(_PART_GAP_S)

    def _report_offline(self, exc: Exception) -> None:
        """Say why we are not connected — once when it starts, then rarely.

        The edge retries for ever on purpose (the brain may simply be off), but
        a silent retry loop is indistinguishable from a hang, and the reason is
        the whole diagnosis: a refused connection means nothing is listening, a
        401 means Cloudflare Access is in front, a name error means DNS.
        """
        reason = f"{type(exc).__name__}: {exc}".rstrip(": ")
        log.info("not connected (%s)", reason)
        now = time.monotonic()
        first = self._failures == 0
        self._failures += 1
        if first or now - self._reported_offline_at > 30:
            self._reported_offline_at = now
            print(f"! cannot reach {self.edge.server_url} — {reason}", flush=True)
            if first:
                print("  retrying… (-v for every attempt, Ctrl-C to stop)", flush=True)

    def _report_online(self) -> None:
        if self._failures:
            print(f"  …connected after {self._failures} attempt(s).", flush=True)
        self._failures = 0

    async def _offline_earcon(self) -> None:
        await asyncio.to_thread(self.player.play, _earcon(), self.options.sample_rate)

    # -- the socket ----------------------------------------------------

    async def _send(self, msg: dict) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps(msg))
        except Exception as exc:  # noqa: BLE001 — the reconnect loop handles it
            log.debug("send failed: %s", exc)

    async def _receive(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if not isinstance(msg, dict):
                continue
            await self._handle(msg)

    async def _handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == P.S2C.SPEECH:
            if self._speech is not None:
                self._speech.put_nowait(msg)
            return
        if kind == P.S2C.READY:
            self.mode = msg.get("mode", self.mode)
            self.apply_mic_state()
            return
        if kind == P.S2C.STATE:
            mode, mic = msg.get("mode"), msg.get("mic")
            if mode:
                self.mode = mode
            if isinstance(mic, bool):
                self.mic_wanted = mic
            self.apply_mic_state()
            return
        if kind == P.S2C.EVENT:
            kinds = msg.get("kind")
            data = msg.get("data") or {}
            if kinds == "stopPlayback":
                self.player.stop()
                self._drain_speech()
                self._playing_id = None
            elif kinds == "mic" and isinstance(data.get("on"), bool):
                self.mic_wanted = data["on"]
                self.apply_mic_state()
            elif kinds == "modeChanged" and data.get("mode"):
                self.mode = data["mode"]
                self.apply_mic_state()
            return
        if kind == P.S2C.HEARD:
            log.info("heard: %r", msg.get("text"))
            return
        if kind == P.S2C.TEXT:
            print(f"Jarvis: {msg.get('text')}", flush=True)
            return
        if kind == P.S2C.ERROR:
            log.warning("brain: %s", msg.get("message"))
            print(f"! {msg.get('message')}", flush=True)
            return

    async def _session(self, ws) -> None:
        self._ws = ws
        await self._send(P.hello(self.config.edge_token or "", self.edge.device_id))
        await self._send(
            P.control(P.CONTROL.SET_SPEECH, on=True, sample_rate=self.config.voder.sample_rate)
        )
        self.segmenter.reset()
        self._speaking_sent = False
        self.apply_mic_state()
        try:
            await self._receive(ws)
        finally:
            self._ws = None
            self.mic.stop()
            self.segmenter.reset()

    async def run(self) -> None:
        """Connect, and keep connecting. Backs off up to ``reconnect_max_s``."""
        self._loop = asyncio.get_running_loop()
        self._frames = asyncio.Queue(maxsize=256)
        self._speech = asyncio.Queue()
        self.running = True
        self._setup_button()
        pump = asyncio.ensure_future(self._pump())
        play = asyncio.ensure_future(self._play_loop())
        backoff = 1.0
        try:
            while self.running:
                try:
                    async with self._open() as ws:
                        log.info("connected to %s", self.edge.server_url)
                        print(f"edge '{self.edge.device_id}' connected.", flush=True)
                        self._report_online()
                        backoff = 1.0
                        await self._session(ws)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — every failure reconnects
                    self._report_offline(exc)
                if not self.running:
                    break
                await asyncio.sleep(backoff)
                backoff = min(self.edge.reconnect_max_s, backoff * 2)
        finally:
            self.running = False
            for task in (pump, play):
                task.cancel()
            await asyncio.gather(pump, play, return_exceptions=True)
            self.mic.stop()
            self.player.close()
            if self._button is not None:
                self._button.close()

    def _open(self):
        if self._connect is not None:
            return self._connect()
        import ssl

        import websockets

        kwargs = {"max_size": self.config.server.max_size, "ping_interval": 20}
        if self.edge.server_url.startswith("wss://"):
            context = ssl.create_default_context()
            if self.edge.tls_ca:
                context.load_verify_locations(self.edge.tls_ca)
            kwargs["ssl"] = context
        return websockets.connect(self.edge.server_url, **kwargs)

    def stop(self) -> None:
        self.running = False


def run_edge(config) -> int:
    edge = Edge(config)
    if not config.edge_token:
        print(
            "error: no JARVIS_EDGE_TOKEN in .env — the brain will refuse this device.",
            flush=True,
        )
        return 2
    print(
        f"JARVIS edge '{config.edge.device_id}' -> {config.edge.server_url}. Ctrl-C to quit.",
        flush=True,
    )
    try:
        asyncio.run(edge.run())
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    return 0
