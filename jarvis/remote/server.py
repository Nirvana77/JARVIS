"""The brain's WebSocket server, and ``RemoteLink`` (M3 decisions 7, 9, 10).

``RemoteLink`` plays the orchestrator's ``wake``, ``mic``, ``stt`` and ``tts``
roles, exactly as ``TextIO`` plays ``mic`` and ``stt`` from queued text — **the
orchestrator is not rewritten**. By the time it sees anything, the segments have
been transcribed by ``jarvis-whisper``, gated by the addressing mode and merged
by the hold window; what reaches ``record_utterance`` is one finished utterance
of text.

The threading, stated once because everything below depends on it: the
WebSocket handler runs on the event loop, and the orchestrator calls the four
roles from worker threads (``asyncio.to_thread``). So the queue between them is
a plain ``queue.Queue``, states and messages go back through
``run_coroutine_threadsafe``, and nothing on the loop side ever blocks.

Security (decision 9): the link is assumed to be on the public internet. TLS is
required unless the brain is on loopback behind a proxy, every device has its
own pre-shared token compared with ``hmac.compare_digest``, and a missing or
late ``hello`` closes the socket. Tokens and audio are never logged.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import queue
import ssl
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from jarvis.audio.voder_client import build_voder
from jarvis.audio.whisper_client import build_transcriber
from jarvis.core.speech import speakable, split_sentences
from jarvis.remote import protocol as P
from jarvis.remote.addressing import (
    MODE_LABEL,
    DeviceModes,
    HoldWindow,
    Mode,
    apply_mode_command,
    route,
)
from jarvis.remote.intake import AudioIntake

log = logging.getLogger(__name__)

#: A frame of silence handed to the wake-word role, which on this path is not a
#: detector at all — the addressing gate is.
_SILENT_FRAME = np.zeros(1280, dtype=np.int16)

#: How long a bad token costs *that address* before it may try again, doubling
#: per failure and cleared by a success. The point is to make guessing slow
#: without locking out the Pi in the hall after somebody fat-fingers an .env.
_AUTH_BACKOFF_S = 2.0
_AUTH_BACKOFF_MAX_S = 30.0

#: `clientLog` lines are capped by the protocol; this is how many a minute.
_CLIENT_LOG_PER_MIN = 30


class Cancelled(Exception):
    """Re-exported so callers don't have to know it comes from the interrupter."""


class EdgeConnection:
    """One edge socket, and what the brain knows about it."""

    def __init__(self, ws, device_id: str) -> None:
        self.ws = ws
        self.device_id = device_id
        #: only a connection that asked for speech gets `speech` messages, and
        #: it gets them on its own socket — never a broadcast, never replayed
        self.speech_on = False
        self.mic_on = True
        self.speech_sample_rate = 0
        self.log_budget = _CLIENT_LOG_PER_MIN
        self.log_window = time.monotonic()

    async def send(self, msg: dict) -> bool:
        try:
            await self.ws.send(json.dumps(msg))
            return True
        except Exception as exc:  # noqa: BLE001 — a closed socket is normal
            log.debug("send to %s failed: %s", self.device_id, exc)
            return False

    def allow_client_log(self) -> bool:
        now = time.monotonic()
        if now - self.log_window > 60:
            self.log_window = now
            self.log_budget = _CLIENT_LOG_PER_MIN
        if self.log_budget <= 0:
            return False
        self.log_budget -= 1
        return True


@dataclass
class DeviceSession:
    """Per-connection state the server owns: the mode, the hold window, and the
    task watching it."""

    connection: EdgeConnection
    mode: str
    previous_mode: str | None
    window: HoldWindow
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


# -- the link --------------------------------------------------------------


class RemoteLink:
    """``wake`` + ``mic`` + ``stt`` + ``tts``, backed by an edge over a socket.

    One object satisfies all four roles because the orchestrator only ever
    reaches each through its own attribute — the same trick ``TextIO`` uses.
    """

    def __init__(self, config, *, voder=None) -> None:
        self.config = config
        self.voder = voder if voder is not None else build_voder(
            config.voder.url,
            sample_rate=config.voder.sample_rate,
            timeout_s=config.voder.timeout_s,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connection: EdgeConnection | None = None
        self._orchestrator = None

        #: finished utterances waiting for the orchestrator, oldest first
        self._utterances: queue.Queue[str] = queue.Queue()
        self._pending: str | None = None       # popped, not yet "transcribed"
        self._arrival = threading.Event()      # something to read
        self._waiting = False                  # `record_utterance` is blocked
        # "Something the user said is on its way" — see `incoming`.
        self._speaking_now = False
        self._transcribing = 0
        self._holding = False
        #: once words are in flight, wait at least this long past each sign of
        #: life: the hangover, plus Whisper, plus the hold window, plus slack
        self.inflight_extend_s = max(4.0, config.addressing.hold_ms / 1000 + 3.0)
        #: and an absolute ceiling for one capture, so an edge whose detector
        #: jams open cannot hold a turn until the process is killed
        self.inflight_max_s = 60.0

        # speaking
        self._speech_seq = 0
        self._playing: str | None = None
        self._played = threading.Event()
        self._interrupted = threading.Event()
        self._disconnected = threading.Event()
        self.last_avg_logprob: float | None = None
        self.state = "idle"
        #: when the utterance reached the orchestrator, for the per-turn timing
        #: line — the one number that says whether a slow turn was the skill,
        #: the voder or neither
        self._turn_started: float | None = None

    # -- wiring (event loop side) -----------------------------------------

    def attach(self, orchestrator) -> None:
        self._orchestrator = orchestrator

    @property
    def conversation(self) -> bool:
        """The conversation window: ByName accepts the next utterance *without*
        the name while this is open.

        It is open for the whole of a wake session, not merely while
        ``record_utterance`` happens to be blocked. The difference is the user
        answering while the reply is still playing, or while an ``_ask`` prompt
        is being spoken: with the narrower rule their answer arrives a fraction
        of a second before JARVIS starts listening for it, and is dropped for
        not saying the name — which is exactly the moment saying the name feels
        most absurd.
        """
        if self._waiting:
            return True
        orchestrator = self._orchestrator
        return bool(orchestrator is not None and orchestrator.running and not orchestrator.standby)

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()

    @property
    def connection(self) -> EdgeConnection | None:
        return self._connection

    def connect(self, connection: EdgeConnection) -> None:
        self._connection = connection
        self._disconnected.clear()

    def disconnect(self, connection: EdgeConnection) -> None:
        if self._connection is not connection:
            return
        self._connection = None
        # A closed socket cannot report that it stopped talking, so nothing is
        # on its way any more.
        self._speaking_now = False
        self._transcribing = 0
        self._holding = False
        # A pending `record_utterance` ends the session cleanly rather than
        # waiting out its window against a socket that is gone.
        self._disconnected.set()
        self._interrupted.set()
        self._played.set()
        self._arrival.set()
        self.cancel("the edge disconnected")

    @property
    def incoming(self) -> bool:
        """Something the user said is on its way, and the listening window must
        not give up on it.

        Three ways that is true, and all three matter: the edge's detector is
        open (they are talking *now*), a segment is in Whisper (they have
        stopped, the words are in flight), or the hold window is holding a
        fragment (they paused mid-thought). A window that counted only the
        clock would lapse during any of them — which it did, in a live run,
        with the transcript already logged on the same screen as "Standing by,
        sir."
        """
        return self._speaking_now or self._transcribing > 0 or self._holding

    def set_speaking(self, on: bool) -> None:
        self._speaking_now = bool(on)

    def begin_transcribe(self) -> None:
        self._transcribing += 1

    def end_transcribe(self) -> None:
        self._transcribing = max(0, self._transcribing - 1)

    def set_holding(self, holding: bool) -> None:
        self._holding = bool(holding)

    def submit(self, text: str) -> None:
        """An utterance got through the gate and the hold window."""
        text = (text or "").strip()
        if not text:
            return
        self._turn_started = time.monotonic()
        self._utterances.put(text)
        self._arrival.set()

    def interrupt(self) -> None:
        """The edge's button, or a spoken "stop"."""
        self._interrupted.set()
        self._played.set()
        self.cancel("interrupted")

    def cancel(self, reason: str) -> None:
        orchestrator = self._orchestrator
        if orchestrator is not None:
            orchestrator.interrupter.trigger_cancel(reason)

    def playback_done(self, speech_id: str) -> None:
        if self._playing is None or speech_id == self._playing:
            self._played.set()

    # -- sending from a worker thread -------------------------------------

    def _send(self, msg: dict) -> None:
        """Fire-and-forget from whichever thread. Never raises: a message the
        edge didn't get is not worth ending a turn over."""
        loop, connection = self._loop, self._connection
        if loop is None or connection is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(connection.send(msg), loop)
        except RuntimeError:  # the loop is closing
            pass

    def set_state(self, value: str) -> None:
        """`idle` / `listening` / `thinking` / `acting` / `speaking`, plus the
        mode, which a client watching only `state` would otherwise never see."""
        self.state = value
        connection = self._connection
        if connection is None:
            return
        mode = self._mode_of(connection.device_id)
        self._send(P.state(value, mode, connection.mic_on))

    def _mode_of(self, device_id: str) -> str:
        modes = getattr(self, "modes", None)
        if modes is None:
            return self.config.addressing.default_mode
        return modes.get(device_id).mode

    # -- as "wake" ---------------------------------------------------------

    def reset(self) -> None:
        pass

    def triggered(self, frame) -> bool:
        """True when an *addressed* utterance is queued — the equivalent of the
        wake word firing. Un-addressed speech never reaches the queue: the gate
        dropped it before this."""
        return not self._utterances.empty()

    # -- as "mic" ----------------------------------------------------------

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def drain(self) -> None:
        """Half-duplex housekeeping the local mic needs and this doesn't: while
        JARVIS speaks, the edge isn't feeding its segmenter at all."""

    def read(self, timeout: float | None = None):
        """``_await_wake``'s loop runs unchanged; there is no audio to look at,
        so this just paces it and returns silence."""
        self._arrival.wait(timeout if timeout is not None else 1.0)
        self._arrival.clear()
        return _SILENT_FRAME

    def record_utterance(self, window, silence_s, grace, cancel_flag=None, min_speech=3):
        """Pop the next queued utterance, waiting up to ``grace`` for it.

        ``grace`` is the orchestrator's "how long to wait for them to **start**
        talking" — to a microphone, that is all it ever meant, because
        ``record_utterance`` there returns only once the utterance is over, and
        a long sentence costs the window nothing.

        The same has to be true here, and it is not automatic: the words arrive
        as a finished transcript, so their speaking time, the segmenter's
        hangover, Whisper and the hold window would all be charged to the
        window. So while :attr:`incoming` says something is on its way, the
        deadline keeps moving — bounded by :attr:`inflight_max_s`, because an
        edge whose detector jams open must not hold the turn for ever.

        ``window`` and ``silence_s`` belong to a microphone and mean nothing
        here. Returns non-empty dummy audio (as ``TextIO`` does) when there is
        an utterance, and silence when the window lapsed.
        """
        started = time.monotonic()
        deadline = started + max(0.0, float(grace))
        ceiling = deadline + self.inflight_max_s
        self._waiting = True
        try:
            while True:
                if cancel_flag is not None and cancel_flag.is_set():
                    return np.zeros(0, dtype=np.float32)
                if self._disconnected.is_set():
                    return np.zeros(0, dtype=np.float32)
                try:
                    self._pending = self._utterances.get(timeout=0.1)
                    return np.ones(1600, dtype=np.float32) * 0.1
                except queue.Empty:
                    pass
                now = time.monotonic()
                if self.incoming:
                    deadline = min(ceiling, max(deadline, now + self.inflight_extend_s))
                if now >= deadline or now >= ceiling:
                    return np.zeros(0, dtype=np.float32)
        finally:
            self._waiting = False

    # -- as "stt" ----------------------------------------------------------

    def transcribe(self, audio, cancel=None) -> str:
        """Already transcribed, out on the GPU box, before the gate saw it."""
        text, self._pending = self._pending, None
        return text or ""

    # -- as "tts" ----------------------------------------------------------

    def say(self, text: str) -> None:
        """Speak one answer: text first, then sentence by sentence to the voder
        and out to the edge. Blocks until the edge says it finished playing, or
        until an interrupt — which keeps the orchestrator's half-duplex
        assumption true over a network.

        Never raises. A disconnect or an interrupt mid-answer drops the unsent
        sentences and returns; the *next* ``record_utterance`` is what ends the
        session, so the orchestrator's own Cancelled handling stays the only
        path out.
        """
        text = (text or "").strip()
        if not text:
            return
        print(f"Jarvis: {text}", flush=True)  # the legacy visible transcript
        connection = self._connection
        if connection is None:
            return

        spoken = speakable(text)
        self._send(P.text(spoken or text))
        if not connection.speech_on or not getattr(self.voder, "available", False):
            return

        self.set_state("speaking")
        self._speech_seq += 1
        speech_id = f"s{self._speech_seq}"
        self._playing = speech_id
        self._played.clear()
        self._interrupted.clear()

        parts = split_sentences(spoken)
        seconds = 0.0
        sent_final = False
        said_at = time.monotonic()
        first_part_ms: int | None = None
        for i, part in enumerate(parts):
            if self._interrupted.is_set() or self._connection is None:
                break
            out = self.voder.speak_or_none(part)
            if out is None:  # the voder went away mid-answer: text only
                break
            pcm, rate = out
            seconds += len(pcm) / max(1, rate)
            final = i == len(parts) - 1
            self._send(P.speech(speech_id, i, part, P.encode_pcm(pcm), rate, final))
            sent_final = final
            if first_part_ms is None:
                first_part_ms = int((time.monotonic() - said_at) * 1000)
                if self._turn_started is not None:
                    log.info(
                        "turn: utterance -> answer %dms, -> first speech %dms "
                        "(%d sentence(s))",
                        int((said_at - self._turn_started) * 1000),
                        int((time.monotonic() - self._turn_started) * 1000),
                        len(parts),
                    )
        if not sent_final:
            # Whatever was left is not coming: tell the edge so it doesn't wait
            # for a `final` it will never get.
            self._send(P.event("stopPlayback", {"id": speech_id}))
            self._playing = None
            return

        # Wait for the edge to finish playing. The timeout is the audio's own
        # length plus slack: a wedged edge must not hold the turn for ever.
        self._played.wait(timeout=min(seconds + 10.0, 120.0))
        self._playing = None

    def close(self) -> None:
        pass


# -- the server ------------------------------------------------------------


def build_ssl_context(config) -> ssl.SSLContext | None:
    """TLS for the listener, or ``None`` when there is none to build.

    Refuses a non-loopback bind without TLS: the link is assumed to be on the
    public internet, and unencrypted speech on it is not a trade-off anybody
    chose, it is one they didn't notice.
    """
    server = config.server
    if server.tls_enabled:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(server.tls_cert, server.tls_key)
        return context
    if _is_loopback(server.host):
        return None  # a reverse proxy in front, or a dev box
    if server.allow_insecure:
        log.warning(
            "!! [server] allow_insecure: listening on %s with no TLS. Everything "
            "said in the room crosses the network in clear. LAN and development "
            "only — set tls_cert/tls_key, or listen on 127.0.0.1 behind a proxy.",
            server.host,
        )
        return None
    raise RuntimeError(
        f"refusing to listen on {server.host}:{server.port} without TLS. Set "
        f"[server] tls_cert/tls_key, bind 127.0.0.1 behind a TLS-terminating "
        f"reverse proxy, or (LAN/dev only) set allow_insecure = true."
    )


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost", "")


class RemoteServer:
    """The WebSocket listener: auth, the addressing gate, the hold window, and
    the intake. Everything it decides ends up as one call into ``RemoteLink``."""

    def __init__(
        self,
        config,
        link: RemoteLink,
        *,
        transcriber=None,
        modes: DeviceModes | None = None,
        tokens: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.link = link
        self.transcriber = transcriber if transcriber is not None else build_transcriber(
            config.whisper.url, config.whisper.timeout_s
        )
        self.modes = modes if modes is not None else DeviceModes(
            config.remote_dir, config.addressing.default_mode
        )
        link.modes = self.modes
        self.tokens = tokens if tokens is not None else dict(config.edge_tokens)
        # Parsed here so a mistyped CIDR stops the brain at startup rather than
        # quietly disabling the thing that tells clients apart.
        self._trusted_networks = config.server.trusted_networks()
        self.sessions: dict[str, DeviceSession] = {}
        #: client key -> when it may try again, and how many times it has failed
        self._backoff: dict[str, float] = {}
        self._failures: dict[str, int] = {}

    # -- auth ----------------------------------------------------------

    def client_key(self, ws) -> str:
        """Who this connection is, for the backoff and the log.

        Normally the socket's own address. Behind something that terminates TLS
        in front of the brain — a Cloudflare Tunnel, a reverse proxy — every
        connection arrives from 127.0.0.1 instead, and keying the backoff on
        that would let one stranger guessing tokens lock out the real edge. So
        when ``[server] trusted_proxy_header`` is set **and the connection came
        from loopback**, the header wins.

        The trust condition is the whole of the security argument: only
        something that sits in front of the brain can set a header we did not
        put there ourselves. From anywhere else the header is a claim the client
        made about itself, and believing it would hand every guesser a way to
        wipe their own backoff by inventing an address.

        Loopback is trusted implicitly (a proxy on this machine). A proxy that
        is *not* on loopback — ``cloudflared`` in a container reaching the brain
        over the Docker bridge, or a proxy on another host — has to be named in
        ``[server] trusted_proxy_peers``.
        """
        try:
            direct = str(ws.remote_address[0])
        except Exception:  # noqa: BLE001
            return "?"
        header = self.config.server.trusted_proxy_header
        if header and self._peer_is_trusted(direct):
            try:
                value = ws.request.headers.get(header) or ""
            except Exception:  # noqa: BLE001
                value = ""
            # X-Forwarded-For is a chain; the client is the first entry.
            first = value.split(",")[0].strip()
            if first:
                return first
        return direct

    def _peer_is_trusted(self, address: str) -> bool:
        """May this peer tell us who the real client is?"""
        if _is_loopback(address):
            return True
        if not self._trusted_networks:
            return False
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return False
        return any(parsed in network for network in self._trusted_networks)

    def _prune_backoff(self) -> None:
        """Forget addresses whose backoff has long passed. On the open internet
        the key space is every address there is, so this table cannot be allowed
        to grow for the life of the process."""
        if len(self._backoff) < 256:
            return
        now = time.monotonic()
        stale = [key for key, until in self._backoff.items() if until + 3600 < now]
        for key in stale:
            self._backoff.pop(key, None)
            self._failures.pop(key, None)

    def _authorise(self, msg: dict) -> str | None:
        """The token for this device, compared in constant time. Returns an
        error string, or ``None`` when it is allowed."""
        device_id = msg.get("device_id", "")
        expected = self.tokens.get(device_id)
        if not expected:
            # Same answer either way: "no such device" and "wrong token" are
            # the same sentence to whoever is guessing.
            return "unauthorized"
        if not hmac.compare_digest(str(msg.get("token", "")), str(expected)):
            return "unauthorized"
        return None

    # -- the connection ------------------------------------------------

    async def handle(self, ws) -> None:
        peer = self.client_key(ws)
        self._prune_backoff()
        if time.monotonic() < self._backoff.get(peer, 0.0):
            await ws.close(P.CLOSE.UNAUTHORIZED, "slow down")
            return

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=self.config.server.hello_timeout_s)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            await ws.close(P.CLOSE.BAD_PROTOCOL, "no hello")
            return

        ok, msg = P.validate_c2s(raw)
        if not ok or msg.get("type") != P.C2S.HELLO:
            # Before `hello`, a malformed message closes the socket.
            await ws.close(P.CLOSE.BAD_PROTOCOL, msg if isinstance(msg, str) else "hello first")
            return
        if msg["protocol"] != P.PROTOCOL_VERSION:
            await ws.close(
                P.CLOSE.BAD_PROTOCOL,
                f"protocol {msg['protocol']}, this brain speaks {P.PROTOCOL_VERSION}",
            )
            return
        denial = self._authorise(msg)
        if denial:
            # Per address, and doubling: the point is to make guessing slow,
            # not to lock out the Pi in the hall after somebody fat-fingers
            # an .env.
            failures = self._failures[peer] = self._failures.get(peer, 0) + 1
            step = min(_AUTH_BACKOFF_MAX_S, _AUTH_BACKOFF_S * 2 ** (failures - 1))
            self._backoff[peer] = time.monotonic() + step
            log.warning(
                "refused %s from %s (attempt %d, %.0fs)",
                msg.get("device_id"), peer, failures, step,
            )
            await ws.close(P.CLOSE.UNAUTHORIZED, denial)
            return
        self._backoff.pop(peer, None)
        self._failures.pop(peer, None)

        device_id = msg["device_id"]
        existing = self.link.connection
        if existing is not None and existing.device_id != device_id:
            # One edge per brain for now, and saying so is better than two
            # rooms fighting over one conversation.
            await ws.close(P.CLOSE.UNAUTHORIZED, "another device is connected")
            return
        if existing is not None:
            log.info("edge %s reconnected — replacing the old socket", device_id)
            await existing.ws.close(P.CLOSE.SERVER_SHUTDOWN, "replaced")

        connection = EdgeConnection(ws, device_id)
        stored = self.modes.get(device_id)
        session = DeviceSession(
            connection=connection,
            mode=stored.mode,
            previous_mode=stored.previous_mode,
            window=HoldWindow(self.config.addressing.hold_ms),
        )
        self.sessions[device_id] = session
        self.link.bind_loop()
        self.link.connect(connection)
        session.task = asyncio.ensure_future(self._hold_loop(session))
        log.info("edge %s connected from %s (mode %s)", device_id, peer, session.mode)
        await connection.send(P.ready(session.mode, speech=False))
        await connection.send(P.state(self.link.state, session.mode, connection.mic_on))

        intake = AudioIntake(
            lambda pcm: asyncio.to_thread(self.transcriber.transcribe_full, pcm),
            send=connection.send,
        )
        try:
            async for raw in ws:
                await self._message(session, intake, raw)
        except Exception as exc:  # noqa: BLE001 — a dropped link is not an error
            log.info("edge %s disconnected: %s", device_id, type(exc).__name__)
        finally:
            session.task.cancel()
            try:
                await session.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            session.window.clear()
            self.sessions.pop(device_id, None)
            self.link.disconnect(connection)
            log.info("edge %s gone", device_id)

    async def _message(self, session: DeviceSession, intake: AudioIntake, raw) -> None:
        connection = session.connection
        ok, msg = P.validate_c2s(raw)
        if not ok:
            # After `hello`, a bad frame gets an error and the socket stays up:
            # one malformed message shouldn't cost the user their conversation.
            await connection.send(P.error(msg))
            return

        kind = msg["type"]
        if kind == P.C2S.AUDIO:
            # Held open across the transcription: the user has stopped talking,
            # but their words are in flight and the listening window must not
            # lapse while Whisper has them.
            self.link.begin_transcribe()
            try:
                heard = await intake.receive(msg)
            finally:
                self.link.end_transcribe()
            if heard is not None:
                await self._admit(session, heard.text)
            return
        if kind == P.C2S.SPEAKING:
            session.window.speaking(msg["on"])
            self.link.set_speaking(msg["on"])
            session.wake.set()
            return
        if kind == P.C2S.INTERRUPT:
            log.info("interrupt from %s", connection.device_id)
            session.window.clear()
            self.link.set_holding(False)
            self.link.interrupt()
            return
        if kind == P.C2S.CONTROL:
            await self._control(session, msg["action"], msg.get("args") or {})
            return

    async def _control(self, session: DeviceSession, action: str, args: dict) -> None:
        connection = session.connection
        if action == P.CONTROL.SET_SPEECH:
            connection.speech_on = bool(args.get("on", True))
            rate = args.get("sample_rate")
            if isinstance(rate, int) and 0 < rate <= 48000:
                connection.speech_sample_rate = rate
            await connection.send(P.ready(session.mode, connection.speech_on))
            return
        if action == P.CONTROL.MIC:
            connection.mic_on = bool(args.get("on", True))
            await connection.send(P.state(self.link.state, session.mode, connection.mic_on))
            return
        if action == P.CONTROL.PLAYBACK_DONE:
            self.link.playback_done(str(args.get("id", "")))
            self.link.set_state("idle")
            return
        if action == P.CONTROL.SET_MODE:
            await self._set_mode(session, args.get("mode"))
            return
        if action == P.CONTROL.CLIENT_LOG:
            if connection.allow_client_log():
                level = str(args.get("level", "info"))
                line = str(args.get("text", ""))[: P.MAX_CLIENT_LOG]
                log.log(
                    logging.WARNING if level in ("warn", "warning", "error") else logging.INFO,
                    "[%s] %s", connection.device_id, line,
                )
            return
        await connection.send(P.error(f'unknown control action "{action}"'))

    # -- the gate ------------------------------------------------------

    async def _admit(self, session: DeviceSession, text: str) -> None:
        """One transcript, through the mode commands and then the gate."""
        connection = session.connection
        routed = route(
            text,
            mode=session.mode,
            names=self.config.addressing.names,
            # The conversation window: the orchestrator is listening for a
            # reply, or a fragment of this thought is already being held.
            conversation=self.link.conversation or session.window.held,
        )

        if routed.kind == "mode":
            await self._set_mode(session, routed.to)
            return
        if routed.kind == "mic":
            connection.mic_on = routed.on
            await connection.send(P.event("mic", {"on": routed.on}))
            await connection.send(P.state(self.link.state, session.mode, connection.mic_on))
            log.info("mic %s on %s", "on" if routed.on else "off", connection.device_id)
            return
        if routed.kind == "dropped":
            # `heard` already went out: "it heard me and decided I wasn't
            # talking to it" and "it didn't hear me" are different problems.
            log.debug("dropped (%s): %r", routed.reason, routed.text or text)
            return
        if routed.kind != "jarvis":
            return

        if routed.bare:
            # The name on its own. Nothing to hold and nothing to merge it
            # with: let the orchestrator answer it now.
            self.link.submit(text.strip())
            await connection.send(P.state("thinking", session.mode, connection.mic_on))
            return

        session.window.add(routed.text)
        self.link.set_holding(True)
        session.wake.set()
        # Still *listening* — words are being held — as opposed to thinking,
        # which is the orchestrator having the whole utterance.
        await connection.send(P.state("listening", session.mode, connection.mic_on))

    async def _set_mode(self, session: DeviceSession, to) -> None:
        connection = session.connection
        before = session.mode
        session.mode, session.previous_mode = apply_mode_command(
            to, session.mode, session.previous_mode
        )
        self.modes.set(connection.device_id, session.mode, session.previous_mode)
        if session.mode != Mode.PUSHTOTALK:
            connection.mic_on = True
        await connection.send(
            P.event("modeChanged", {"mode": session.mode, "previous": before})
        )
        # One short line, because this is the confirmation that the user is not
        # talking into a paused microphone.
        await connection.send(P.text(f"Input: {MODE_LABEL[session.mode]}.", "system"))
        await connection.send(P.state(self.link.state, session.mode, connection.mic_on))
        log.info("mode %s -> %s on %s", before, session.mode, connection.device_id)

    # -- the hold window -----------------------------------------------

    async def _hold_loop(self, session: DeviceSession) -> None:
        """One timer per connection: wait exactly as long as the window says it
        has left, and release what is held when it runs out."""
        while True:
            due = session.window.due_in()
            if due is None:
                await session.wake.wait()
                session.wake.clear()
                continue
            if due > 0:
                try:
                    await asyncio.wait_for(session.wake.wait(), timeout=due)
                except asyncio.TimeoutError:
                    pass
                session.wake.clear()
                continue
            text = session.window.release()
            self.link.set_holding(False)
            if text:
                log.info("utterance -> %r", text)
                self.link.submit(text)
                await session.connection.send(
                    P.state("thinking", session.mode, session.connection.mic_on)
                )

    # -- listening -----------------------------------------------------

    async def serve(self):
        """Start listening. Returns the ``websockets`` server object."""
        import websockets

        context = build_ssl_context(self.config)
        server = await websockets.serve(
            self.handle,
            self.config.server.host,
            self.config.server.port,
            ssl=context,
            max_size=self.config.server.max_size,
            ping_interval=self.config.server.ping_interval_s,
            ping_timeout=self.config.server.ping_interval_s,
        )
        scheme = "wss" if context else "ws"
        log.info(
            "listening on %s://%s:%d", scheme, self.config.server.host, self.config.server.port
        )
        return server
