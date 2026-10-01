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
import base64
import binascii
import dataclasses
import hmac
import ipaddress
import itertools
import json
import logging
import queue
import re
import secrets
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

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
from jarvis.remote.firmware import Firmware, FirmwareStore, refusal
from jarvis.remote import pairing, powerlog
from jarvis.remote.intake import AudioIntake
from jarvis.skills.edge import EdgeTools

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

#: How long an edge has to answer a `call`. A tool does something quick and
#: says so (the ringing, the timer itself, run on after the answer).
CALL_TIMEOUT_S = 8.0
#: The edge tool a notification goes to, and how many wait for an edge that
#: is away (the oldest go first).
NOTIFY_TOOL = "notify"
NOTIFY_QUEUE = 20
#: and how long one may be: it is shown on a watch
MAX_NOTIFY_TEXT = 300
#: The edge tool that sends the power log up (``fetch_power_log``), and how
#: long the whole upload may take: a full day is under 1 MB.
POWER_TOOL = "send_power_log"
POWER_FETCH_TIMEOUT_S = 120.0


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
    #: the firmware version it reported in `hello`; None = cannot flash itself
    fw: str | None = None
    #: an update asked for by voice, announced once the reply has played
    pending_ota: Firmware | None = None
    #: `call`s sent and not yet answered: id -> the future its `result` resolves
    calls: dict[str, asyncio.Future] = field(default_factory=dict)
    #: a power log upload under way: resolved with the file count on `done`
    upload: asyncio.Future | None = None


@dataclass(frozen=True)
class FirmwareUpdate:
    """What ``RemoteServer.update_firmware`` did: ``sent`` (announced, or
    about to be), ``current`` (it runs that already), ``dev`` / ``newer`` (it
    runs a dev build or a later version, which it would refuse to replace —
    ``running`` says which), ``none`` (nothing staged), ``unsupported`` (the
    edge never said what it runs), ``offline``."""

    status: str
    version: str = ""
    device_id: str = ""
    running: str = ""


@dataclass
class PendingPair:
    """A device waiting for its code to be confirmed."""

    secret: pairing.Secret
    approved: asyncio.Future


@dataclass(frozen=True)
class ToolResult:
    """What a ``call`` to an edge tool came to: ``ok`` / ``failed`` (the edge
    answered, ``say`` is its line), ``timeout``, ``offline`` (no edge, or it
    went away), ``unsupported`` (the edge never declared that tool)."""

    status: str
    say: str = ""


@dataclass(frozen=True)
class PowerFetch:
    """What ``fetch_power_log`` came to: ``ok`` (``files`` came up),
    ``offline``, ``unsupported`` (no ``send_power_log``), ``timeout``,
    ``failed``. ``report``/``spoken`` read whatever the brain has for the
    day, fetched now or not."""

    status: str
    files: int = 0
    report: str = ""
    spoken: str = ""


class EdgeControl:
    """What a skill may ask of the connected edges (``ctx.edges``).

    Skills run on a worker thread (the orchestrator's ``asyncio.to_thread``),
    so each call hops onto the server's loop and waits for the answer. Never
    call it from the loop itself.
    """

    def __init__(self, server: "RemoteServer", timeout_s: float = 5.0) -> None:
        self._server = server
        self._timeout_s = timeout_s

    def update_firmware(self) -> FirmwareUpdate:
        loop = self._server.link._loop
        if loop is None or not loop.is_running():
            return FirmwareUpdate("offline")
        future = asyncio.run_coroutine_threadsafe(self._server.update_firmware(), loop)
        try:
            return future.result(self._timeout_s)
        except Exception:  # noqa: BLE001 — a timeout or a closing loop
            future.cancel()
            return FirmwareUpdate("offline")

    def power_report(self, day: str | None = None) -> PowerFetch:
        """Fetch the watch's power log and read the day (the watch_power skill)."""
        loop = self._server.link._loop
        if loop is None or not loop.is_running():
            return self._server.power_report(PowerFetch("offline"), day)
        future = asyncio.run_coroutine_threadsafe(self._server.fetch_power_log(day=day), loop)
        try:
            return future.result(POWER_FETCH_TIMEOUT_S + CALL_TIMEOUT_S + self._timeout_s)
        except Exception:  # noqa: BLE001 — a timeout or a closing loop
            future.cancel()
            return self._server.power_report(PowerFetch("timeout"), day)

    def call(self, tool: str, args: dict) -> ToolResult:
        """Run an edge tool (``jarvis/skills/edge.py``) and wait for its answer."""
        loop = self._server.link._loop
        if loop is None or not loop.is_running():
            return ToolResult("offline")
        future = asyncio.run_coroutine_threadsafe(self._server.call_tool(tool, args), loop)
        try:
            return future.result(CALL_TIMEOUT_S + self._timeout_s)
        except Exception:  # noqa: BLE001 — a timeout or a closing loop
            future.cancel()
            return ToolResult("offline")


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

    def entering_standby(self) -> None:
        """The orchestrator is dropping to standby: say so as a `state`, before
        the standby line goes out as speech. A battery edge powers down on it
        (and may skip that line — it still reports `playbackDone`)."""
        self.set_state("standby")

    def set_state(self, value: str) -> None:
        """`idle` / `listening` / `thinking` / `acting` / `speaking` /
        `standby`, plus the mode, which a client watching only `state` would
        otherwise never see."""
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
        #: tokens handed out by pairing (data/remote/tokens.json), next to .env's
        self.paired = pairing.TokenStore(config.remote_dir)
        #: devices waiting for their pairing code to be confirmed, by device id
        self._pending: dict[str, PendingPair] = {}
        self.firmware = FirmwareStore(config.firmware_dir)
        self.edges = EdgeControl(self)
        #: what each edge said it can do (jarvis/skills/edge.py)
        self.tools = EdgeTools(config.remote_dir)
        #: called (on the loop) with the device id when an edge's tool list
        #: changed — `serve` retrains the classifier then
        self.on_tools_changed = None
        #: notifications waiting for an edge that is away, per device
        self.outbox: dict[str, deque[str]] = {}
        self._call_ids = itertools.count(1)
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
        given = str(msg.get("token", ""))
        # .env's token, and the one pairing handed out: either will do. Both
        # are compared, each in constant time.
        expected = [t for t in (self.tokens.get(device_id), self.paired.get(device_id)) if t]
        matches = [hmac.compare_digest(given, str(t)) for t in expected]
        # Same answer either way: "no such device" and "wrong token" are the
        # same sentence to whoever is guessing.
        return None if any(matches) else "unauthorized"

    def _refused(self, peer: str, device_id) -> None:
        """Per address, and doubling: the point is to make guessing slow, not
        to lock out the Pi in the hall after somebody fat-fingers an .env."""
        failures = self._failures[peer] = self._failures.get(peer, 0) + 1
        step = min(_AUTH_BACKOFF_MAX_S, _AUTH_BACKOFF_S * 2 ** (failures - 1))
        self._backoff[peer] = time.monotonic() + step
        log.warning("refused %s from %s (attempt %d, %.0fs)", device_id, peer, failures, step)

    # -- firmware, over plain HTTP on the same port -----------------------

    async def process_request(self, connection, request):
        """``GET /firmware``: the device's staged OTA image (``firmware.py``).

        Same port, same tunnel, same token as the WebSocket — sent as
        ``Authorization: Bearer <token>`` with ``X-Jarvis-Device: <id>`` — and
        the same per-address backoff, so this is not a second door for
        guessing. Every other path goes on to the WebSocket handshake.
        """
        from http import HTTPStatus

        from websockets.datastructures import Headers
        from websockets.http11 import Response

        path = request.path.split("?", 1)[0]
        if path == P.PAIR_PATH:
            return self._http_pair(connection, request)
        if path not in (P.FIRMWARE_PATH, P.NOTIFY_PATH, P.POWER_PATH):
            return None
        device_id, refusal = self._http_auth(connection, request)
        if refusal is not None:
            return refusal
        if path == P.NOTIFY_PATH:
            return await self._http_notify(connection, request, device_id)
        if path == P.POWER_PATH:
            return await self._http_power(connection, request, device_id)

        image = self.firmware.get(device_id)
        if image is None:
            return connection.respond(HTTPStatus.NOT_FOUND, "no firmware staged\n")
        try:
            body = await asyncio.to_thread(image.path.read_bytes)
        except OSError:
            return connection.respond(HTTPStatus.NOT_FOUND, "no firmware staged\n")
        log.info("serving firmware %s (%d bytes) to %s", image.version, len(body), device_id)
        headers = Headers(
            [
                ("Content-Type", "application/octet-stream"),
                ("Content-Length", str(len(body))),
                # Through a tunnel: never let a cache hand this to anyone else.
                ("Cache-Control", "no-store"),
                ("X-Firmware-Version", image.version),
                ("X-Firmware-Sha256", image.sha256),
                ("Connection", "close"),
            ]
        )
        return Response(200, "OK", headers, body)

    def _http_auth(self, connection, request):
        """``(device_id, None)``, or ``(None, the refusal to send)``: the
        device's own token as ``Authorization: Bearer``, named by
        ``X-Jarvis-Device``, with the WebSocket's per-address backoff."""
        from http import HTTPStatus

        peer = self.client_key(connection)
        self._prune_backoff()
        if time.monotonic() < self._backoff.get(peer, 0.0):
            return None, connection.respond(HTTPStatus.TOO_MANY_REQUESTS, "slow down\n")
        device_id = request.headers.get("X-Jarvis-Device", "")
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or self._authorise(
            {"device_id": device_id, "token": token.strip()}
        ):
            self._refused(peer, device_id or "?")
            return None, connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        self._backoff.pop(peer, None)
        self._failures.pop(peer, None)
        return device_id, None

    async def _http_notify(self, connection, request, device_id: str):
        """``GET /notify?text=...``: a notification for that device's edge
        (``python -m jarvis notify``). A GET, because this port speaks the
        WebSocket handshake and nothing with a body."""
        from http import HTTPStatus

        query = parse_qs(urlsplit(request.path).query)
        text = " ".join((query.get("text") or [""])[0].split())[:MAX_NOTIFY_TEXT]
        if not text:
            return connection.respond(HTTPStatus.BAD_REQUEST, "text= is empty\n")
        status = await self.notify(text, device_id)
        log.info("notify for %s: %s", device_id, status)
        return connection.respond(HTTPStatus.OK, f"{status}\n")

    def _http_pair(self, connection, request):
        """``GET /pair?code=NNNNNN``: confirm a device's pairing code. Takes
        the admin secret (``data/remote/admin.token``), never a device token:
        approving a new device is the brain's owner's call."""
        from http import HTTPStatus

        peer = self.client_key(connection)
        self._prune_backoff()
        if time.monotonic() < self._backoff.get(peer, 0.0):
            return connection.respond(HTTPStatus.TOO_MANY_REQUESTS, "slow down\n")
        scheme, _, given = request.headers.get("Authorization", "").partition(" ")
        admin = pairing.admin_secret(self.config.remote_dir)
        if scheme.lower() != "bearer" or not hmac.compare_digest(given.strip(), admin):
            self._refused(peer, "admin")
            return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        self._backoff.pop(peer, None)
        self._failures.pop(peer, None)
        code = "".join(ch for ch in (parse_qs(urlsplit(request.path).query).get("code") or [""])[0] if ch.isdigit())
        device_id = self.approve_pairing(code)
        if device_id is None:
            return connection.respond(HTTPStatus.NOT_FOUND, "no device is waiting with that code\n")
        return connection.respond(HTTPStatus.OK, f"paired: {device_id}\n")

    # -- pairing -------------------------------------------------------

    #: how long a device waits for its code to be confirmed, and how many may
    PAIR_TIMEOUT_S = 120.0
    MAX_PENDING_PAIRS = 4

    def approve_pairing(self, code: str) -> str | None:
        """The device showing this code, approved; None if none is."""
        for device_id, pending in self._pending.items():
            if hmac.compare_digest(pending.secret.code, code) and not pending.approved.done():
                pending.approved.set_result(True)
                return device_id
        return None

    async def _pair(self, ws, peer: str, msg: dict) -> None:
        """A device with no token: exchange keys, wait for its code to be
        confirmed, then send it a token sealed with the shared key."""
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        device_id = msg["device_id"]
        if device_id not in self._pending and len(self._pending) >= self.MAX_PENDING_PAIRS:
            await ws.close(P.CLOSE.PAIR_EXPIRED, "too many pairing requests")
            return
        device_pub = base64.b64decode(msg["key"])
        private = X25519PrivateKey.generate()
        brain_pub = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        try:
            shared = private.exchange(pairing.public_key(device_pub))
        except ValueError:  # a low-order point: not a key
            await ws.close(P.CLOSE.BAD_MESSAGE, "bad key")
            return
        secret = pairing.derive(shared, device_id, device_pub, brain_pub)
        pending = PendingPair(secret, asyncio.get_running_loop().create_future())
        old = self._pending.get(device_id)
        if old is not None and not old.approved.done():
            old.approved.set_result(False)  # a newer request from it replaces this one
        self._pending[device_id] = pending
        log.info(
            "pairing request from %s (%s): confirm the code it shows with "
            "`python -m jarvis pair <code>`", device_id, peer,
        )
        try:
            await ws.send(json.dumps(P.pairing(brain_pub, self.PAIR_TIMEOUT_S)))
            closed = asyncio.ensure_future(ws.wait_closed())
            done, _ = await asyncio.wait(
                {pending.approved, closed}, timeout=self.PAIR_TIMEOUT_S,
                return_when=asyncio.FIRST_COMPLETED,
            )
            closed.cancel()
            # Decided: from here on its code approves nothing, closing or not.
            if self._pending.get(device_id) is pending:
                del self._pending[device_id]
            if pending.approved not in done or not pending.approved.result():
                if pending.approved not in done:
                    log.info("pairing request from %s expired", device_id)
                    self._refused(peer, device_id)  # and slows down whoever is asking
                await ws.close(P.CLOSE.PAIR_EXPIRED, "pairing expired")
                return
            token = secrets.token_urlsafe(32)
            self.paired.set(device_id, token)
            nonce, box = pairing.seal(secret.key, device_id, token)
            await ws.send(json.dumps(P.paired(nonce, box)))
            log.info("paired %s", device_id)
            await ws.close(1000, "paired")
        except Exception as exc:  # noqa: BLE001 — the device went away
            log.info("pairing with %s ended: %s", device_id, type(exc).__name__)
        finally:
            if self._pending.get(device_id) is pending:
                del self._pending[device_id]

    async def _http_power(self, connection, request, device_id: str):
        """``GET /power?day=YYYY-MM-DD&fetch=1``: fetch the edge's power log
        (unless ``fetch=0``) and read the day (``python -m jarvis power``)."""
        from http import HTTPStatus

        query = parse_qs(urlsplit(request.path).query)
        day = (query.get("day") or [None])[0]
        if day is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", day):
            return connection.respond(HTTPStatus.BAD_REQUEST, "day=YYYY-MM-DD\n")
        if (query.get("fetch") or ["1"])[0] == "0":
            result = self.power_report(PowerFetch("skipped"), day, device_id)
        else:
            result = await self.fetch_power_log(day=day, device_id=device_id)
        head = (
            f"fetched {result.files} file(s)" if result.status == "ok"
            else f"not fetched: {result.status}"
        )
        return connection.respond(HTTPStatus.OK, f"{head}\n{result.report}\n")

    # -- the power log -------------------------------------------------

    def _power_dir(self, device_id: str) -> Path:
        return self.config.remote_dir / "power" / device_id

    def power_report(self, result: PowerFetch, day: str | None = None,
                     device_id: str | None = None) -> PowerFetch:
        """``result`` with the day's report added, from what the brain has."""
        device_id = device_id or next(iter(self.sessions), None) or self._power_device()
        day = day or time.strftime("%Y-%m-%d")
        path = self._power_dir(device_id or "?") / f"{day}.csv"
        if not path.is_file():
            return dataclasses.replace(
                result, report=f"no power log for {day}",
                spoken="There's nothing in the watch's power log for today yet.",
            )
        summary = powerlog.summarize(powerlog.read_rows([path]))
        return dataclasses.replace(
            result, report=powerlog.report(summary, f"Power log {day}"),
            spoken=powerlog.spoken(summary),
        )

    def _power_device(self) -> str | None:
        for device_id in self.tools.devices():
            if any(t["name"] == POWER_TOOL for t in self.tools.tools(device_id)):
                return device_id
        return next(iter(self.tokens), None)

    async def fetch_power_log(self, *, day: str | None = None, device_id: str | None = None,
                              timeout_s: float = POWER_FETCH_TIMEOUT_S) -> PowerFetch:
        """Ask the edge for the power log it has and we lack (``have``: our
        size of each file), then wait for its ``done``."""
        session = next(iter(self.sessions.values()), None)
        if session is None or (device_id and session.connection.device_id != device_id):
            return self.power_report(PowerFetch("offline"), day, device_id)
        device_id = session.connection.device_id
        folder = self._power_dir(device_id)
        have = {p.name: p.stat().st_size for p in folder.glob("*.csv")} if folder.is_dir() else {}
        session.upload = asyncio.get_running_loop().create_future()
        try:
            result = await self.call_tool(POWER_TOOL, {"have": have})
            if result.status != "ok":
                status = "failed" if result.status == "failed" else result.status
                return self.power_report(PowerFetch(status), day, device_id)
            files = await asyncio.wait_for(session.upload, timeout_s)
            if files < 0:
                return self.power_report(PowerFetch("offline"), day, device_id)
        except asyncio.TimeoutError:
            log.warning("power log from %s: no done within %.0f s", device_id, timeout_s)
            return self.power_report(PowerFetch("timeout"), day, device_id)
        finally:
            session.upload = None
        log.info("power log from %s: %d file(s)", device_id, files)
        return self.power_report(PowerFetch("ok", files=files), day, device_id)

    def _file(self, session: DeviceSession, msg: dict) -> None:
        """One ``file`` message: a chunk written at its offset, or ``done``."""
        if msg.get("done"):
            if session.upload is not None and not session.upload.done():
                session.upload.set_result(int(msg.get("files", 0)))
            return
        try:
            data = base64.b64decode(msg["b64"], validate=True)
        except (binascii.Error, ValueError):
            log.warning("file %s: bad base64", msg["name"])
            return
        folder = self._power_dir(session.connection.device_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / msg["name"]
        size = path.stat().st_size if path.exists() else 0
        offset = msg["offset"]
        if offset > size:
            # A gap: we lost a chunk. The next fetch asks from `size` again.
            log.warning("file %s: chunk at %d, but we have %d bytes", msg["name"], offset, size)
            return
        with open(path, "r+b" if path.exists() else "wb") as f:
            f.seek(offset)
            f.write(data)
            f.truncate()

    # -- edge tools ----------------------------------------------------

    async def call_tool(
        self, tool: str, args: dict, *, timeout_s: float = CALL_TIMEOUT_S
    ) -> ToolResult:
        """Send ``call`` to the connected edge and wait for its ``result``."""
        session = next(iter(self.sessions.values()), None)  # one edge per brain
        if session is None:
            return ToolResult("offline")
        device_id = session.connection.device_id
        if tool not in {t["name"] for t in self.tools.tools(device_id)}:
            return ToolResult("unsupported")
        call_id = f"c{next(self._call_ids)}"
        future = asyncio.get_running_loop().create_future()
        session.calls[call_id] = future
        try:
            if not await session.connection.send(P.call(call_id, tool, args)):
                return ToolResult("offline")
            log.info("call %s %s(%s) on %s", call_id, tool, args, device_id)
            return await asyncio.wait_for(future, timeout_s)
        except asyncio.TimeoutError:
            log.warning("call %s %s: %s did not answer", call_id, tool, device_id)
            return ToolResult("timeout")
        finally:
            session.calls.pop(call_id, None)

    def _result(self, session: DeviceSession, msg: dict) -> None:
        future = session.calls.get(msg["id"])
        if future is None or future.done():
            log.debug("result %s: nobody is waiting for it", msg["id"])
            return
        future.set_result(ToolResult("ok" if msg["ok"] else "failed", say=msg.get("say", "")))

    def _notify_device(self) -> str | None:
        """The device a notification is for when nobody said: the one that
        declared ``notify``, else the only one there is."""
        for device_id in self.tools.devices():
            if any(t["name"] == NOTIFY_TOOL for t in self.tools.tools(device_id)):
                return device_id
        return next(iter(self.tokens), None)

    async def notify(self, text: str, device_id: str | None = None) -> str:
        """Show ``text`` on the edge: ``sent``, ``queued`` (it is away; it
        gets it when it connects), or ``unsupported`` (it has no ``notify``)."""
        device_id = device_id or next(iter(self.sessions), None) or self._notify_device()
        if device_id is None:
            return "unsupported"
        if device_id in self.sessions:
            result = await self.call_tool(NOTIFY_TOOL, {"text": text})
            if result.status in ("ok", "failed"):
                return "sent"
            if result.status == "unsupported":
                return "unsupported"
        self.outbox.setdefault(device_id, deque(maxlen=NOTIFY_QUEUE)).append(text)
        return "queued"

    async def _deliver_outbox(self, session: DeviceSession) -> None:
        """What was notified while the edge was away, oldest first."""
        queue = self.outbox.get(session.connection.device_id)
        while queue and self.sessions.get(session.connection.device_id) is session:
            result = await self.call_tool(NOTIFY_TOOL, {"text": queue[0]})
            if result.status not in ("ok", "failed"):
                break  # gone again, or no notify: keep the rest
            queue.popleft()

    # -- firmware, on request (the update_watch skill) --------------------

    #: if the reply never reports playbackDone, announce anyway after this
    OTA_ANNOUNCE_FALLBACK_S = 30.0

    async def update_firmware(self) -> FirmwareUpdate:
        """Offer the staged image to the connected edge now, not on its next
        connect. With speech on, the ``ota`` event waits for the reply to
        finish playing (``playbackDone``) so the download does not start while
        JARVIS is still saying so."""
        session = next(iter(self.sessions.values()), None)  # one edge per brain
        if session is None:
            return FirmwareUpdate("offline")
        device_id = session.connection.device_id
        if not session.fw:
            return FirmwareUpdate("unsupported", device_id=device_id)
        image = self.firmware.get(device_id)
        if image is None:
            return FirmwareUpdate("none", device_id=device_id)
        why_not = refusal(session.fw, image.version)
        if why_not == "current":
            return FirmwareUpdate("current", version=image.version, device_id=device_id)
        if why_not:
            log.info("edge %s runs %s; not offering %s (%s)", device_id, session.fw, image.version, why_not)
            return FirmwareUpdate(why_not, version=image.version, device_id=device_id, running=session.fw)
        log.info("edge %s runs %s; update to %s asked for", device_id, session.fw, image.version)
        session.pending_ota = image
        if not session.connection.speech_on:
            await self._announce_pending_ota(session)
        else:
            asyncio.get_running_loop().call_later(
                self.OTA_ANNOUNCE_FALLBACK_S,
                lambda: asyncio.ensure_future(self._announce_pending_ota(session)),
            )
        return FirmwareUpdate("sent", version=image.version, device_id=device_id)

    async def _announce_pending_ota(self, session: DeviceSession) -> None:
        image, session.pending_ota = session.pending_ota, None
        if image is not None and self.sessions.get(session.connection.device_id) is session:
            await session.connection.send(P.event("ota", image.announcement(P.FIRMWARE_PATH)))

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
        if ok and msg.get("type") == P.C2S.PAIR:
            if msg["protocol"] != P.PROTOCOL_VERSION:
                await ws.close(P.CLOSE.BAD_PROTOCOL, f"this brain speaks {P.PROTOCOL_VERSION}")
                return
            await self._pair(ws, peer, msg)
            return
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
            self._refused(peer, msg.get("device_id"))
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
            fw=msg.get("fw"),
        )
        self.sessions[device_id] = session
        if msg.get("tools") is not None and self.tools.save(device_id, msg["tools"]):
            if self.on_tools_changed is not None:
                self.on_tools_changed(device_id)
        self.link.bind_loop()
        self.link.connect(connection)
        session.task = asyncio.ensure_future(self._hold_loop(session))
        log.info("edge %s connected from %s (mode %s)", device_id, peer, session.mode)
        await connection.send(P.ready(session.mode, speech=False))
        await connection.send(P.state(self.link.state, session.mode, connection.mic_on))
        image = self.firmware.offer(device_id, msg.get("fw"))
        if image is not None:
            log.info("edge %s runs %s; offering firmware %s", device_id, msg["fw"], image.version)
            await connection.send(P.event("ota", image.announcement(P.FIRMWARE_PATH)))
        if self.outbox.get(device_id):
            # Not awaited here: the answers come in through the loop below.
            asyncio.ensure_future(self._deliver_outbox(session))

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
            for future in session.calls.values():
                if not future.done():
                    future.set_result(ToolResult("offline"))
            if session.upload is not None and not session.upload.done():
                session.upload.set_result(-1)  # gone mid-upload
            # A reconnect registers its new session before this (replaced)
            # socket's cleanup gets here: only remove our own.
            if self.sessions.get(device_id) is session:
                del self.sessions[device_id]
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
        if kind == P.C2S.RESULT:
            self._result(session, msg)
            return
        if kind == P.C2S.FILE:
            self._file(session, msg)
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
            await self._announce_pending_ota(session)
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
            process_request=self.process_request,
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
