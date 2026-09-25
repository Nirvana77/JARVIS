"""The brain starts its own services (M3).

``python -m jarvis serve`` is one command, not three terminals: it spawns
``jarvis-whisper`` and ``jarvis-voder`` and shuts them down again on the way
out. They stay separate *processes*, which is the whole point of them — a model
that wedges is one process to kill, and restarting the brain does not reload a
GPU model.

Two rules everything here turns on:

* **Adopt what is already answering.** A service under systemd, or one left
  behind by a brain that was killed rather than stopped, must not be started a
  second time on the same port. The health endpoint is the test, so it works
  whoever started it.
* **Stop only what we started.** Shutting the brain down must not take out a
  service it merely adopted — that one belongs to systemd, or to somebody else.

Nothing here is required: with ``autostart = false``, or a URL of ``"off"``, or
a service that simply fails to come up, the brain runs exactly as before and
says "Cannot hear you: the transcription service is not running" until it
appears. A service is a convenience, not a dependency.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

#: repo root — where `services/` lives
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: how long to let a service shut down politely before killing it
DEFAULT_STOP_GRACE_S = 5.0

#: how long `serve` waits for a service to answer before carrying on without it
DEFAULT_WAIT_S = 60.0

#: how many times the watchdog will revive one service before leaving it down
DEFAULT_MAX_RESTARTS = 3


@dataclass
class ManagedService:
    """One service the brain may start: what to run, and how to tell it's up."""

    name: str
    url: str
    argv: list[str]
    #: True when the service answers its health endpoint
    health: Callable[[], bool]
    autostart: bool = True
    #: passed to the child as its working directory
    cwd: Path = field(default=_REPO_ROOT)

    @property
    def off(self) -> bool:
        return not self.url or self.url.strip().lower() == "off"


class ServiceSupervisor:
    """Start, adopt and stop a set of :class:`ManagedService`."""

    def __init__(
        self,
        services: list[ManagedService],
        *,
        spawn=subprocess.Popen,
        sleep=time.sleep,
        clock=time.monotonic,
        stop_grace_s: float = DEFAULT_STOP_GRACE_S,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
    ) -> None:
        self.services = services
        self.spawn = spawn
        self.sleep = sleep
        self.clock = clock
        self.stop_grace_s = stop_grace_s
        self.max_restarts = max_restarts
        #: name -> process, for the ones this supervisor started itself
        self.started: dict[str, object] = {}
        self._restarts: dict[str, int] = {}
        #: services the watchdog has stopped trying to revive
        self.give_ups: list[str] = []

    def _service(self, name: str) -> ManagedService | None:
        return next((s for s in self.services if s.name == name), None)

    def start_all(self, wait_s: float = DEFAULT_WAIT_S) -> dict[str, str]:
        """Returns name -> one of ``off``, ``adopted``, ``started``,
        ``starting``, ``exited (N)`` or ``failed: …``. Never raises."""
        return {service.name: self._start(service, wait_s) for service in self.services}

    def _start(self, service: ManagedService, wait_s: float) -> str:
        if service.off:
            return "off"
        if self._healthy(service):
            log.info("%s is already running at %s — adopting it", service.name, service.url)
            return "adopted"
        if not service.autostart:
            return "not running (autostart off)"

        try:
            process = self.spawn(
                service.argv,
                cwd=str(service.cwd),
                # stdout/stderr are inherited on purpose: the services prefix
                # their lines ("[whisper] …"), so their startup and their
                # failures land wherever the brain's own output does —
                # a terminal, or the journal.
                stdin=subprocess.DEVNULL,
            )
        except (OSError, ValueError) as exc:
            log.warning("could not start %s: %s", service.name, exc)
            return f"failed: {exc}"

        self.started[service.name] = process
        log.info("started %s: %s", service.name, " ".join(service.argv))

        deadline = self.clock() + max(0.0, wait_s)
        while True:
            code = process.poll()
            if code is not None:
                self.started.pop(service.name, None)
                return f"exited ({code}) — see its output above"
            if self._healthy(service):
                return "started"
            if self.clock() >= deadline:
                # A first run downloads weights, which can take minutes. The
                # brain serves text meanwhile and picks the service up when it
                # answers, so this is not a failure.
                return "starting"
            self.sleep(0.25)

    def restart(self, name: str, wait_s: float = DEFAULT_WAIT_S) -> str:
        """Stop a service we started and start it again. Only ours: a service
        that was adopted belongs to whoever started it."""
        process = self.started.pop(name, None)
        if process is not None:
            self._stop(name, process)
        service = self._service(name)
        if service is None:
            return "unknown service"
        return self._start(service, wait_s)

    async def watch(self, interval_s: float = 5.0, rounds: int | None = None) -> None:
        """Bring back a service we started that has since died.

        A model that falls over mid-conversation should come back by itself;
        otherwise the brain goes on saying "Cannot hear you" until somebody
        reads the log. Bounded by ``max_restarts`` per service, because a
        service that *cannot* start — a bad model name, a GPU that isn't there —
        must not be respawned in a tight loop for the life of the brain.

        ``rounds`` is for tests; left None it runs until cancelled.
        """
        import asyncio

        done = 0
        while rounds is None or done < rounds:
            done += 1
            for service in self.services:
                name = service.name
                process = self.started.get(name)
                if process is None or process.poll() is None:
                    continue  # adopted, never started, or still alive
                code = process.poll()
                self.started.pop(name, None)
                if self._restarts.get(name, 0) >= self.max_restarts:
                    if name not in self.give_ups:
                        self.give_ups.append(name)
                        log.error(
                            "%s died (%s) and has been restarted %d times — leaving it "
                            "down. Start it by hand once you know why.",
                            name, code, self.max_restarts,
                        )
                    continue
                self._restarts[name] = self._restarts.get(name, 0) + 1
                log.warning(
                    "%s died (%s) — restarting (%d/%d)",
                    name, code, self._restarts[name], self.max_restarts,
                )
                self._start(service, wait_s=0.0)
            if rounds is None or done < rounds:
                await asyncio.sleep(interval_s)

    def stop_all(self) -> None:
        for name, process in list(self.started.items()):
            self._stop(name, process)
            self.started.pop(name, None)

    def _stop(self, name: str, process) -> None:
        if process.poll() is not None:
            return
        log.info("stopping %s", name)
        try:
            process.terminate()
            process.wait(timeout=self.stop_grace_s)
        except subprocess.TimeoutExpired:
            log.warning("%s ignored SIGTERM — killing it", name)
            try:
                process.kill()
                process.wait(timeout=self.stop_grace_s)
            except Exception as exc:  # noqa: BLE001 — shutdown must finish
                log.warning("could not kill %s: %s", name, exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not stop %s: %s", name, exc)

    def _healthy(self, service: ManagedService) -> bool:
        try:
            return bool(service.health())
        except Exception:  # noqa: BLE001 — "not answering" is the answer
            return False


# -- what the brain runs ---------------------------------------------------


def _interpreter(configured: str) -> str:
    """The venv Python for a service, or the brain's own.

    Defaulting to the brain's own interpreter is what makes this work with no
    setup at all: ``requirements.txt`` already installs faster-whisper and
    piper-tts, so one venv runs everything. A separate venv is for the case the
    separation was invented for — CUDA wheels that only the transcription
    service should carry.
    """
    if not configured:
        return sys.executable
    return str(Path(os.path.expanduser(configured)))


def _port(url: str, default: int) -> int:
    from urllib.parse import urlparse

    try:
        return urlparse(url).port or default
    except ValueError:
        return default


def whisper_service(config) -> ManagedService:
    """``jarvis-whisper``, told the same model settings as ``[stt]`` — one
    place decides which model JARVIS uses, wherever it runs."""
    from jarvis.audio.whisper_client import build_transcriber

    client = build_transcriber(config.whisper.url, config.whisper.timeout_s)
    argv = [
        _interpreter(config.whisper.python),
        str(_REPO_ROOT / "services" / "whisper" / "serve.py"),
        "--model", config.stt.model,
        "--device", config.stt.device,
        "--compute-type", config.stt.compute_type,
        "--download-root", str(config.whisper_dir),
        "--port", str(_port(config.whisper.url, 3461)),
        "--quiet",   # its per-request lines duplicate the brain's own
    ]
    return ManagedService(
        name="jarvis-whisper",
        url=config.whisper.url,
        argv=argv,
        health=lambda: bool(client.health().get("ok")),
        autostart=config.whisper.autostart,
    )


def voder_service(config, voice: str | None = None) -> ManagedService:
    """``jarvis-voder``, told the persona's voice (falling back to ``[tts]``)."""
    from jarvis.audio.voder_client import build_voder

    client = build_voder(config.voder.url, timeout_s=config.voder.timeout_s)
    argv = [
        _interpreter(config.voder.python),
        str(_REPO_ROOT / "services" / "voder" / "serve.py"),
        "--voice", voice or config.tts.voice,
        "--voice-dir", str(config.piper_dir),
        "--port", str(_port(config.voder.url, 3462)),
        "--quiet",
    ]
    return ManagedService(
        name="jarvis-voder",
        url=config.voder.url,
        argv=argv,
        health=lambda: bool(client.health().get("ok")),
        autostart=config.voder.autostart,
    )
