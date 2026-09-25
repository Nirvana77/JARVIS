"""The brain starting its own services (M3).

`python -m jarvis serve` spawns jarvis-whisper and jarvis-voder rather than
making somebody open three terminals — but they stay separate *processes*,
because that is the whole point of them: a wedged model is one process to
restart, and a brain restart does not reload a GPU model.

Two rules everything here turns on:

* **adopt what is already running.** A service under systemd, or left over from
  a brain that was killed, must not be started twice on the same port.
* **stop only what we started.** Shutting the brain down must not take out a
  service it merely adopted.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import replace

import pytest

from jarvis.remote.supervisor import (
    ManagedService,
    ServiceSupervisor,
    voder_service,
    whisper_service,
)
from tests.remote_harness import make_config


class FakeProcess:
    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.returncode = self.returncode if self.returncode is not None else 0
        return self.returncode


class Spawner:
    def __init__(self, fail: Exception | None = None):
        self.calls: list[list[str]] = []
        self.processes: list[FakeProcess] = []
        self.fail = fail

    def __call__(self, argv, **kwargs):
        if self.fail:
            raise self.fail
        self.calls.append(list(argv))
        process = FakeProcess(argv, **kwargs)
        self.processes.append(process)
        return process


def _service(name="jarvis-whisper", *, healthy, argv=("python", "serve.py")):
    """`healthy` is a list of answers, consumed one per health check."""
    answers = list(healthy)

    def health():
        return answers.pop(0) if answers else (healthy[-1] if healthy else False)

    return ManagedService(name=name, url="http://127.0.0.1:3461", argv=list(argv), health=health)


# -- adopting ---------------------------------------------------------------


def test_a_service_that_is_already_answering_is_adopted_not_restarted():
    spawn = Spawner()
    supervisor = ServiceSupervisor([_service(healthy=[True])], spawn=spawn)
    assert supervisor.start_all() == {"jarvis-whisper": "adopted"}
    assert spawn.calls == [], "nothing should have been started"


def test_stopping_does_not_touch_an_adopted_service():
    """Its systemd unit owns it, or another brain does."""
    spawn = Spawner()
    supervisor = ServiceSupervisor([_service(healthy=[True])], spawn=spawn)
    supervisor.start_all()
    supervisor.stop_all()
    assert spawn.processes == []


# -- starting ---------------------------------------------------------------


def test_a_service_that_is_not_answering_is_started_and_waited_for():
    spawn = Spawner()
    # down, down, then up: it took a moment to load its model
    service = _service(healthy=[False, False, True])
    supervisor = ServiceSupervisor([service], spawn=spawn, sleep=lambda _s: None)
    assert supervisor.start_all(wait_s=5) == {"jarvis-whisper": "started"}
    assert spawn.calls == [["python", "serve.py"]]


def test_a_slow_service_is_left_starting_rather_than_failing_the_brain():
    """Weights can take minutes to download on a first run. The brain serves
    text and says "Cannot hear you" until it is up — it must not refuse to
    start over it."""
    spawn = Spawner()
    service = _service(healthy=[False])
    supervisor = ServiceSupervisor([service], spawn=spawn, sleep=lambda _s: None)
    assert supervisor.start_all(wait_s=0.2) == {"jarvis-whisper": "starting"}
    assert len(spawn.calls) == 1


def test_a_service_that_dies_immediately_is_reported_with_its_code():
    spawn = Spawner()
    service = _service(healthy=[False])
    supervisor = ServiceSupervisor([service], spawn=spawn, sleep=lambda _s: None)

    def die_at_once(argv, **kwargs):
        process = FakeProcess(argv, **kwargs)
        process.returncode = 2
        spawn.processes.append(process)
        spawn.calls.append(list(argv))
        return process

    supervisor.spawn = die_at_once
    status = supervisor.start_all(wait_s=5)
    assert "exited" in status["jarvis-whisper"]
    assert "2" in status["jarvis-whisper"]


def test_a_missing_interpreter_is_a_sentence_not_a_traceback():
    spawn = Spawner(fail=FileNotFoundError(2, "No such file or directory"))
    supervisor = ServiceSupervisor([_service(healthy=[False])], spawn=spawn)
    status = supervisor.start_all(wait_s=0.1)
    assert status["jarvis-whisper"].startswith("failed")
    assert "No such file" in status["jarvis-whisper"]


def test_stopping_terminates_what_we_started():
    spawn = Spawner()
    supervisor = ServiceSupervisor(
        [_service(healthy=[False, True])], spawn=spawn, sleep=lambda _s: None
    )
    supervisor.start_all(wait_s=5)
    supervisor.stop_all()
    assert spawn.processes[0].terminated is True


def test_a_service_turned_off_is_never_started():
    spawn = Spawner()
    service = ManagedService(
        name="jarvis-voder", url="off", argv=["python"], health=lambda: False
    )
    supervisor = ServiceSupervisor([service], spawn=spawn)
    assert supervisor.start_all() == {"jarvis-voder": "off"}
    assert spawn.calls == []


def test_autostart_false_reports_the_service_as_left_alone():
    spawn = Spawner()
    service = ManagedService(
        name="jarvis-whisper",
        url="http://127.0.0.1:3461",
        argv=["python"],
        health=lambda: False,
        autostart=False,
    )
    supervisor = ServiceSupervisor([service], spawn=spawn)
    assert supervisor.start_all() == {"jarvis-whisper": "not running (autostart off)"}
    assert spawn.calls == []


# -- restarting, while the brain keeps running -----------------------------


def test_a_service_can_be_restarted_on_demand():
    spawn = Spawner()
    supervisor = ServiceSupervisor(
        [_service(healthy=[False, True, False, True])], spawn=spawn, sleep=lambda _s: None
    )
    supervisor.start_all(wait_s=5)
    first = supervisor.started["jarvis-whisper"]
    assert supervisor.restart("jarvis-whisper", wait_s=5) == "started"
    assert first.terminated is True
    assert len(spawn.calls) == 2
    assert supervisor.started["jarvis-whisper"] is not first


def test_the_watchdog_restarts_a_service_that_died():
    """A model that crashes mid-conversation should come back by itself —
    otherwise the brain keeps saying "Cannot hear you" until somebody notices."""
    spawn = Spawner()
    # health follows reality: something answers only while a process is alive.
    # (A canned "yes" would make the restart adopt itself, which is the correct
    # behaviour for a port something else has taken over — but not this test.)
    service = ManagedService(
        name="jarvis-whisper",
        url="http://127.0.0.1:3461",
        argv=["python", "serve.py"],
        health=lambda: any(p.poll() is None for p in spawn.processes),
    )
    supervisor = ServiceSupervisor(spawn=spawn, services=[service], sleep=lambda _s: None)
    supervisor.start_all(wait_s=5)
    process = supervisor.started["jarvis-whisper"]

    process.returncode = 1  # it fell over
    asyncio_run(supervisor.watch(interval_s=0.01, rounds=1))

    assert len(spawn.calls) == 2, "it was started again"
    assert supervisor.started["jarvis-whisper"] is not process


def test_the_watchdog_leaves_an_adopted_service_alone():
    spawn = Spawner()
    supervisor = ServiceSupervisor([_service(healthy=[True])], spawn=spawn)
    supervisor.start_all()
    asyncio_run(supervisor.watch(interval_s=0.01, rounds=2))
    assert spawn.calls == [], "systemd owns that one, not us"


def test_the_watchdog_gives_up_rather_than_spinning():
    """A service that cannot start — a bad model name, no GPU — must not be
    respawned in a tight loop for the life of the brain."""
    spawn = Spawner()
    service = _service(healthy=[False] * 50)
    supervisor = ServiceSupervisor(
        [service], spawn=spawn, sleep=lambda _s: None, max_restarts=3
    )
    supervisor.start_all(wait_s=0.01)
    for _ in range(10):
        for process in spawn.processes:
            process.returncode = 1
        asyncio_run(supervisor.watch(interval_s=0.01, rounds=1))
    assert len(spawn.calls) == 1 + 3, f"one start plus {3} restarts, got {len(spawn.calls)}"
    assert supervisor.give_ups == ["jarvis-whisper"]


def test_the_watchdog_stops_cleanly_and_leaves_no_threads():
    import asyncio

    spawn = Spawner()
    supervisor = ServiceSupervisor(
        [_service(healthy=[False, True])], spawn=spawn, sleep=lambda _s: None
    )
    supervisor.start_all(wait_s=5)
    before = set(threading.enumerate())

    async def scenario():
        task = asyncio.ensure_future(supervisor.watch(interval_s=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    supervisor.stop_all()
    assert set(threading.enumerate()) == before


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


# -- what gets spawned ------------------------------------------------------


def test_the_whisper_command_carries_the_model_config(tmp_path):
    config = make_config(tmp_path)
    config = replace(
        config,
        stt=replace(config.stt, model="small.en", device="cuda", compute_type="float16"),
    )
    service = whisper_service(config)
    argv = service.argv
    assert argv[0] == sys.executable, "no venv configured -> the brain's own interpreter"
    assert argv[1].endswith("services/whisper/serve.py")
    assert "--model" in argv and argv[argv.index("--model") + 1] == "small.en"
    assert argv[argv.index("--device") + 1] == "cuda"
    assert argv[argv.index("--compute-type") + 1] == "float16"
    assert argv[argv.index("--port") + 1] == "3461"
    # the weights land in the brain's own model directory, not a stray cache
    assert argv[argv.index("--download-root") + 1] == str(config.whisper_dir)


def test_the_voder_command_carries_the_voice(tmp_path):
    config = make_config(tmp_path)
    service = voder_service(config, voice="en_GB-alan-medium")
    argv = service.argv
    assert argv[1].endswith("services/voder/serve.py")
    assert argv[argv.index("--voice") + 1] == "en_GB-alan-medium"
    assert argv[argv.index("--voice-dir") + 1] == str(config.piper_dir)
    assert argv[argv.index("--port") + 1] == "3462"


def test_a_configured_venv_interpreter_is_used(tmp_path):
    interpreter = tmp_path / "whisper-venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("")
    base = make_config(tmp_path)
    config = replace(base, whisper=replace(base.whisper, python=str(interpreter)))
    assert whisper_service(config).argv[0] == str(interpreter)


def test_a_venv_path_with_a_tilde_is_expanded(tmp_path):
    base = make_config(tmp_path)
    config = replace(base, whisper=replace(base.whisper, python="~/nope/bin/python"))
    argv0 = whisper_service(config).argv[0]
    assert "~" not in argv0 and argv0.endswith("/nope/bin/python")


def test_a_service_url_that_is_off_is_not_managed(tmp_path):
    base = make_config(tmp_path)
    config = replace(base, whisper=replace(base.whisper, url="off"))
    assert whisper_service(config).url == "off"


# -- a real subprocess ------------------------------------------------------


def test_it_really_starts_and_really_stops_a_process(tmp_path):
    """The fakes above check the decisions; this checks the process handling —
    that a child is spawned, waited for, and actually gone afterwards."""
    script = tmp_path / "fake_service.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys, time
            (sys.argv[1] and open(sys.argv[1], "w").write("up"))
            while True:
                time.sleep(0.05)
            """
        )
    )
    flag = tmp_path / "ready"
    service = ManagedService(
        name="fake",
        url="http://127.0.0.1:1",
        argv=[sys.executable, str(script), str(flag)],
        health=flag.is_file,
    )
    before = set(threading.enumerate())
    supervisor = ServiceSupervisor([service])
    try:
        assert supervisor.start_all(wait_s=10) == {"fake": "started"}
        process = supervisor.started["fake"]
        assert process.poll() is None, "still running"
    finally:
        supervisor.stop_all()
    assert process.poll() is not None, "the child is gone"
    assert set(threading.enumerate()) == before, "no threads left behind"


def test_stopping_kills_a_child_that_ignores_being_asked(tmp_path):
    script = tmp_path / "stubborn.py"
    script.write_text(
        textwrap.dedent(
            """
            import signal, sys, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            open(sys.argv[1], "w").write("up")
            while True:
                time.sleep(0.05)
            """
        )
    )
    flag = tmp_path / "ready"
    service = ManagedService(
        name="stubborn",
        url="http://127.0.0.1:1",
        argv=[sys.executable, str(script), str(flag)],
        health=flag.is_file,
    )
    supervisor = ServiceSupervisor([service], stop_grace_s=0.5)
    supervisor.start_all(wait_s=10)
    process = supervisor.started["stubborn"]
    supervisor.stop_all()
    # SIGTERM was ignored, so it had to be killed — and shutting the brain down
    # still finished, which is the point: one wedged service cannot hang it.
    assert process.poll() is not None
    assert process.returncode < 0, f"expected a signal, got {process.returncode}"
