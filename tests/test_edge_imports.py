"""The edge installs numpy, sounddevice and websockets — and nothing else.

That is not a style preference: the edge is a Raspberry Pi in a room, and the
moment ``python -m jarvis edge`` pulls in fastembed or faster-whisper it stops
being installable there at all. An import added in the wrong module is the
easiest possible way to break it, and the hardest to notice on a dev box where
everything happens to be installed — so it is checked here, in a subprocess,
against a real interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys

#: Everything that makes the brain the brain. None of it may be reachable from
#: the edge's import graph.
FORBIDDEN = (
    "fastembed",
    "faster_whisper",
    "piper",
    "sklearn",
    "anthropic",
    "openwakeword",
    "onnxruntime",
    "torch",
    "transformers",
    "huggingface_hub",
)


def _imported_by(source: str) -> set[str]:
    # Assembled line by line, not with an f-string in a triple-quoted block: a
    # multi-line `source` interpolated into an indented template is a syntax
    # error, and the test would then "pass" by failing to run anything.
    script = "\n".join(
        [
            "import json, sys",
            source,
            f"print(json.dumps(sorted(m for m in {FORBIDDEN!r} if m in sys.modules)))",
        ]
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def test_the_edge_entry_point_loads_no_models():
    assert _imported_by("from jarvis.remote.edge import run_edge, Edge") == set()


def test_the_cli_itself_loads_no_models():
    """`python -m jarvis edge` has to get as far as the subcommand without
    importing `jarvis.app`, which is the whole brain."""
    assert _imported_by("import jarvis.__main__") == set()


def test_the_edge_can_be_built_and_torn_down_with_nothing_installed():
    """Constructing an `Edge` must not touch a sound card or a model either —
    only when it actually starts listening."""
    imported = _imported_by(
        "from jarvis.config import load_config\n"
        "from jarvis.remote.edge import Edge\n"
        "edge = Edge(load_config())\n"
        "assert edge.segmenter.opts.sample_rate == 16000\n"
    )
    assert imported == set()


def test_the_pieces_the_edge_shares_with_the_brain_stay_light():
    for module in (
        "jarvis.audio.segment",
        "jarvis.audio.player",
        "jarvis.remote.protocol",
        "jarvis.config",
    ):
        assert _imported_by(f"import {module}") == set(), module


def test_an_edge_that_cannot_connect_says_so():
    """It retries for ever by design — but silence is indistinguishable from a
    hang, and the reason (a refused connection, a 401 from Access, a bad
    hostname) is the whole diagnosis."""
    import asyncio
    import contextlib
    import io

    from jarvis.config import load_config
    from jarvis.remote.edge import Edge

    config = load_config()

    class Boom:
        async def __aenter__(self):
            raise ConnectionRefusedError("[Errno 111] Connect call failed")

        async def __aexit__(self, *a):
            return False

    edge = Edge(config, connect=Boom)
    out = io.StringIO()

    async def scenario():
        task = asyncio.ensure_future(edge.run())
        await asyncio.sleep(0.2)
        edge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    with contextlib.redirect_stdout(out):
        asyncio.run(scenario())

    printed = out.getvalue()
    assert "cannot reach" in printed, printed
    assert config.edge.server_url in printed
    assert "Connect call failed" in printed, "the reason, not just the fact"


def test_the_brain_by_contrast_does_load_them():
    """The other half of the check: if `jarvis.app` stopped importing the
    models, this test passing would mean nothing."""
    loaded = _imported_by("import jarvis.app")
    assert loaded, "expected the brain's import graph to pull in the models"
