"""Route a sounddevice stream to a specific PipeWire node.

PortAudio can't name a PipeWire node directly, but the PipeWire ALSA plugin
reads ``PIPEWIRE_NODE`` when a pcm is opened. It's set only while the stream
opens: capture and TTS open their streams in the same process, each possibly
aimed at a different node.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def pipewire_target(node: str, device: int | str | None = None) -> Iterator[int | str | None]:
    """Yield the device to open a stream on: ``"pipewire"`` with the env var
    pointing at ``node``, or ``device`` untouched when ``node`` is empty."""
    if not node:
        yield device
        return
    prev = os.environ.get("PIPEWIRE_NODE")
    os.environ["PIPEWIRE_NODE"] = node
    try:
        yield "pipewire"
    finally:
        if prev is None:
            os.environ.pop("PIPEWIRE_NODE", None)
        else:
            os.environ["PIPEWIRE_NODE"] = prev
