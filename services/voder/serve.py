#!/usr/bin/env python3
"""jarvis-voder — the speech synthesis service (M3 decision 8).

Piper, loaded once and kept warm, behind the smallest HTTP surface that does the
job. The brain posts a sentence and gets PCM back; the edge plays it.

    POST /speak    {"text", "voice"?, "sample_rate"?}
                   -> raw 16-bit LE mono PCM, with
                      X-Voder-Ms (synthesis time) and X-Voder-Sample-Rate
    GET  /voices   -> {"voices": [...], "default": "..."}
    GET  /healthz  -> {"ok", "voice", "sample_rate", "warm", "utterances"}

Same reasoning as the whisper service: a model that takes a second to load
should stay warm across brain restarts, a synthesis that wedges should be
survivable by killing one process, and the brain must still start when this is
down (it degrades to sending ``text`` only). It runs on the CPU, which keeps it
off Whisper's GPU.

Binds 127.0.0.1 and has no auth: a port only this machine can open needs no
password.

**This file deliberately does not import ``jarvis``** — it runs in its own venv
with piper-tts and nothing else. Point ``--voice-dir`` at the brain's
``data/models/piper`` (or a copy), and keep ``--voice`` in step with the
persona's voice.

    python3 -m venv ~/.local/share/jarvis/voder-venv
    ~/.local/share/jarvis/voder-venv/bin/pip install --upgrade pip piper-tts numpy
    ~/.local/share/jarvis/voder-venv/bin/python services/voder/serve.py \\
        --voice-dir data/models/piper --port 3462
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

#: A sentence, not an essay — the brain splits an answer into sentences before
#: it gets here (jarvis/core/speech.py), so anything this long is a bug
#: upstream, and synthesising it would block the queue behind it.
MAX_TEXT_CHARS = 2000
MAX_BODY_BYTES = 64 * 1024


def resample(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear resampling. The same maths as the edge's, and for the same reason:
    the difference from a windowed-sinc is inaudible on speech through a small
    speaker, and this has no dependencies."""
    if from_rate == to_rate or len(pcm) == 0:
        return pcm
    ratio = from_rate / to_rate
    length = int(len(pcm) / ratio)
    at = np.arange(length, dtype=np.float64) * ratio
    i0 = np.floor(at).astype(np.int64)
    i1 = np.minimum(len(pcm) - 1, i0 + 1)
    t = at - i0
    src = pcm.astype(np.float64)
    return (src[i0] * (1 - t) + src[i1] * t).astype(np.int16)


class Voder:
    """The voices, and the one lock that serialises synthesis.

    Piper is fast but not thread-safe, and the server is threaded so a health
    check never waits behind a sentence.
    """

    def __init__(self, voice_dir, default_voice):
        self.voice_dir = Path(voice_dir)
        self.default_voice = default_voice
        self.lock = threading.Lock()
        self.count = 0
        self.warm = False
        self._voices: dict[str, object] = {}
        self._rates: dict[str, int] = {}

    def available(self) -> list[str]:
        return sorted(p.stem for p in self.voice_dir.glob("*.onnx"))

    def load(self, name: str):
        """Load and cache one voice. Raises FileNotFoundError if it isn't there
        — a missing voice is a setup problem with a name, not a silent silence."""
        if name in self._voices:
            return self._voices[name]
        path = self.voice_dir / f"{name}.onnx"
        if not path.is_file() or not path.with_suffix(".onnx.json").is_file():
            raise FileNotFoundError(
                f"voice {name!r} is not in {self.voice_dir} "
                f"(have: {', '.join(self.available()) or 'none'})"
            )
        from piper import PiperVoice

        voice = PiperVoice.load(str(path), config_path=str(path.with_suffix(".onnx.json")))
        self._voices[name] = voice
        self._rates[name] = int(getattr(voice.config, "sample_rate", 22050))
        return voice

    def sample_rate(self, name: str) -> int:
        self.load(name)
        return self._rates[name]

    def warmup(self):
        """One synthesis at startup, so the first real sentence doesn't pay for
        the ONNX session's first run."""
        try:
            self.speak("Ready.", self.default_voice, 0)
            self.warm = True
        except Exception as exc:  # noqa: BLE001 — reported, never fatal
            sys.stderr.write(f"[voder] warmup failed: {exc!r}\n")

    def speak(self, text: str, voice_name: str, sample_rate: int):
        """Returns (pcm int16, rate, ms). ``sample_rate`` 0 means the voice's
        own rate, which is the one that sounds best."""
        started = time.time()
        with self.lock:
            voice = self.load(voice_name)
            chunks = [c.audio_int16_array for c in voice.synthesize(text)]
        native = self._rates[voice_name]
        pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        rate = native
        if sample_rate and sample_rate != native:
            pcm = resample(pcm, native, sample_rate)
            rate = sample_rate
        self.count += 1
        return pcm, rate, int((time.time() - started) * 1000)


def make_handler(v: Voder, quiet=False):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write("[voder] %s\n" % (fmt % args))

        def _json(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/healthz":
                try:
                    rate = v.sample_rate(v.default_voice)
                    ok, detail = True, None
                except Exception as exc:  # noqa: BLE001
                    rate, ok, detail = 0, False, str(exc)
                return self._json(200, {
                    "ok": ok,
                    "voice": v.default_voice,
                    "sample_rate": rate,
                    "warm": v.warm,
                    "utterances": v.count,
                    "error": detail,
                })
            if path == "/voices":
                return self._json(200, {"voices": v.available(), "default": v.default_voice})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0] != "/speak":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "bad Content-Length"})
            if length <= 0:
                return self._json(400, {"error": "empty body"})
            if length > MAX_BODY_BYTES:
                return self._json(413, {"error": "body too large"})
            try:
                request = json.loads(self.rfile.read(length))
            except ValueError:
                return self._json(400, {"error": "body is not JSON"})
            if not isinstance(request, dict):
                return self._json(400, {"error": "body must be an object"})

            text = request.get("text")
            if not isinstance(text, str) or not text.strip():
                return self._json(400, {"error": "speak needs text"})
            if len(text) > MAX_TEXT_CHARS:
                return self._json(413, {"error": f"text is {len(text)} chars, limit is {MAX_TEXT_CHARS}"})
            voice_name = request.get("voice") or v.default_voice
            if not isinstance(voice_name, str) or "/" in voice_name or "\\" in voice_name:
                # it becomes a filename under the voice directory
                return self._json(400, {"error": "voice must be a voice name"})
            rate_in = request.get("sample_rate", 0)
            if not isinstance(rate_in, int) or isinstance(rate_in, bool) or not (0 <= rate_in <= 48000):
                return self._json(400, {"error": "sample_rate must be 0..48000"})

            try:
                pcm, rate, ms = v.speak(text.strip(), voice_name, rate_in)
            except FileNotFoundError as exc:
                return self._json(404, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 — reported, not swallowed
                sys.stderr.write(f"[voder] synthesis failed: {exc!r}\n")
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

            raw = pcm.tobytes()
            if not quiet:
                sys.stderr.write(f"[voder] {ms}ms {len(pcm) / max(1, rate):.1f}s {text[:60]!r}\n")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("X-Voder-Ms", str(ms))
            self.send_header("X-Voder-Sample-Rate", str(rate))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def serve(v, host="127.0.0.1", port=3462, quiet=False, port_file=None):
    httpd = ThreadingHTTPServer((host, port), make_handler(v, quiet))
    httpd.daemon_threads = True
    bound = httpd.server_address[1]
    if port_file:
        with open(port_file, "w") as fh:
            fh.write(str(bound))
    print(f"jarvis-voder listening on http://{host}:{bound}", flush=True)
    return httpd


def main(argv=None):
    ap = argparse.ArgumentParser(description="JARVIS speech synthesis service (M3)")
    ap.add_argument("--voice", default=os.environ.get("JARVIS_VODER_VOICE", "en_GB-alan-medium"),
                    help="Piper voice name; keep in step with the persona's voice")
    ap.add_argument("--voice-dir", default=os.environ.get("JARVIS_PIPER_DIR", "data/models/piper"))
    ap.add_argument("--host", default="127.0.0.1", help="loopback only; there is no auth")
    ap.add_argument("--port", type=int, default=3462, help="0 asks the OS for a free one")
    ap.add_argument("--port-file", default=None)
    ap.add_argument("--warm-only", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    v = Voder(args.voice_dir, args.voice)
    started = time.time()
    v.warmup()
    sys.stderr.write(
        f"[voder] voice {args.voice} from {args.voice_dir} warm in "
        f"{int((time.time() - started) * 1000)} ms\n"
    )
    if args.warm_only:
        return 0 if v.warm else 1

    httpd = serve(v, args.host, args.port, args.quiet, args.port_file)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
