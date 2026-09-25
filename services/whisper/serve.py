#!/usr/bin/env python3
"""jarvis-whisper — the transcription service (M3 decision 8).

Whisper on the GPU, loaded once and kept warm, behind the smallest HTTP surface
that does the job. The brain talks to it over loopback
(``jarvis/audio/whisper_client.py``); nothing else may reach it, which is why it
binds 127.0.0.1 and has no auth: a port only this machine can open needs no
password, and a password in a unit file is one more thing to leak.

    POST /transcribe   body = 16 kHz signed-16-bit-LE mono PCM
                       ->  {"text", "language", "languageProbability",
                            "confidence", "ms", "dropped"}
    GET  /healthz      ->  {"ok", "model", "device", "warm", "transcriptions"}

Why a separate process at all: a GPU model that takes seconds to load should
stay warm across brain restarts, a transcription that wedges should be
survivable by killing one process, and the brain must still start — text mode,
typed input — when this is down.

**This file deliberately does not import ``jarvis``.** It runs in its own venv,
next to faster-whisper and the CUDA libraries, on a machine that may have none
of JARVIS's other dependencies. Keep ``--model`` in step with the brain's
``[stt] model``.

Setting it up (Ubuntu/Fedora, a CUDA box; drop the nvidia wheels for CPU):

    python3 -m venv ~/.local/share/jarvis/whisper-venv
    ~/.local/share/jarvis/whisper-venv/bin/pip install --upgrade pip faster-whisper
    # CTranslate2 needs CUDA's own libraries and does not depend on them
    # itself; without these the model loads and then fails at the first
    # inference with "Library libcublas.so.12 is not found". See
    # _preload_cuda_libraries below.
    ~/.local/share/jarvis/whisper-venv/bin/pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

    # once, to fetch the weights into the model directory
    ~/.local/share/jarvis/whisper-venv/bin/python services/whisper/serve.py --warm-only

    # normal run: the port [whisper] url defaults to
    ~/.local/share/jarvis/whisper-venv/bin/python services/whisper/serve.py --port 3461

``--port 0`` asks the OS for a free port and prints the one it got, which is
what the test suite uses — a test that guessed a port would be testing whichever
process happened to own it.

A port of Mike's ``services/whisper/serve.py``.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2

#: The brain's own limit is 1 MB (``jarvis/remote/intake.py``); this is the same
#: wall on this side of the loopback, because "the caller already checked" is
#: how a service ends up with no limit at all.
MAX_BODY_BYTES = 4 * 1024 * 1024

# The second line of defence against Whisper's answer to silence.
#
# Measured by Mike on large-v3-turbo: three seconds of digital silence
# transcribes as "Thank you." with no_speech_prob 0.00 and avg_logprob -0.22,
# and low-level noise as "You". The per-segment statistics do NOT catch that —
# the model is confident about the words it invented — so the guard that
# actually works is Silero VAD in front of the encoder (vad_filter below). This
# pair stays as a floor for the segments VAD passes and the model is still
# guessing at; it is deliberately loose enough never to drop quiet speech.
NO_SPEECH_MAX = 0.6
AVG_LOGPROB_MIN = -1.0

#: A short vocabulary hint, handed to the decoder as the text that preceded this
#: utterance. The mode commands are two-word fragments with no sentence around
#: them, which is the case Whisper is worst at — and if they are not transcribed
#: exactly, the matcher in ``jarvis/remote/addressing.py`` cannot help however
#: good it is. Kept to the vocabulary, with no instructions in it: a longer
#: prompt starts appearing in the output of unrelated speech.
DEFAULT_HINT = (
    "Jarvis. Hey Jarvis. Pause input. Continue input. "
    "Change input to always, by name, or push to talk. Turn off the mic."
)


def _preload_cuda_libraries():
    """Make the venv's own cuBLAS and cuDNN findable by CTranslate2.

    CTranslate2 dlopen()s libcublas.so.12 and libcudnn_ops.so.9 by bare name, so
    without a system CUDA install it fails at the first inference with "Library
    libcublas.so.12 is not found" — *after* the model has loaded, which makes it
    look like a model problem rather than a missing dependency. pip puts those
    libraries inside nvidia/*/lib in site-packages, and loading them RTLD_GLOBAL
    here is what puts them in the process's own namespace.

    Setting LD_LIBRARY_PATH in the unit file would work too, and would have to
    be got right in every place the service is ever started. This works wherever
    it is run from, including a test harness.
    """
    roots = [os.path.join(p, "nvidia") for p in sys.path if p.endswith("site-packages")]
    loaded = []
    for root in roots:
        # cuBLAS before cuDNN: cuDNN's own libraries link against it, and the
        # loader will not go looking for a dependency it has not been told about.
        for pattern in (
            "cublas/lib/libcublas*.so.*",
            "cublas/lib/libcublasLt*.so.*",
            "cuda_nvrtc/lib/libnvrtc*.so.*",
            "cudnn/lib/libcudnn*.so.*",
        ):
            for path in sorted(glob.glob(os.path.join(root, pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(path))
                except OSError:
                    pass  # a variant this card does not need
    return loaded


class Transcriber:
    """The model, and the one lock that serialises it.

    CTranslate2 models are not safe to call concurrently from two threads, and
    the server is threaded so a health check never waits behind a transcription.
    One utterance at a time is also the honest shape of the workload: there is
    one person talking.
    """

    def __init__(self, model_name, device, compute_type, download_root=None, hint=DEFAULT_HINT):
        self.preloaded = _preload_cuda_libraries() if device == "cuda" else []
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.device = device
        self.hint = hint or None
        self.lock = threading.Lock()
        self.count = 0
        self.warm = False
        started = time.time()
        self.model = WhisperModel(
            model_name, device=device, compute_type=compute_type, download_root=download_root
        )
        self.load_ms = int((time.time() - started) * 1000)

    def warmup(self):
        """One real inference at startup, and it has to be a real one.

        The first call after a load pays for CUDA kernel compilation and the
        tokeniser, and it is five to ten times the cost of the second. The
        latency budget allows 800 ms for transcription; paying that debt here
        means the user never does.

        It also has to *reach the encoder*, which is the part that touches CUDA
        — otherwise a GPU install that cannot load libcublas reports itself warm
        and healthy and then fails on the user's first sentence, which is
        exactly what happened. Quiet noise does not reach it: Silero drops it as
        non-speech and the model is never run. So: a tone the detector keeps,
        and `vad=False` so nothing can drop it on the way.
        """
        t = np.arange(SAMPLE_RATE // 2, dtype=np.float32) / SAMPLE_RATE
        tone = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        self.run(tone, vad=False)
        self.warm = True

    def run(self, audio, vad=True):
        """float32 mono at 16 kHz in, one dict out."""
        started = time.time()
        with self.lock:
            segments, info = self.model.transcribe(
                audio,
                # beam 1: on a five-second utterance beam search buys a fraction
                # of a percent of accuracy for most of the latency budget.
                beam_size=1,
                language=None,
                task="transcribe",
                # On, and it is the one thing standing between silence and an
                # invented sentence in the conversation (see NO_SPEECH_MAX). It
                # does not re-segment what the edge already segmented: the
                # pieces are joined below, and Silero only removes non-speech,
                # so a sentence spoken with ordinary pauses comes back whole.
                vad_filter=vad,
                initial_prompt=self.hint,
                condition_on_previous_text=False,
            )
            kept, dropped = [], None
            for s in segments:
                if s.no_speech_prob > NO_SPEECH_MAX and s.avg_logprob < AVG_LOGPROB_MIN:
                    dropped = f"no_speech={s.no_speech_prob:.2f} logprob={s.avg_logprob:.2f}"
                    continue
                kept.append(s)

        text = " ".join(s.text.strip() for s in kept).strip()
        # Mean per-token log probability, exponentiated: the model's own
        # confidence in the words it chose, in the 0..1 the protocol carries.
        confidence = None
        if kept:
            weights = [max(1, int((s.end - s.start) * 10)) for s in kept]
            avg = sum(s.avg_logprob * w for s, w in zip(kept, weights)) / sum(weights)
            confidence = round(float(np.exp(avg)), 4)

        self.count += 1
        return {
            "text": text,
            "language": info.language,
            "languageProbability": round(float(info.language_probability), 4),
            "confidence": confidence,
            "ms": int((time.time() - started) * 1000),
            "dropped": dropped,
        }


#: The failure this service exists to make survivable, and the one it used to
#: hide: CTranslate2 dlopen()s CUDA's libraries by bare name, so a GPU box
#: without them loads the model happily and dies at the first inference.
_CUDA_LIBRARY_ERRORS = ("libcublas", "libcudnn", "libnvrtc", "cublasLt")


def cuda_hint(exc: Exception) -> str | None:
    """The fix for a missing-CUDA-library error, or None if that is not what
    this is."""
    message = str(exc)
    if not any(name in message for name in _CUDA_LIBRARY_ERRORS):
        return None
    return (
        "CTranslate2 needs CUDA's own libraries and does not depend on them "
        "itself. Install them into the venv that runs THIS service:\n"
        "    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n"
        "(or run with --device cpu --compute-type int8)"
    )


def _explain(exc: Exception, device: str) -> None:
    """Say what to do about a startup failure, on stderr, without a traceback."""
    hint = cuda_hint(exc)
    if hint:
        sys.stderr.write(f"[whisper] {hint}\n")
    elif device == "cuda":
        sys.stderr.write(
            "[whisper] --device cuda was asked for. If this machine has no working "
            "CUDA, run with --device cpu --compute-type int8 (and set [stt] device "
            "in the brain's config.toml to match).\n"
        )


def make_handler(t, quiet=False):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write("[whisper] %s\n" % (fmt % args))

        def _json(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.split("?")[0] != "/healthz":
                return self._json(404, {"error": "not found"})
            self._json(200, {
                "ok": True,
                "model": t.model_name,
                "device": t.device,
                "warm": t.warm,
                "loadMs": t.load_ms,
                "transcriptions": t.count,
                "hint": bool(t.hint),
                "sampleRate": SAMPLE_RATE,
            })

        def do_POST(self):
            if self.path.split("?")[0] != "/transcribe":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "bad Content-Length"})
            # Refused before reading: a body this big is not an utterance, and
            # reading it to find that out is the denial of service itself.
            if length <= 0:
                return self._json(400, {"error": "empty body"})
            if length > MAX_BODY_BYTES:
                return self._json(413, {"error": f"body is {length} bytes, limit is {MAX_BODY_BYTES}"})

            body = self.rfile.read(length)
            if len(body) != length:
                return self._json(400, {"error": "short body"})
            if len(body) % BYTES_PER_SAMPLE:
                return self._json(400, {"error": "not whole 16-bit samples"})

            # s16le to the float32 in -1..1 that Whisper wants. One copy, and no
            # ffmpeg in the path: the edge already produced the format.
            audio = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
            try:
                result = t.run(audio)
            except Exception as exc:  # noqa: BLE001 — reported, not swallowed
                sys.stderr.write(f"[whisper] transcribe failed: {exc!r}\n")
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

            if not quiet:
                sys.stderr.write(
                    f"[whisper] {len(audio) / SAMPLE_RATE:.1f}s -> {result['ms']}ms "
                    f"{result['language']} {result['text'][:60]!r}\n"
                )
            self._json(200, result)

    return Handler


def serve(t, host="127.0.0.1", port=3461, quiet=False, port_file=None):
    httpd = ThreadingHTTPServer((host, port), make_handler(t, quiet))
    httpd.daemon_threads = True
    bound = httpd.server_address[1]
    if port_file:
        with open(port_file, "w") as fh:
            fh.write(str(bound))
    # The line the operator and the test harness both read. On stdout and
    # flushed, because a buffered "I am ready" is the same as not being ready.
    print(f"jarvis-whisper listening on http://{host}:{bound}", flush=True)
    return httpd


def main(argv=None):
    ap = argparse.ArgumentParser(description="JARVIS transcription service (M3)")
    ap.add_argument("--model", default=os.environ.get("JARVIS_WHISPER_MODEL", "small.en"),
                    help="faster-whisper model name; keep in step with the brain's [stt] model")
    ap.add_argument("--device", default=os.environ.get("JARVIS_WHISPER_DEVICE", "cuda"))
    ap.add_argument("--compute-type", default=os.environ.get("JARVIS_WHISPER_COMPUTE", "float16"))
    ap.add_argument("--download-root", default=os.environ.get("JARVIS_WHISPER_DIR") or None,
                    help="where the weights live (the brain's data/models/whisper)")
    ap.add_argument("--host", default="127.0.0.1", help="loopback only; there is no auth")
    ap.add_argument("--port", type=int, default=3461, help="0 asks the OS for a free one")
    ap.add_argument("--port-file", default=None, help="write the bound port here")
    ap.add_argument("--hint", default=os.environ.get("JARVIS_WHISPER_HINT", DEFAULT_HINT),
                    help="vocabulary hint given to the decoder; empty string turns it off")
    ap.add_argument("--warm-only", action="store_true", help="load the model, warm it, and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        t = Transcriber(args.model, args.device, args.compute_type, args.download_root, args.hint)
    except Exception as exc:  # noqa: BLE001 — a setup problem, not a crash
        sys.stderr.write(f"[whisper] could not load {args.model} on {args.device}: {exc}\n")
        _explain(exc, args.device)
        return 1
    sys.stderr.write(
        f"[whisper] loaded {args.model} on {args.device} ({args.compute_type}) in {t.load_ms} ms\n"
    )
    warm_started = time.time()
    try:
        t.warmup()
    except Exception as exc:  # noqa: BLE001 — refuse to serve rather than lie
        sys.stderr.write(f"[whisper] FAILED the warmup inference: {exc}\n")
        _explain(exc, args.device)
        sys.stderr.write(
            "[whisper] not starting: a service that cannot transcribe is worse "
            "than one that is down, because the brain would report itself ready.\n"
        )
        return 1
    sys.stderr.write(f"[whisper] warm in {int((time.time() - warm_started) * 1000)} ms\n")
    if args.warm_only:
        return 0

    httpd = serve(t, args.host, args.port, args.quiet, args.port_file)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
