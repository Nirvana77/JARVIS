"""The two warm services (M3 decision 8) and the brain's clients for them.

The HTTP surface is tested against the real ``ThreadingHTTPServer`` on port 0
with a *stubbed model*: loading faster-whisper or Piper would make the suite
take a minute and need a GPU, and what is being checked here is the contract —
bounds, status codes, headers, and what the client does when the service is
down.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from jarvis.audio.voder_client import SynthesisUnavailable, VoderClient, build_voder
from jarvis.audio.whisper_client import (
    NullTranscriber,
    TranscriptionUnavailable,
    WhisperClient,
    build_transcriber,
)

SERVICES = Path(__file__).resolve().parent.parent / "services"


def _load(name: str, path: Path):
    """Import a service script by path — `services/` is not a package, on
    purpose: these run in their own venvs and never import `jarvis`."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def whisper_service():
    return _load("jarvis_whisper_serve", SERVICES / "whisper" / "serve.py")


@pytest.fixture(scope="module")
def voder_service():
    return _load("jarvis_voder_serve", SERVICES / "voder" / "serve.py")


class _Serving:
    """A service on 127.0.0.1:0, in a thread that is joined on the way out —
    conftest's leak check is watching."""

    def __init__(self, httpd):
        self.httpd = httpd
        self.url = f"http://127.0.0.1:{httpd.server_address[1]}"
        self.thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


# -- the whisper service ---------------------------------------------------


class FakeModel:
    """Stands in for the loaded faster-whisper model: it never sees audio, it
    only has to have the shape ``Transcriber.run`` expects."""

    model_name = "fake.en"
    device = "cpu"
    load_ms = 1

    def __init__(self, text="search black holes"):
        self.text = text
        self.count = 0
        self.warm = True
        self.hint = "Jarvis."

    def run(self, audio):
        self.count += 1
        return {
            "text": self.text,
            "language": "en",
            "languageProbability": 0.99,
            "confidence": 0.88,
            "ms": 12,
            "dropped": None,
            "samples": len(audio),
        }


@pytest.fixture
def whisper(whisper_service):
    model = FakeModel()
    service = _Serving(whisper_service.serve(model, port=0, quiet=True))
    yield service, model
    service.close()


def test_healthz_says_what_is_loaded_and_whether_it_is_warm(whisper):
    import httpx

    service, _ = whisper
    body = httpx.get(f"{service.url}/healthz", timeout=5).json()
    assert body["ok"] is True
    assert body["model"] == "fake.en"
    assert body["warm"] is True
    assert body["sampleRate"] == 16000


def test_transcribe_takes_raw_pcm_and_answers_json(whisper):
    service, model = whisper
    client = WhisperClient(service.url)
    out = client.transcribe_full(np.zeros(16000, dtype=np.int16))
    assert out.text == "search black holes"
    assert out.language == "en"
    assert out.confidence == 0.88
    assert model.count == 1
    # the orchestrator's `heard` line wants a logprob, not a probability
    assert client.last_avg_logprob == pytest.approx(float(np.log(0.88)))


def test_an_odd_length_body_is_refused(whisper):
    import httpx

    service, model = whisper
    response = httpx.post(
        f"{service.url}/transcribe", content=b"\x01" * 4001, timeout=5
    )
    assert response.status_code == 400
    assert "16-bit" in response.json()["error"]
    assert model.count == 0


def test_an_empty_or_oversized_body_is_refused_before_it_is_read(whisper, whisper_service):
    import httpx

    import http.client

    service, model = whisper
    assert httpx.post(f"{service.url}/transcribe", content=b"", timeout=5).status_code == 400

    # A declared length with no body behind it: the point is that the service
    # answers without reading a byte of it, which is the whole defence — so it
    # has to be sent past httpx, which would refuse to under-deliver.
    host, port = service.url.rsplit(":", 1)
    conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=5)
    conn.putrequest("POST", "/transcribe")
    conn.putheader("Content-Length", str(whisper_service.MAX_BODY_BYTES + 1))
    conn.endheaders()
    assert conn.getresponse().status == 413
    conn.close()
    assert model.count == 0


def test_an_unknown_path_is_a_404_not_a_stack_trace(whisper):
    import httpx

    service, _ = whisper
    assert httpx.get(f"{service.url}/", timeout=5).status_code == 404
    assert httpx.post(f"{service.url}/anything", content=b"ab", timeout=5).status_code == 404


def test_a_model_that_blows_up_is_a_500_and_the_service_keeps_serving(whisper):
    import httpx

    service, model = whisper

    def boom(audio):
        raise RuntimeError("cuda fell over")

    model.run = boom
    response = httpx.post(f"{service.url}/transcribe", content=b"\x00" * 3200, timeout=5)
    assert response.status_code == 500
    assert "cuda fell over" in response.json()["error"]
    # still up
    assert httpx.get(f"{service.url}/healthz", timeout=5).json()["ok"] is True


def test_the_vocabulary_hint_names_jarvis_and_the_mode_commands(whisper_service):
    """Whisper is worst at two-word fragments with no sentence around them, and
    a mode command that is not transcribed exactly cannot be matched however
    good the matcher is."""
    hint = whisper_service.DEFAULT_HINT.lower()
    assert "jarvis" in hint
    assert "pause input" in hint and "continue input" in hint
    assert "push to talk" in hint


# -- the whisper client ----------------------------------------------------


def test_a_service_that_is_not_running_is_a_sentence_not_a_stack_trace():
    client = WhisperClient("http://127.0.0.1:1")  # nothing listens on port 1
    with pytest.raises(TranscriptionUnavailable) as exc:
        client.transcribe(np.zeros(1600, dtype=np.int16))
    assert "not running" in str(exc.value)
    with pytest.raises(TranscriptionUnavailable):
        client.health()


def test_whisper_off_gives_one_code_path_not_a_none():
    stt = build_transcriber("off")
    assert isinstance(stt, NullTranscriber)
    with pytest.raises(TranscriptionUnavailable):
        stt.transcribe(np.zeros(1600, dtype=np.int16))
    assert stt.health()["ok"] is False
    assert isinstance(build_transcriber("http://127.0.0.1:3461"), WhisperClient)


# -- the voder service -----------------------------------------------------


class FakeVoder:
    """The voder with Piper taken out: same interface, a tone instead of a voice."""

    default_voice = "en_GB-alan-medium"
    warm = True

    def __init__(self):
        self.count = 0
        self.said: list[str] = []

    def available(self):
        return [self.default_voice, "en_US-other-medium"]

    def sample_rate(self, name):
        if name not in self.available():
            raise FileNotFoundError(f"voice {name!r} is not there")
        return 22050

    def speak(self, text, voice_name, sample_rate):
        if voice_name not in self.available():
            raise FileNotFoundError(f"voice {voice_name!r} is not there")
        self.count += 1
        self.said.append(text)
        rate = sample_rate or 22050
        return np.zeros(rate // 2, dtype=np.int16), rate, 7


@pytest.fixture
def voder(voder_service):
    fake = FakeVoder()
    service = _Serving(voder_service.serve(fake, port=0, quiet=True))
    yield service, fake
    service.close()


def test_speak_returns_pcm_with_the_synthesis_time_in_a_header(voder):
    import httpx

    service, fake = voder
    response = httpx.post(
        f"{service.url}/speak", json={"text": "Of course, sir.", "sample_rate": 16000}, timeout=5
    )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert response.headers["X-Voder-Ms"] == "7"
    assert response.headers["X-Voder-Sample-Rate"] == "16000"
    assert len(response.content) == 8000 * 2
    assert fake.said == ["Of course, sir."]


def test_voices_and_healthz(voder):
    import httpx

    service, _ = voder
    voices = httpx.get(f"{service.url}/voices", timeout=5).json()
    assert voices["default"] == "en_GB-alan-medium"
    assert "en_US-other-medium" in voices["voices"]
    health = httpx.get(f"{service.url}/healthz", timeout=5).json()
    assert health["ok"] is True and health["sample_rate"] == 22050


@pytest.mark.parametrize(
    "body,status",
    [
        ({"text": ""}, 400),
        ({}, 400),
        ({"text": "x" * 3000}, 413),
        ({"text": "hello", "voice": "../../etc/passwd"}, 400),
        ({"text": "hello", "voice": "nonexistent"}, 404),
        ({"text": "hello", "sample_rate": 999999}, 400),
    ],
)
def test_the_voder_refuses_what_it_should(voder, body, status):
    import httpx

    service, _ = voder
    assert httpx.post(f"{service.url}/speak", json=body, timeout=5).status_code == status


def test_a_body_that_is_not_json_is_a_400(voder):
    import httpx

    service, _ = voder
    response = httpx.post(f"{service.url}/speak", content=b"not json", timeout=5)
    assert response.status_code == 400


def test_the_resampler_keeps_the_length_ratio(voder_service):
    tone = (np.sin(np.arange(22050) / 10) * 10000).astype(np.int16)
    out = voder_service.resample(tone, 22050, 16000)
    assert out.dtype == np.int16
    assert len(out) == pytest.approx(16000, rel=0.01)
    assert voder_service.resample(tone, 16000, 16000) is tone


# -- the voder client ------------------------------------------------------


def test_the_voder_client_speaks_a_sentence(voder):
    service, fake = voder
    client = VoderClient(service.url, sample_rate=16000)
    pcm, rate = client.speak("Of course, sir.")
    assert rate == 16000
    assert pcm.dtype == np.int16 and len(pcm) == 8000
    assert client.available is True


def test_a_voder_that_is_down_degrades_to_text_only():
    client = VoderClient("http://127.0.0.1:1")
    with pytest.raises(SynthesisUnavailable):
        client.speak("anything")
    assert build_voder("off").speak_or_none("anything") is None
    assert build_voder("off").available is False
