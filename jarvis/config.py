"""Load ``config.toml`` (+ ``.env`` for secrets) into a frozen ``Config``.

``config.toml`` holds every non-secret knob. ``.env`` holds only secrets and is
loaded so ``os.getenv`` sees them elsewhere; nothing here reads the API key.

Two environment variables still override the file, matching the legacy code:
``JARVIS_PERSONA`` (beats ``[persona].active``) and ``language`` (beats
``[general].language``).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # .env holds only secrets; M1 needs none
    def load_dotenv(*_a, **_k) -> bool:  # type: ignore[misc]
        return False

# Repo root = parent of this package directory. Paths in config.toml are
# resolved against the current working directory first (CLAUDE.md: "always run
# from the repo root"), then against this as a fallback.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_NAME = "config.toml"


@dataclass(frozen=True)
class GeneralConfig:
    language: str = "en"


@dataclass(frozen=True)
class PersonaConfig:
    active: str = "jarvis"


@dataclass(frozen=True)
class WakeConfig:
    model: str = "hey_jarvis"
    threshold: float = 0.5


@dataclass(frozen=True)
class CaptureConfig:
    sample_rate: int = 16000
    frame_ms: int = 80
    window_timeout_s: float = 8.0
    silence_s: float = 1.0
    #: after a command, keep listening this long for a follow-up (no wake word)
    #: before announcing standby
    follow_up_s: float = 10.0

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)


@dataclass(frozen=True)
class STTConfig:
    # ".en" models beat the multilingual ones for English at the same size/speed
    model: str = "small.en"
    compute_type: str = "int8"
    device: str = "cpu"
    language: str = "en"


@dataclass(frozen=True)
class TTSConfig:
    voice: str = "en_GB-alan-medium"


@dataclass(frozen=True)
class NLUConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    threshold: float = 0.35
    #: min cosine similarity to any training phrase; below this -> "unknown"
    similarity_floor: float = 0.30


@dataclass(frozen=True)
class ReasonerConfig:
    enabled: bool = True
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"


@dataclass(frozen=True)
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    nlu: NLUConfig = field(default_factory=NLUConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    #: repo-root-relative directory for models / NLU artifacts / skill scratch
    data_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data")
    #: Hugging Face token (from .env / env) for authenticated model downloads
    hf_token: str | None = None

    # -- derived paths -------------------------------------------------------
    @property
    def nlu_model_dir(self) -> Path:
        return self.data_dir / "models" / "nlu"

    @property
    def whisper_dir(self) -> Path:
        return self.data_dir / "models" / "whisper"

    @property
    def piper_dir(self) -> Path:
        return self.data_dir / "models" / "piper"

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "nlu" / "corpus.sqlite"

    def skill_data_dir(self, name: str) -> Path:
        return self.data_dir / "skills" / name


def _find_config(path: str | os.PathLike[str] | None) -> Path | None:
    if path is not None:
        return Path(path).expanduser().resolve()
    cwd_candidate = Path.cwd() / _DEFAULT_CONFIG_NAME
    if cwd_candidate.is_file():
        return cwd_candidate
    root_candidate = _REPO_ROOT / _DEFAULT_CONFIG_NAME
    if root_candidate.is_file():
        return root_candidate
    return None


def _section(raw: dict, key: str) -> dict:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"config.toml: [{key}] must be a table")
    return value


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Build a :class:`Config`. Missing file -> all defaults. Env overrides win."""
    load_dotenv()  # make secrets visible to os.getenv elsewhere; harmless if absent

    raw: dict = {}
    config_path = _find_config(path)
    if config_path is not None:
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)

    general = _section(raw, "general")
    persona = _section(raw, "persona")
    wake = _section(raw, "wake")
    capture = _section(raw, "capture")
    stt = _section(raw, "stt")
    tts = _section(raw, "tts")
    nlu = _section(raw, "nlu")
    reasoner = _section(raw, "reasoner")
    paths = _section(raw, "paths")

    # Environment overrides (kept from the legacy code).
    language = os.getenv("language") or general.get("language", "en")
    active_persona = os.getenv("JARVIS_PERSONA") or persona.get("active", "jarvis")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    data_dir_raw = paths.get("data_dir", "data")
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        data_dir = (_REPO_ROOT / data_dir).resolve()

    return Config(
        general=GeneralConfig(language=language),
        persona=PersonaConfig(active=active_persona),
        wake=WakeConfig(
            model=wake.get("model", "hey_jarvis"),
            threshold=float(wake.get("threshold", 0.5)),
        ),
        capture=CaptureConfig(
            sample_rate=int(capture.get("sample_rate", 16000)),
            frame_ms=int(capture.get("frame_ms", 80)),
            window_timeout_s=float(capture.get("window_timeout_s", 8.0)),
            silence_s=float(capture.get("silence_s", 1.0)),
            follow_up_s=float(capture.get("follow_up_s", 10.0)),
        ),
        stt=STTConfig(
            model=stt.get("model", "small.en"),
            compute_type=stt.get("compute_type", "int8"),
            device=stt.get("device", "cpu"),
            language=stt.get("language", language),
        ),
        tts=TTSConfig(voice=tts.get("voice", "en_GB-alan-medium")),
        nlu=NLUConfig(
            embedding_model=nlu.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            threshold=float(nlu.get("threshold", 0.35)),
            similarity_floor=float(nlu.get("similarity_floor", 0.30)),
        ),
        reasoner=ReasonerConfig(
            enabled=bool(reasoner.get("enabled", True)),
            base_url=reasoner.get("base_url", "http://localhost:11434"),
            model=reasoner.get("model", "qwen2.5:3b"),
        ),
        data_dir=data_dir,
        hf_token=hf_token,
    )
