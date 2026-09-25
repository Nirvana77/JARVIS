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
from dataclasses import dataclass, field, fields, replace
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
    #: EXTRA (off by default, needs per-mic tuning): voice barge-in — a fresh
    #: sentence spoken while a command is being transcribed abandons that
    #: transcription and becomes the new command. Not part of the core loop.
    barge_in: bool = False
    #: a barge-in needs this much sustained speech before it fires (noise filter)
    barge_in_min_speech_s: float = 0.6
    #: absolute RMS threshold for barge-in speech detection; 0 = auto-calibrate.
    #: set this from `python -m jarvis mic` if auto-calibration misjudges your mic
    barge_in_threshold: float = 0.0
    #: press Enter in the terminal to cancel the current listen / transcription
    allow_interrupt: bool = True
    #: absolute RMS threshold the *main* utterance recorder uses instead of
    #: auto-calibrating; 0 = auto-calibrate. Set this from `python -m jarvis
    #: mic` if commands are getting cut short / transcribed as nonsense near
    #: the end — auto-calibration can mistake your own loud opening words for
    #: the noise floor if you start talking immediately after the wake word.
    vad_threshold: float = 0.0
    #: PortAudio input device (index or name, see `python -m sounddevice`);
    #: None = the system default input
    device: int | str | None = None
    #: PipeWire source node to record from (`pactl list short sources`);
    #: "" = none. Takes precedence over `device`.
    pipewire_node: str = ""

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)


@dataclass(frozen=True)
class STTConfig:
    # ".en" models beat the multilingual ones for English at the same size/speed
    model: str = "medium.en"
    compute_type: str = "int8"
    device: str = "cpu"
    language: str = "en"


@dataclass(frozen=True)
class TTSConfig:
    voice: str = "en_GB-alan-medium"
    #: PipeWire sink node to play through (`pactl list short sinks`); "" = default
    pipewire_node: str = ""


@dataclass(frozen=True)
class NLUConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    threshold: float = 0.35
    #: min cosine similarity to any training phrase; below this -> "unknown"
    similarity_floor: float = 0.30
    #: teach/edit_skill/revert_skill launch a whole multi-turn dialog (and
    #: revert_skill mutates a skill's live code) — a wrong guess there costs
    #: far more than a wrong guess on an ordinary skill, so they need a
    #: higher bar than the general "unknown" cutoff above before JARVIS
    #: commits to one instead of just saying it didn't catch the command.
    meta_action_threshold: float = 0.6


@dataclass(frozen=True)
class ReasonerConfig:
    enabled: bool = True
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"


@dataclass(frozen=True)
class FactoryConfig:
    """M2: the Claude-backed skill factory. The API key/workspace id are
    secrets and stay in ``.env`` — see ``Config.anthropic_api_key`` /
    ``Config.anthropic_workspace_id``."""

    model: str = "claude-opus-5"
    sandbox_timeout_s: float = 10.0
    sandbox_mem_mb: int = 512
    sandbox_cpu_s: int = 5


@dataclass(frozen=True)
class ServerConfig:
    """M3: ``python -m jarvis serve`` — the brain's WebSocket listener.

    The link is assumed to be on the public internet, so TLS is required: either
    ``tls_cert``/``tls_key`` are set, or the brain listens on loopback behind a
    TLS-terminating reverse proxy. A non-loopback bind without TLS is refused
    unless ``allow_insecure`` (LAN/dev only, and it logs a loud warning).
    """

    host: str = "0.0.0.0"
    port: int = 8765
    tls_cert: str = ""
    tls_key: str = ""
    allow_insecure: bool = False
    #: WebSocket frame ceiling. Above `protocol.MAX_AUDIO_BASE64`, so the two
    #: limits cannot disagree about which one refused a segment.
    max_size: int = 4 * 1024 * 1024
    #: keepalive; a peer that stops answering is dropped (mobile/Wi-Fi failure)
    ping_interval_s: float = 20.0
    #: a connection that has not said `hello` by then is closed with 4002
    hello_timeout_s: float = 10.0
    #: Header naming the real client when something terminates TLS in front of
    #: the brain — ``"CF-Connecting-IP"`` for a Cloudflare Tunnel,
    #: ``"X-Forwarded-For"`` for most reverse proxies. Empty = trust nothing.
    #:
    #: It exists because every connection through a tunnel arrives from
    #: 127.0.0.1, and the auth backoff keyed on that cannot tell the Pi in the
    #: hall from somebody hammering the public hostname — so a stranger's failed
    #: guesses would lock out the real edge. The header is believed **only when
    #: the connection itself came from loopback**, i.e. from a proxy on this
    #: machine; from anywhere else it is just somebody's claim about themselves.
    trusted_proxy_header: str = ""
    #: Addresses or CIDRs whose ``trusted_proxy_header`` is also believed, on
    #: top of loopback — for a proxy that is *not* on this machine's loopback:
    #: ``cloudflared`` in Docker (it reaches the brain over the bridge, e.g.
    #: ``172.16.0.0/12``), or a proxy on another host on the LAN.
    #:
    #: Everything inside these ranges can claim to be any client, so keep them
    #: as narrow as the deployment allows — the proxy's own address, not its
    #: whole subnet, wherever that is knowable.
    trusted_proxy_peers: tuple[str, ...] = ()

    def trusted_networks(self) -> list:
        """``trusted_proxy_peers`` parsed. Raises ``ValueError`` naming the bad
        entry: a typo here means the backoff quietly stops telling clients
        apart, which is precisely the failure nobody notices."""
        import ipaddress

        out = []
        for entry in self.trusted_proxy_peers:
            try:
                out.append(ipaddress.ip_network(str(entry), strict=False))
            except ValueError as exc:
                raise ValueError(
                    f"config.toml [server] trusted_proxy_peers: {entry!r} is not "
                    f"an IP address or CIDR ({exc})"
                ) from exc
        return out

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_cert and self.tls_key)


@dataclass(frozen=True)
class WhisperConfig:
    """M3: the brain's client for the warm ``jarvis-whisper`` service.
    ``url = "off"`` gives a ``NullTranscriber`` — one code path, always."""

    url: str = "http://127.0.0.1:3461"
    timeout_s: float = 20.0
    #: `serve` starts the service itself unless something already answers on
    #: `url` — one command instead of three terminals. It still runs as its own
    #: process, and one already running (systemd, or a previous brain) is
    #: adopted rather than started twice.
    autostart: bool = True
    #: the interpreter to start it with; "" = the brain's own, which already
    #: has faster-whisper. A separate venv is for CUDA wheels.
    python: str = ""


@dataclass(frozen=True)
class VoderConfig:
    """M3: the brain's client for ``jarvis-voder``. ``url = "off"`` degrades to
    sending ``text`` only."""

    url: str = "http://127.0.0.1:3462"
    timeout_s: float = 20.0
    sample_rate: int = 16000
    #: as `[whisper] autostart` / `python`
    autostart: bool = True
    python: str = ""


@dataclass(frozen=True)
class AddressingConfig:
    """M3 decisions 5-6: what reaches JARVIS on the remote path, and how long a
    thought is allowed to take. There is no wake word there — the addressing
    modes are the wake word."""

    default_mode: str = "byname"
    names: tuple[str, ...] = ("jarvis", "hey jarvis")
    #: Merge the fragments of one thought; 0 disables the hold window.
    #:
    #: This is paid on *every* turn, so it is the single biggest thing between
    #: the user finishing a sentence and hearing an answer. Mike's value is
    #: 2000; JARVIS uses half that, because the hold does not have to do as
    #: much work here: the edge's ``speaking{on}`` pauses the countdown, so the
    #: window only has to be long enough to *notice a continuation starting*,
    #: not to swallow one whole. What it buys, end to end, is a tolerated pause
    #: of ``hangover + transcription + hold`` — about 2 s on a GPU brain and
    #: 2.7 s on a slow CPU one. Raise it if you think out loud mid-command.
    hold_ms: int = 1000


@dataclass(frozen=True)
class EdgeConfig:
    """M3: ``python -m jarvis edge`` — the audio satellite. It owns a mic, a
    speaker and (optionally) a button, and nothing else: no models, no ONNX."""

    server_url: str = "ws://127.0.0.1:8765"
    device_id: str = "edge"
    tls_ca: str = ""
    reconnect_max_s: int = 30
    #: GPIO pin for the push-to-talk button; 0 = no button
    ptt_gpio: int = 0
    #: PortAudio input/output devices, as in [capture]/[tts]
    input_device: int | str | None = None
    output_device: int | str | None = None
    input_pipewire_node: str = ""
    output_pipewire_node: str = ""
    #: raw `[edge.segment]` overrides of Mike's measured defaults
    segment: dict = field(default_factory=dict)

    def segmenter_options(self):
        """Build the :class:`~jarvis.audio.segment.SegmenterOptions` for this
        room. An unknown knob is an error, not a silent no-op — a typo in a
        tuning value that quietly does nothing is worse than a refused start."""
        from jarvis.audio.segment import DEFAULTS, SegmenterOptions

        known = {f.name for f in fields(SegmenterOptions)}
        unknown = sorted(set(self.segment) - known)
        if unknown:
            raise ValueError(
                f"config.toml [edge.segment]: unknown setting(s) {', '.join(unknown)}; "
                f"known: {', '.join(sorted(known))}"
            )
        return replace(DEFAULTS, **self.segment)


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
    factory: FactoryConfig = field(default_factory=FactoryConfig)
    # M3: the remote-edge split. Unused by the all-in-one path.
    server: ServerConfig = field(default_factory=ServerConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    voder: VoderConfig = field(default_factory=VoderConfig)
    addressing: AddressingConfig = field(default_factory=AddressingConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    #: repo-root-relative directory for models / NLU artifacts / skill scratch
    data_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data")
    #: Hugging Face token (from .env / env) for authenticated model downloads
    hf_token: str | None = None
    #: M3 (.env only): the edge's own token, and the brain's `device_id:token`
    #: table. Never in config.toml, never logged.
    edge_token: str | None = None
    edge_tokens: dict[str, str] = field(default_factory=dict)
    #: Anthropic credentials (.env only; never in config.toml) — used solely
    #: by the M2 skill factory, never on the hot path
    anthropic_api_key: str | None = None
    anthropic_workspace_id: str | None = None

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

    def skill_versions_dir(self, name: str) -> Path:
        """M2: last-3 rollback history for a learned skill (`vN.py` + meta.json)."""
        return self.data_dir / "skills" / "_versions" / name

    @property
    def remote_dir(self) -> Path:
        """M3: per-edge-device state (the addressing mode), one small file each."""
        return self.data_dir / "remote"

    @property
    def skill_quarantine_dir(self) -> Path:
        """M2: self-check failures and skills displaced by `revert_skill` — kept
        for inspection, never imported."""
        return self.data_dir / "skills" / "_quarantine"


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


def _mode_or_default(value) -> str:
    """An addressing mode from the file, or the default. A typo must not leave
    the brain in a mode nothing matches."""
    from jarvis.remote.addressing import DEFAULT_MODE, is_mode  # stdlib-only

    return value if is_mode(value) else DEFAULT_MODE


def _parse_edge_tokens(raw: str | None) -> dict[str, str]:
    """``JARVIS_EDGE_TOKENS="livingroom:s3cret,kitchen:other"`` -> a dict.

    A malformed entry is skipped rather than fatal: one typo in .env should cost
    that one device its connection, not every device theirs.
    """
    out: dict[str, str] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        device_id, _, token = entry.partition(":")
        device_id, token = device_id.strip(), token.strip()
        if device_id and token:
            out[device_id] = token
    return out


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
    factory = _section(raw, "factory")
    paths = _section(raw, "paths")
    # M3
    server = _section(raw, "server")
    whisper = _section(raw, "whisper")
    voder = _section(raw, "voder")
    addressing = _section(raw, "addressing")
    edge = _section(raw, "edge")

    # Environment overrides (kept from the legacy code).
    language = os.getenv("language") or general.get("language", "en")
    active_persona = os.getenv("JARVIS_PERSONA") or persona.get("active", "jarvis")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    # Same fallback anthropic_helper.py uses: ANTHROPIC_API_KEY is the SDK's own
    # env var; `api_key` is what older .env files used.
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("api_key")
    anthropic_workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
    edge_token = os.getenv("JARVIS_EDGE_TOKEN")
    edge_tokens = _parse_edge_tokens(os.getenv("JARVIS_EDGE_TOKENS"))

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
            barge_in=bool(capture.get("barge_in", False)),
            barge_in_min_speech_s=float(capture.get("barge_in_min_speech_s", 0.6)),
            barge_in_threshold=float(capture.get("barge_in_threshold", 0.0)),
            allow_interrupt=bool(capture.get("allow_interrupt", True)),
            vad_threshold=float(capture.get("vad_threshold", 0.0)),
            device=capture.get("device") if capture.get("device") != "" else None,
            pipewire_node=str(capture.get("pipewire_node", "")),
        ),
        stt=STTConfig(
            model=stt.get("model", "medium.en"),
            compute_type=stt.get("compute_type", "int8"),
            device=stt.get("device", "cpu"),
            language=stt.get("language", language),
        ),
        tts=TTSConfig(
            voice=tts.get("voice", "en_GB-alan-medium"),
            pipewire_node=str(tts.get("pipewire_node", "")),
        ),
        nlu=NLUConfig(
            embedding_model=nlu.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            threshold=float(nlu.get("threshold", 0.35)),
            similarity_floor=float(nlu.get("similarity_floor", 0.30)),
            meta_action_threshold=float(nlu.get("meta_action_threshold", 0.6)),
        ),
        reasoner=ReasonerConfig(
            enabled=bool(reasoner.get("enabled", True)),
            base_url=reasoner.get("base_url", "http://localhost:11434"),
            model=reasoner.get("model", "qwen2.5:3b"),
        ),
        factory=FactoryConfig(
            model=os.getenv("ANTHROPIC_MODEL") or factory.get("model", "claude-opus-5"),
            sandbox_timeout_s=float(factory.get("sandbox_timeout_s", 10.0)),
            sandbox_mem_mb=int(factory.get("sandbox_mem_mb", 512)),
            sandbox_cpu_s=int(factory.get("sandbox_cpu_s", 5)),
        ),
        server=ServerConfig(
            host=str(server.get("host", "0.0.0.0")),
            port=int(server.get("port", 8765)),
            tls_cert=str(server.get("tls_cert", "")),
            tls_key=str(server.get("tls_key", "")),
            allow_insecure=bool(server.get("allow_insecure", False)),
            max_size=int(server.get("max_size", 4 * 1024 * 1024)),
            ping_interval_s=float(server.get("ping_interval_s", 20.0)),
            hello_timeout_s=float(server.get("hello_timeout_s", 10.0)),
            trusted_proxy_header=str(server.get("trusted_proxy_header", "")),
            trusted_proxy_peers=tuple(
                str(p) for p in server.get("trusted_proxy_peers", [])
            ),
        ),
        whisper=WhisperConfig(
            url=str(whisper.get("url", "http://127.0.0.1:3461")),
            timeout_s=float(whisper.get("timeout_s", 20.0)),
            autostart=bool(whisper.get("autostart", True)),
            python=str(whisper.get("python", "")),
        ),
        voder=VoderConfig(
            url=str(voder.get("url", "http://127.0.0.1:3462")),
            timeout_s=float(voder.get("timeout_s", 20.0)),
            sample_rate=int(voder.get("sample_rate", 16000)),
            autostart=bool(voder.get("autostart", True)),
            python=str(voder.get("python", "")),
        ),
        addressing=AddressingConfig(
            default_mode=_mode_or_default(addressing.get("default_mode")),
            names=tuple(str(n) for n in addressing.get("names", ["jarvis", "hey jarvis"])),
            hold_ms=int(addressing.get("hold_ms", 1000)),
        ),
        edge=EdgeConfig(
            server_url=str(edge.get("server_url", "ws://127.0.0.1:8765")),
            device_id=str(edge.get("device_id", "edge")),
            tls_ca=str(edge.get("tls_ca", "")),
            reconnect_max_s=int(edge.get("reconnect_max_s", 30)),
            ptt_gpio=int(edge.get("ptt_gpio", 0)),
            input_device=edge.get("input_device") if edge.get("input_device") != "" else None,
            output_device=edge.get("output_device") if edge.get("output_device") != "" else None,
            input_pipewire_node=str(edge.get("input_pipewire_node", "")),
            output_pipewire_node=str(edge.get("output_pipewire_node", "")),
            segment=dict(_section(edge, "segment")),
        ),
        data_dir=data_dir,
        hf_token=hf_token,
        anthropic_api_key=anthropic_api_key,
        anthropic_workspace_id=anthropic_workspace_id,
        edge_token=edge_token,
        edge_tokens=edge_tokens,
    )
