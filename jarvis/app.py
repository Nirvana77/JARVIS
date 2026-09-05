"""Assembly + one-shot commands for ``python -m jarvis``.

Wires the real components together (``build_orchestrator``) and implements the
non-interactive subcommands: ``--selftest``, ``models pull``, ``nlu rebuild``,
and text mode (``build_text_orchestrator`` / ``run_text_mode`` — the real
pipeline minus audio, for scripted conversation testing; see
``jarvis/audio/text_io.py``).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from jarvis.config import Config
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.persona import Persona
from jarvis.core.reasoner import Reasoner
from jarvis.factory.claude_client import ClaudeClient
from jarvis.factory.sandbox import SubprocessSandbox
from jarvis.nlu.classifier import Classifier
from jarvis.nlu.corpus import build_corpus, intent_meta, write_corpus_db
from jarvis.nlu.train import TrainResult, latest_version, train
from jarvis.skills.registry import Registry

log = logging.getLogger(__name__)


# -- NLU bootstrap ----------------------------------------------------------

def rebuild_nlu(config: Config, registry: Registry) -> TrainResult:
    corpus = build_corpus(manifests=registry.manifests())
    write_corpus_db(corpus, config.corpus_path)
    result = train(corpus, config.nlu.embedding_model, config.nlu_model_dir)
    log.info(
        "trained NLU v%d — %d examples, %d intents, accuracy=%s",
        result.version, result.n_examples, result.n_labels, result.accuracy,
    )
    return result


def ensure_nlu(config: Config, registry: Registry) -> int:
    """Train v1 if there is no model yet. Returns the current version."""
    version = latest_version(config.nlu_model_dir)
    if version is None:
        log.info("no NLU model found — training v1")
        return rebuild_nlu(config, registry).version
    return version


def load_classifier(config: Config) -> Classifier:
    return Classifier.load(
        config.nlu_model_dir,
        config.nlu.embedding_model,
        config.nlu.threshold,
        config.nlu.similarity_floor,
    )


# -- full assembly --------------------------------------------------------

def build_orchestrator(config: Config) -> Orchestrator:
    from jarvis.audio.capture import Microphone
    from jarvis.audio.stt import Transcriber
    from jarvis.audio.tts import Speaker
    from jarvis.audio.wake import WakeWord

    # Everything heavy is loaded here, with progress, so a first-run model
    # download happens visibly at startup instead of silently mid-conversation.
    print(f"· persona '{config.persona.active}'", flush=True)
    reasoner = Reasoner.from_config(config)
    persona = Persona.load(config.persona.active, config, reasoner)
    tts = Speaker(persona.voice or config.tts.voice, config.piper_dir)
    if not tts.available():
        print(
            f"  ! Piper voice '{tts.voice}' not downloaded — replies will be "
            f"printed only. Run: python -m jarvis models pull",
            flush=True,
        )

    registry = Registry.discover(config, reasoner, say=tts.say)
    print("· NLU model", flush=True)
    ensure_nlu(config, registry)
    nlu = load_classifier(config)
    print(
        f"  NLU v{nlu.version} · {len(registry.names())} skills · "
        f"reasoner: {'ollama:' + reasoner.model if reasoner.available else 'none'}",
        flush=True,
    )

    print(
        f"· speech recogniser '{config.stt.model}' "
        f"(first run downloads it, ~30-60s)",
        flush=True,
    )
    stt = Transcriber(
        config.stt.model,
        config.stt.device,
        config.stt.compute_type,
        config.whisper_dir,
        config.stt.language,
    )
    stt.load()

    print("· wake word", flush=True)
    wake = WakeWord(config.wake.model, config.wake.threshold)
    wake.load()
    mic = Microphone(
        config.capture.sample_rate,
        config.capture.frame_samples,
        vad_threshold=config.capture.vad_threshold,
    )

    claude_client = ClaudeClient.from_config(config)
    sandbox = SubprocessSandbox(
        timeout_s=config.factory.sandbox_timeout_s,
        mem_mb=config.factory.sandbox_mem_mb,
        cpu_s=config.factory.sandbox_cpu_s,
    )
    print(
        f"· skill factory: {'claude:' + claude_client.model if claude_client.available else 'unavailable (no API key)'}"
        f" · sandbox net-isolation: {'yes' if sandbox.net_isolated else 'no (unshare unavailable)'}",
        flush=True,
    )

    return Orchestrator(
        config=config,
        wake=wake,
        mic=mic,
        stt=stt,
        tts=tts,
        nlu=nlu,
        persona=persona,
        registry=registry,
        intent_meta=intent_meta(),
        claude_client=claude_client,
        sandbox=sandbox,
    )


# -- text mode: no mic/wake-word/STT/audio-TTS, for scripted testing --------

def build_text_orchestrator(config: Config, lines: list[str] | None = None) -> Orchestrator:
    """Same wiring as `build_orchestrator`, minus everything audio: wake/mic/
    STT are `TextIO`/`TextWake` (fed `lines`, or stdin when `lines is None`)
    and TTS is a print-only stand-in — never touches real audio hardware.
    Real persona/NLU/skills/factory, so this exercises the actual pipeline."""
    from jarvis.audio.text_io import TextIO, TextTTS, TextWake

    print(f"· persona '{config.persona.active}' (text mode)", flush=True)
    reasoner = Reasoner.from_config(config)
    persona = Persona.load(config.persona.active, config, reasoner)
    tts = TextTTS()

    registry = Registry.discover(config, reasoner, say=tts.say)
    ensure_nlu(config, registry)
    nlu = load_classifier(config)

    claude_client = ClaudeClient.from_config(config)
    sandbox = SubprocessSandbox(
        timeout_s=config.factory.sandbox_timeout_s,
        mem_mb=config.factory.sandbox_mem_mb,
        cpu_s=config.factory.sandbox_cpu_s,
    )
    print(
        f"  NLU v{nlu.version} · {len(registry.names())} skills · "
        f"factory: {'claude:' + claude_client.model if claude_client.available else 'unavailable'}",
        flush=True,
    )

    orchestrator = Orchestrator(
        config=config,
        wake=TextWake(),
        mic=(text_io := TextIO(lines)),
        stt=text_io,
        tts=tts,
        nlu=nlu,
        persona=persona,
        registry=registry,
        intent_meta=intent_meta(),
        claude_client=claude_client,
        sandbox=sandbox,
    )
    text_io.on_exhausted = orchestrator.stop
    return orchestrator


def _read_script_lines(path: str) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")]


def run_text_mode(config: Config, script_path: str | None = None) -> int:
    lines = _read_script_lines(script_path) if script_path else None
    orchestrator = build_text_orchestrator(config, lines=lines)
    if lines is None:
        print(
            "JARVIS text mode — type a line, Enter to send (or pipe lines via "
            'stdin). Every line is treated as already-woken; say "shut down" '
            "or Ctrl-D to exit.",
            flush=True,
        )
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    return 0


# -- subcommands --------------------------------------------------------------

def selftest(config: Config) -> int:
    reasoner = Reasoner.from_config(config)
    registry = Registry.discover(config, reasoner)
    version = ensure_nlu(config, registry)
    nlu = load_classifier(config)
    persona = Persona.load(config.persona.active, config, reasoner)
    claude_client = ClaudeClient.from_config(config)
    sandbox = SubprocessSandbox(
        timeout_s=config.factory.sandbox_timeout_s,
        mem_mb=config.factory.sandbox_mem_mb,
        cpu_s=config.factory.sandbox_cpu_s,
    )

    print("JARVIS self-test")
    print(f"  persona        : {persona.name} ({persona.display_name})")
    print(f"  persona voice  : {persona.voice}")
    print(f"  style lines    : {len(persona.style_lines)} from {len(persona.style_line_sources)} file(s)")
    print(f"  NLU model      : v{version}  ({nlu.embedding_model})")
    print(f"  intents        : {len(nlu.labels)}  -> {', '.join(nlu.labels)}")
    print(f"  skills         : {len(registry.names())}  -> {', '.join(registry.names())}")
    print(f"  reasoner       : {'ollama:' + reasoner.model if reasoner.available else 'none (plain phrasing)'}")
    print(f"  factory        : {'claude:' + claude_client.model if claude_client.available else 'unavailable (no API key)'}")
    print(f"  sandbox        : net-isolation {'yes' if sandbox.net_isolated else 'no (unshare unavailable)'}")

    probes = [
        ("search black holes", "search"),
        ("play some jazz", "play"),
        ("open github", "open_app"),
        ("go to sleep", "goodbye"),
        ("asdfqwer zxcvzxcv", "unknown"),
    ]
    print("  sanity         :")
    ok = True
    for text, expected in probes:
        label, conf = nlu.predict(text)
        mark = "ok" if label == expected else "??"
        if label != expected:
            ok = False
        print(f"    [{mark}] {text!r:24} -> {label} ({conf:.2f}), expected {expected}")

    print("PASS" if ok else "PASS (with classification warnings)")
    return 0


def mic_meter(config: Config) -> int:
    """Live RMS meter with the same VAD maths `record_utterance` uses, so you
    can see whether normal speech — including the *ends* of sentences —
    crosses the threshold that would actually be used."""
    from jarvis.audio.capture import Microphone

    mic = Microphone(config.capture.sample_rate, config.capture.frame_samples)
    vad_override = getattr(config.capture, "vad_threshold", 0.0)
    barge_override = getattr(config.capture, "barge_in_threshold", 0.0)
    override = vad_override or barge_override
    print("Microphone level meter — speak normally, including full sentences. Ctrl-C to stop.")
    print("The │ marker is the speech threshold; a bar reaching it = 'SPEECH'.")
    print("Watch whether it stays 'SPEECH' right through the end of what you say —")
    print("if the level dips below the marker while you're still talking, set")
    print("[capture] vad_threshold in config.toml to a value below what you see here.")
    if vad_override:
        print(f"(threshold pinned to vad_threshold = {vad_override})")
    elif barge_override:
        print(f"(threshold pinned to barge_in_threshold = {barge_override})")
    print()

    mic.start()
    early: list[float] = []
    floor: float | None = None
    peak = 0.0
    span = 0.35
    try:
        i = 0
        while True:
            try:
                frame = mic.read(0.5)
            except Exception:
                continue
            rms = Microphone.frame_rms(frame)
            peak = max(peak, rms)
            if i < Microphone.CALIBRATION_FRAMES:
                early.append(rms)
            elif floor is None:
                floor = Microphone.calibrate_floor(early)
            thr = override or Microphone.speech_threshold(floor or 0.012)

            width = 52
            filled = int(min(rms, span) / span * width)
            mark = int(min(thr, span) / span * width)
            bar = "".join(
                "│" if k == mark else ("#" if k < filled else " ") for k in range(width)
            )
            tag = "SPEECH" if rms > thr else " ...  "
            print(
                f"\r  {tag}  rms={rms:.3f}  floor={floor or 0:.3f}  thr={thr:.3f}  "
                f"peak={peak:.3f}  [{bar}]",
                end="",
                flush=True,
            )
            i += 1
    except KeyboardInterrupt:
        print("\n")
    finally:
        mic.stop()
    return 0


def nlu_rebuild(config: Config) -> int:
    registry = Registry.discover(config, Reasoner.from_config(config))
    result = rebuild_nlu(config, registry)
    kept = sorted(p.name for p in config.nlu_model_dir.glob("v*"))
    print(f"NLU retrained -> v{result.version} "
          f"({result.n_examples} examples, accuracy={result.accuracy}); kept {kept}")
    return 0


def models_pull(config: Config) -> int:
    ok = True
    print(f"Hugging Face auth: {'token set' if config.hf_token else 'anonymous'}")

    print("warming fastembed NLU model...")
    try:
        from fastembed import TextEmbedding

        next(iter(TextEmbedding(model_name=config.nlu.embedding_model).embed(["warm"])))
        print("  fastembed: ready")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  fastembed: FAILED ({exc})")

    print(f"downloading faster-whisper '{config.stt.model}'...")
    try:
        from jarvis.audio.stt import Transcriber

        Transcriber(
            config.stt.model,
            config.stt.device,
            config.stt.compute_type,
            config.whisper_dir,
        ).load()
        print(f"  whisper: ready ({config.whisper_dir})")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  whisper: FAILED ({exc})")

    print(f"downloading Piper voice '{config.tts.voice}'...")
    try:
        _pull_piper_voice(config)
        print(f"  piper: ready ({config.piper_dir})")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  piper: FAILED ({exc}) — set [tts].voice to an available "
              f"rhasspy/piper-voices model")

    return 0 if ok else 1


def _pull_piper_voice(config: Config) -> None:
    from huggingface_hub import hf_hub_download

    voice = config.tts.voice  # e.g. en_GB-alan-medium
    lang, name, quality = voice.split("-", 2)
    family = lang.split("_")[0]
    base = f"{family}/{lang}/{name}/{quality}/{voice}"
    config.piper_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".onnx", ".onnx.json"):
        hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=f"{base}{suffix}",
            local_dir=str(config.piper_dir),
            token=config.hf_token,
        )
    # hf_hub_download nests the file under the repo path; flatten it
    import shutil

    for suffix in (".onnx", ".onnx.json"):
        nested = config.piper_dir / f"{base}{suffix}"
        flat = config.piper_dir / f"{voice}{suffix}"
        if nested.is_file() and nested != flat:
            shutil.copy2(nested, flat)
