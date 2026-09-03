"""Assembly + one-shot commands for ``python -m jarvis``.

Wires the real components together (``build_orchestrator``) and implements the
non-interactive subcommands: ``--selftest``, ``models pull``, ``nlu rebuild``.
"""

from __future__ import annotations

import logging

from jarvis.config import Config
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.persona import Persona
from jarvis.core.reasoner import Reasoner
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
    mic = Microphone(config.capture.sample_rate, config.capture.frame_samples)

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
    )


# -- subcommands --------------------------------------------------------------

def selftest(config: Config) -> int:
    reasoner = Reasoner.from_config(config)
    registry = Registry.discover(config, reasoner)
    version = ensure_nlu(config, registry)
    nlu = load_classifier(config)
    persona = Persona.load(config.persona.active, config, reasoner)

    print("JARVIS self-test")
    print(f"  persona        : {persona.name} ({persona.display_name})")
    print(f"  persona voice  : {persona.voice}")
    print(f"  style lines    : {len(persona.style_lines)} from {len(persona.style_line_sources)} file(s)")
    print(f"  NLU model      : v{version}  ({nlu.embedding_model})")
    print(f"  intents        : {len(nlu.labels)}  -> {', '.join(nlu.labels)}")
    print(f"  skills         : {len(registry.names())}  -> {', '.join(registry.names())}")
    print(f"  reasoner       : {'ollama:' + reasoner.model if reasoner.available else 'none (plain phrasing)'}")

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
