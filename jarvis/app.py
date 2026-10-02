"""Assembly + one-shot commands for ``python -m jarvis``.

Wires the real components together (``build_orchestrator``) and implements the
non-interactive subcommands: ``--selftest``, ``models pull``, ``nlu rebuild``,
and text mode (``build_text_orchestrator`` / ``run_text_mode`` — the real
pipeline minus audio, for scripted conversation testing; see
``jarvis/audio/text_io.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from jarvis import knowledge as knowledge_base
from jarvis.config import Config
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.persona import Persona
from jarvis.core.memory import Memory
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
    """Train v1 if there is no model yet — or retrain if a registered skill
    with examples is missing from the model's labels (a builtin added since
    the last training), so a new skill is never silently unreachable. Returns
    the current version."""
    version = latest_version(config.nlu_model_dir)
    if version is None:
        log.info("no NLU model found — training v1")
        return rebuild_nlu(config, registry).version
    try:
        labels_path = config.nlu_model_dir / f"v{version}" / "labels.json"
        labels = set(json.loads(labels_path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return version
    missing = sorted(m.name for m in registry.manifests() if m.examples and m.name not in labels)
    if missing:
        log.info("NLU v%d does not know %s — retraining", version, ", ".join(missing))
        return rebuild_nlu(config, registry).version
    return version


def load_classifier(config: Config) -> Classifier:
    return Classifier.load(
        config.nlu_model_dir,
        config.nlu.embedding_model,
        config.nlu.threshold,
        config.nlu.similarity_floor,
    )


# -- M5: the knowledge base ---------------------------------------------------

def _knowledge_summary(knowledge) -> str:
    stats = knowledge.store.stats()
    return (
        f"{stats['files']} file(s), {stats['facts']} remembered fact(s), "
        f"{stats['chunks']} chunk(s) · index: {stats['backend']}"
    )


def build_knowledge(config: Config):
    """The knowledge base, brought in line with the docs folder *before* JARVIS
    says it is ready — so the first question is asked of a current index, and a
    first-run embedding of a large folder happens visibly here, not silently
    behind the first few turns. Later changes are picked up by the interval
    task the orchestrator runs (`Knowledge.watch`). ``None`` when it is off."""
    knowledge = knowledge_base.build(config)
    if knowledge is None:
        print("· knowledge base: off", flush=True)
        return None
    print(f"· knowledge base ({knowledge.docs_dir})", flush=True)
    try:
        result = knowledge.scan()
        changed = f" · {result.summary()}" if result.changed else ""
        print(f"  {_knowledge_summary(knowledge)}{changed}", flush=True)
    except Exception as exc:  # noqa: BLE001 - JARVIS still starts without it
        log.warning("knowledge scan failed: %s", exc)
        print(f"  ! could not scan it: {exc}", flush=True)
    return knowledge


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
    tts = Speaker(
        persona.voice or config.tts.voice,
        config.piper_dir,
        pipewire_node=config.tts.pipewire_node,
    )
    if not tts.available():
        print(
            f"  ! Piper voice '{tts.voice}' not downloaded — replies will be "
            f"printed only. Run: python -m jarvis models pull",
            flush=True,
        )

    memory = Memory.from_config(config)
    knowledge = build_knowledge(config)
    registry = Registry.discover(
        config, reasoner, say=tts.say, memory=memory, knowledge=knowledge
    )
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
        device=config.capture.device,
        pipewire_node=config.capture.pipewire_node,
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
        reasoner=reasoner,
        memory=memory,
        claude_client=claude_client,
        sandbox=sandbox,
        knowledge=knowledge,
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

    memory = Memory.from_config(config)
    knowledge = build_knowledge(config)
    registry = Registry.discover(
        config, reasoner, say=tts.say, memory=memory, knowledge=knowledge
    )
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
        reasoner=reasoner,
        memory=memory,
        claude_client=claude_client,
        sandbox=sandbox,
        knowledge=knowledge,
    )
    text_io.on_exhausted = orchestrator.stop
    return orchestrator


# -- M3: the brain, with the audio layer out on an edge device -------------

def build_server_orchestrator(config: Config, link, edges=None) -> Orchestrator:
    """Same wiring as `build_orchestrator`, with `link` (a `RemoteLink`) in
    place of wake/mic/STT/TTS — the segments are transcribed by
    `jarvis-whisper` and spoken by `jarvis-voder`, both out of process, so
    nothing heavy loads here. Real persona/NLU/skills/factory, exactly as the
    all-in-one path has them."""
    print(f"· persona '{config.persona.active}' (remote edge)", flush=True)
    reasoner = Reasoner.from_config(config)
    persona = Persona.load(config.persona.active, config, reasoner)
    # The voder speaks in the persona's own voice, out on the service.
    if getattr(link.voder, "voice", None) is None and persona.voice:
        link.voder.voice = persona.voice

    memory = Memory.from_config(config)
    knowledge = build_knowledge(config)
    registry = Registry.discover(
        config, reasoner, say=link.say, edges=edges,
        memory=memory, knowledge=knowledge,
    )
    print("· NLU model", flush=True)
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
        wake=link,
        mic=link,
        stt=link,
        tts=link,
        nlu=nlu,
        persona=persona,
        registry=registry,
        intent_meta=intent_meta(),
        reasoner=reasoner,
        memory=memory,
        claude_client=claude_client,
        sandbox=sandbox,
        knowledge=knowledge,
        # A server: "shut down" from the watch stands by, it does not stop the
        # brain (Ctrl-C / systemd still do).
        allow_shutdown=False,
    )
    link.attach(orchestrator)
    return orchestrator


def run_server_mode(config: Config) -> int:
    """`python -m jarvis serve` — the brain, waiting for an edge."""
    from jarvis.remote.server import RemoteLink, RemoteServer
    from jarvis.remote.supervisor import (
        ServiceSupervisor,
        voder_service,
        whisper_service,
    )

    link = RemoteLink(config)
    try:
        server = RemoteServer(config, link)
    except ValueError as exc:  # a mistyped [server] trusted_proxy_peers entry
        print(f"error: {exc}", flush=True)
        return 2
    persona_voice = Persona.load(config.persona.active, config, None).voice

    # The brain owns its services: one command, not three terminals. They stay
    # separate processes — a wedged model is one process to kill — and one that
    # is already answering (systemd, or a previous brain) is adopted rather
    # than started twice.
    supervisor = ServiceSupervisor(
        [whisper_service(config), voder_service(config, persona_voice)]
    )
    print("· services", flush=True)
    for name, status in supervisor.start_all().items():
        mark = "·" if status in ("started", "adopted") else "!"
        print(f"  {mark} {name}: {status}", flush=True)

    _report_service(
        "speech recogniser", config.whisper.url,
        lambda: server.transcriber.health(),
        lambda h: f"{h.get('model')} on {h.get('device')}{'' if h.get('warm') else ' (cold)'}",
    )
    _report_service(
        "voder", config.voder.url,
        lambda: link.voder.health(),
        lambda h: f"{h.get('voice')} at {h.get('sample_rate')} Hz",
    )
    orchestrator = build_server_orchestrator(config, link, edges=server.edges)
    # An edge with new tools (new firmware): learn them in the background; the
    # new model lands at the next safe point, as a learned skill's does.
    server.on_tools_changed = lambda _device: asyncio.ensure_future(orchestrator.refresh_skills())

    async def main() -> None:
        link.bind_loop()
        ws = await server.serve()
        # Keeps them up while the brain runs: a model that falls over
        # mid-conversation comes back by itself.
        watchdog = asyncio.ensure_future(supervisor.watch())
        scheme = "wss" if config.server.tls_enabled else "ws"
        print(
            f"JARVIS brain ready — {scheme}://{config.server.host}:{config.server.port}, "
            f"{len(server.tokens)} device token(s), default mode "
            f"'{config.addressing.default_mode}'. Ctrl-C to quit.",
            flush=True,
        )
        try:
            await orchestrator.run()
        finally:
            watchdog.cancel()
            try:
                await watchdog
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            ws.close()
            await ws.wait_closed()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    except RuntimeError as exc:  # a refused insecure bind says why, once
        print(f"error: {exc}", flush=True)
        return 2
    finally:
        # only the ones we started; an adopted service is left running
        supervisor.stop_all()
    return 0


def _device_token(config: Config, device_id: str | None) -> str | None:
    """A device's token: .env's, else the one pairing handed out."""
    from jarvis.remote.pairing import TokenStore

    if not device_id:
        return None
    return config.edge_tokens.get(device_id) or TokenStore(config.remote_dir).get(device_id)


def _first_device(config: Config) -> str | None:
    from jarvis.remote.pairing import TokenStore

    return next(iter(config.edge_tokens), None) or next(iter(TokenStore(config.remote_dir).devices()), None)


def pair_request(config: Config, code: str):
    """The ``GET /pair`` that confirms a device's code: ``(url, headers)``,
    with this machine's admin secret."""
    from jarvis.remote.pairing import admin_secret

    digits = "".join(ch for ch in code if ch.isdigit())
    scheme = "https" if config.server.tls_enabled else "http"
    url = f"{scheme}://127.0.0.1:{config.server.port}/pair?code={digits}"
    return url, {"Authorization": f"Bearer {admin_secret(config.remote_dir)}"}


def pair(config: Config, code: str) -> int:
    """`python -m jarvis pair 482913` — approve the device showing that code."""
    import ssl
    import urllib.error
    import urllib.request

    url, headers = pair_request(config, code)
    if len(url.rsplit("=", 1)[1]) != 6:
        print("error: the code is 6 digits, as the device shows it", flush=True)
        return 2
    context = ssl._create_unverified_context() if url.startswith("https") else None  # loopback
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=15, context=context
        ) as resp:
            print(resp.read().decode().strip().replace("paired:", "Paired:"))
            return 0
    except urllib.error.HTTPError as exc:
        reason = exc.read().decode().strip() if exc.code == 404 else f"{exc.code} {exc.reason}"
        print(f"error: {reason}", flush=True)
        return 1
    except OSError as exc:
        print(f"error: no brain on {url.split('/pair')[0]} ({exc})", flush=True)
        return 1


def devices(config: Config, remove: str | None = None) -> int:
    """`python -m jarvis devices [--remove ID]` — who may connect, and how."""
    from jarvis.remote.pairing import TokenStore

    store = TokenStore(config.remote_dir)
    if remove:
        if store.remove(remove):
            print(f"Removed {remove}: its token no longer works (from its next connect).")
            return 0
        if remove in config.edge_tokens:
            print(f"{remove} has its token in .env (JARVIS_EDGE_TOKENS): remove it there.")
            return 1
        print(f"No device {remove}.")
        return 1
    rows = [(d, ".env") for d in sorted(config.edge_tokens)]
    rows += [(d, "paired") for d in store.devices() if d not in config.edge_tokens]
    rows += [(d, ".env + paired") for d in store.devices() if d in config.edge_tokens]
    rows = sorted(dict(rows).items())
    if not rows:
        print("No devices yet: pair one (a device without a token shows a code).")
        return 0
    for device_id, source in rows:
        print(f"  {device_id:<20} {source}")
    return 0


def notify_request(config: Config, text: str, device_id: str | None = None):
    """The ``GET /notify`` the running brain on this machine takes: ``(url,
    headers)``. The device is the one named, else the one whose tools include
    ``notify``, else the first with a token."""
    from urllib.parse import urlencode

    from jarvis.skills.edge import EdgeTools

    if device_id is None:
        store = EdgeTools(config.remote_dir)
        device_id = next(
            (d for d in store.devices() if any(t["name"] == "notify" for t in store.tools(d))),
            _first_device(config),
        )
    token = _device_token(config, device_id)
    if not token:
        raise ValueError("no device token for notify (pair one, or JARVIS_EDGE_TOKENS in .env)")
    scheme = "https" if config.server.tls_enabled else "http"
    url = f"{scheme}://127.0.0.1:{config.server.port}/notify?{urlencode({'text': text})}"
    return url, {"Authorization": f"Bearer {token}", "X-Jarvis-Device": device_id}


def power_request(config: Config, day: str | None = None, fetch: bool = True):
    """The ``GET /power`` the running brain on this machine takes: ``(url,
    headers)``, for the device with ``send_power_log`` (else the first)."""
    from urllib.parse import urlencode

    from jarvis.skills.edge import EdgeTools

    store = EdgeTools(config.remote_dir)
    device_id = next(
        (d for d in store.devices() if any(t["name"] == "send_power_log" for t in store.tools(d))),
        _first_device(config),
    )
    token = _device_token(config, device_id)
    if not token:
        raise ValueError("no device token (pair one, or JARVIS_EDGE_TOKENS in .env)")
    query = {"day": day} if day else {}
    query["fetch"] = "1" if fetch else "0"
    scheme = "https" if config.server.tls_enabled else "http"
    url = f"{scheme}://127.0.0.1:{config.server.port}/power?{urlencode(query)}"
    return url, {"Authorization": f"Bearer {token}", "X-Jarvis-Device": device_id}


def power(config: Config, day: str | None = None, fetch: bool = True) -> int:
    """`python -m jarvis power [--day YYYY-MM-DD] [--no-fetch]` — fetch the
    watch's power log through the running brain and print the day."""
    import ssl
    import urllib.error
    import urllib.request

    try:
        url, headers = power_request(config, day, fetch)
    except ValueError as exc:
        print(f"error: {exc}", flush=True)
        return 2
    context = ssl._create_unverified_context() if url.startswith("https") else None  # loopback
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=180, context=context
        ) as resp:
            print(resp.read().decode().rstrip())
    except urllib.error.HTTPError as exc:
        print(f"error: the brain said {exc.code} {exc.reason}", flush=True)
        return 1
    except OSError as exc:
        print(f"error: no brain on {url.split('/power')[0]} ({exc})", flush=True)
        return 1
    return 0


def notify(config: Config, text: str, device_id: str | None = None) -> int:
    """`python -m jarvis notify "the build is done"` — shown on the watch now,
    or as soon as it connects. For the end of a long command:
    ``make && python -m jarvis notify "Build done"``."""
    import ssl
    import urllib.error
    import urllib.request

    try:
        url, headers = notify_request(config, text, device_id)
    except ValueError as exc:
        print(f"error: {exc}", flush=True)
        return 2
    # Loopback to our own listener: its certificate names the public host,
    # not 127.0.0.1, and there is nobody in between to check for.
    context = ssl._create_unverified_context() if url.startswith("https") else None
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=15, context=context
        ) as resp:
            status = resp.read().decode().strip()
    except urllib.error.HTTPError as exc:
        print(f"error: the brain said {exc.code} {exc.reason}", flush=True)
        return 1
    except OSError as exc:
        print(f"error: no brain on {url.split('/notify')[0]} ({exc})", flush=True)
        return 1
    print({"sent": "Sent.", "queued": "Queued: the watch gets it when it connects.",
           "unsupported": "The watch has no notifications (old firmware?)."}.get(status, status))
    return 0 if status in ("sent", "queued") else 1


def _report_service(label: str, url: str, health, describe) -> None:
    if not url or url.strip().lower() == "off":
        print(f"· {label}: off", flush=True)
        return
    try:
        print(f"· {label}: {describe(health())} ({url})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {label} at {url} is not answering ({exc}) — starting anyway", flush=True)


def run_edge_mode(config: Config) -> int:
    """`python -m jarvis edge` — the audio satellite. Imported lazily so the
    edge's process never loads a model."""
    from jarvis.remote.edge import run_edge

    return run_edge(config)


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
    knowledge = knowledge_base.build(config)
    print(
        "  knowledge      : "
        + (f"{_knowledge_summary(knowledge)} · {knowledge.docs_dir}" if knowledge else "off")
    )

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

    mic = Microphone(
        config.capture.sample_rate,
        config.capture.frame_samples,
        device=config.capture.device,
        pipewire_node=config.capture.pipewire_node,
    )
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
    source = config.capture.pipewire_node or config.capture.device
    print(f"Input: {source if source is not None else 'system default'}"
          " ([capture] pipewire_node / device in config.toml)")
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


def knowledge_command(config: Config, op: str) -> int:
    """`python -m jarvis knowledge scan|status` — index the docs folder now
    instead of waiting for the next interval, or see what is in the index."""
    knowledge = knowledge_base.build(config)
    if knowledge is None:
        print("The knowledge base is off ([knowledge] enabled = false).")
        return 1
    if op == "scan":
        result = knowledge.scan()
        for label, names in (
            ("added", result.added), ("updated", result.updated),
            ("removed", result.removed), ("unreadable", result.failed),
        ):
            for name in names:
                print(f"  {label:<10} {name}")
        print(f"Scanned {knowledge.docs_dir}: {result.summary()}.")
    else:
        print(f"Docs folder : {knowledge.docs_dir}")
        print(f"Database    : {config.knowledge_db_path}")
        for path, kind, chunks, when in knowledge.store.sources():
            if kind == "file":
                print(f"  {chunks:>4} chunk(s)  {when:%Y-%m-%d %H:%M}  {path}")
    print(f"Knowledge base: {_knowledge_summary(knowledge)}.")
    knowledge.close()
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
