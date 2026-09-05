"""``python -m jarvis`` — the entry point for the rebuilt assistant.

    python -m jarvis                 run the voice loop
    python -m jarvis --debug         run with verbose logging + barge-in diagnostics
    python -m jarvis mic             live microphone level meter (tune the VAD)
    python -m jarvis --selftest      load NLU + persona, list skills, exit 0
    python -m jarvis models pull     download whisper + Piper voice, warm fastembed
    python -m jarvis nlu rebuild     rebuild the corpus and retrain the NLU head
    python -m jarvis text            real pipeline, no mic/wake-word/STT/speaker —
                                      type lines or pipe them in; --script FILE
                                      reads a scripted conversation from a file
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import warnings

from jarvis.config import load_config

try:
    from jarvis import app
except ModuleNotFoundError as exc:  # wrong interpreter / deps not installed
    sys.exit(
        f"error: {exc.name} is missing — you're not on the project venv.\n"
        f"  run './jarvis-run {' '.join(sys.argv[1:])}'  (uses ./.venv automatically)\n"
        f"  or:  source .venv/bin/activate  &&  python -m jarvis ..."
    )

# Expected, noisy third-party chatter on a CPU-only / no-token box.
warnings.filterwarnings("ignore", message=r".*CUDAExecutionProvider.*")
warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis", description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="check the setup and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="INFO logging")
    parser.add_argument(
        "--debug", action="store_true", help="DEBUG logging + barge-in diagnostics"
    )
    sub = parser.add_subparsers(dest="command")
    models = sub.add_parser("models", help="model management")
    models.add_argument("op", choices=["pull"])
    nlu = sub.add_parser("nlu", help="NLU model management")
    nlu.add_argument("op", choices=["rebuild"])
    sub.add_parser("mic", help="live microphone level meter")
    text = sub.add_parser(
        "text", help="real pipeline, text in/out — no mic/wake-word/STT/speaker"
    )
    text.add_argument(
        "--script", help="read a scripted conversation from this file instead of stdin"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    level = (
        logging.DEBUG if args.debug
        else logging.INFO if args.verbose
        else logging.WARNING
    )
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    if config.hf_token:  # make sure every HF client sees it, under either name
        os.environ.setdefault("HF_TOKEN", config.hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", config.hf_token)

    if args.selftest:
        return app.selftest(config)
    if args.command == "models":
        return app.models_pull(config)
    if args.command == "nlu":
        return app.nlu_rebuild(config)
    if args.command == "mic":
        return app.mic_meter(config)
    if args.command == "text":
        return app.run_text_mode(config, script_path=args.script)

    try:
        orchestrator = app.build_orchestrator(config)
        extras = []
        if config.capture.barge_in:
            extras.append("voice barge-in on")
        msg = f' ({", ".join(extras)})' if extras else ""
        print(
            f"JARVIS ready — persona '{config.persona.active}', "
            f'say "hey jarvis"{msg}. Ctrl-C to quit.',
            flush=True,
        )
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        # heavy models load eagerly at startup now, so nothing long-running is
        # stuck in a worker thread here — a normal shutdown is clean.
        print("\nShutting down.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
