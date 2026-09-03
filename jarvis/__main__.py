"""``python -m jarvis`` — the entry point for the rebuilt assistant.

    python -m jarvis                 run the voice loop
    python -m jarvis --selftest      load NLU + persona, list skills, exit 0
    python -m jarvis models pull     download whisper + Piper voice, warm fastembed
    python -m jarvis nlu rebuild     rebuild the corpus and retrain the NLU head
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import warnings

from jarvis import app
from jarvis.config import load_config

# Expected, noisy third-party chatter on a CPU-only / no-token box.
warnings.filterwarnings("ignore", message=r".*CUDAExecutionProvider.*")
warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis", description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="check the setup and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="INFO logging")
    sub = parser.add_subparsers(dest="command")
    models = sub.add_parser("models", help="model management")
    models.add_argument("op", choices=["pull"])
    nlu = sub.add_parser("nlu", help="NLU model management")
    nlu.add_argument("op", choices=["rebuild"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
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

    try:
        orchestrator = app.build_orchestrator(config)
        print(
            f"JARVIS ready — persona '{config.persona.active}', "
            f'say "hey jarvis". Ctrl-C to quit.',
            flush=True,
        )
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        # a worker thread may be mid-transcribe / mid-download and won't join;
        # skip the atexit thread-join that would otherwise dump a traceback
        sys.stdout.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
