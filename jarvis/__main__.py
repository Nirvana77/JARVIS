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
import sys
import warnings

from jarvis import app
from jarvis.config import load_config

# onnxruntime logs this once per model on a CPU-only box; it is expected.
warnings.filterwarnings("ignore", message=r".*CUDAExecutionProvider.*")


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

    if args.selftest:
        return app.selftest(config)
    if args.command == "models":
        return app.models_pull(config)
    if args.command == "nlu":
        return app.nlu_rebuild(config)

    orchestrator = app.build_orchestrator(config)
    print(f"JARVIS ready — persona '{config.persona.active}', say \"hey jarvis\". Ctrl-C to quit.")
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
