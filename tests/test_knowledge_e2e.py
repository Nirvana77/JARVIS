"""M5 end to end: the PRD's verification for the knowledge base.

    ingest a sample doc, ask "Jarvis, what does X say about Y", verify a
    grounded answer with a source and **no network call to Claude**

The real pipeline (`build_text_orchestrator`: real NLU, registry, persona and
skills, real MiniLM embeddings) over a tmp docs dir and a tmp data dir — never
the user's own ``~/jarvis/knowledge`` or the repo's ``data/``. The reasoner is
switched off so the answer is the verbatim-snippet tier, which is exact.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import shutil
import subprocess
import sys
import threading

import pytest

from jarvis import app
from jarvis.factory.claude_client import ClaudeClient
from jarvis.skills.builtin import search
from jarvis.skills.registry import Registry
from tests.knowledge_harness import FakeEmbed, with_knowledge_paths

MANUAL = """# Coffee machine

Descale the coffee machine every two months with citric acid.

The water tank holds one and a half litres and should be emptied every evening.
"""


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory, config, embedder):
    """An NLU model that knows every registered skill, trained once: each test
    copies it into its own data dir instead of training again."""
    scratch = tmp_path_factory.mktemp("kb_e2e_nlu")
    cfg = dataclasses.replace(config, data_dir=scratch)
    app.rebuild_nlu(cfg, Registry.discover(cfg))
    return scratch / "models"


@pytest.fixture
def cfg(config, tmp_path, trained_models):
    cfg = with_knowledge_paths(config, tmp_path)
    shutil.copytree(trained_models, cfg.data_dir / "models")
    return dataclasses.replace(
        cfg,
        reasoner=dataclasses.replace(cfg.reasoner, enabled=False),
        anthropic_api_key="blocked-for-this-test",
    )


@pytest.fixture
def no_claude(monkeypatch):
    """Any call to Claude is recorded — and, with the key blocked, could not
    have succeeded anyway."""
    calls: list = []

    def recorded(self, *a, **k):
        calls.append((a, k))
        raise AssertionError("Claude was called for a knowledge answer")

    monkeypatch.setattr(ClaudeClient, "generate_skill", recorded)
    return calls


def _converse(cfg, lines: list[str], capsys, before_run=None) -> str:
    orch = app.build_text_orchestrator(cfg, lines=lines)
    if before_run is not None:
        # Patches to a skill module go in here: `Registry.discover` reloads
        # the skill modules, which would undo one made any earlier.
        before_run()
    asyncio.run(asyncio.wait_for(orch.run(), timeout=60.0))
    assert orch.running is False
    return capsys.readouterr().out


def _replies(out: str) -> list[str]:
    return [ln.removeprefix("Jarvis: ") for ln in out.splitlines() if ln.startswith("Jarvis: ")]


# -- the PRD's verification -----------------------------------------------------

def test_a_document_is_answered_from_with_its_source_and_claude_is_never_called(
    cfg, capsys, no_claude
):
    docs = cfg.knowledge.docs_path
    docs.mkdir(parents=True)
    (docs / "coffee-manual.md").write_text(MANUAL, encoding="utf-8")
    before = threading.active_count()

    out = _converse(cfg, ["what does the coffee manual say about descaling", "shut down"], capsys)

    assert "intent  : recall" in out
    assert (
        "From your notes, sir: Coffee machine: Descale the coffee machine every "
        "two months with citric acid — from coffee manual."
    ) in _replies(out)
    assert no_claude == []
    assert threading.active_count() == before
    assert cfg.knowledge_db_path.is_file()


def test_what_was_said_is_recalled_and_wikipedia_still_answers_the_rest(
    cfg, capsys, no_claude, monkeypatch
):
    asked: list[str] = []

    def title(_session, query):
        asked.append(query)
        return "Photosynthesis"

    def no_network():
        monkeypatch.setattr(search, "_search_title", title)
        monkeypatch.setattr(
            search, "_summary",
            lambda s, t: {"type": "standard", "extract": "Photosynthesis is how plants make sugar."},
        )

    out = _converse(
        cfg,
        [
            "remember that i parked on level three",
            "where did i park",
            "note that the door code is 4821",
            "what is the door code",
            "what is photosynthesis",
            "shut down",
        ],
        capsys,
        before_run=no_network,
    )
    replies = _replies(out)

    assert "I'll remember that, sir." in replies
    assert any(
        r.startswith("From your notes, sir: i parked on level three — from what you told me on ")
        for r in replies
    ), replies
    assert "Noted, sir." in replies
    # "what is ..." is answered from the notes when they have it...
    assert any(
        r.startswith("From your notes, sir: the door code is 4821") and "dictated notes" in r
        for r in replies
    ), replies
    # ...and from Wikipedia when they do not
    assert "According to Wikipedia, sir: Photosynthesis is how plants make sugar." in replies
    assert asked == ["photosynthesis"]
    assert no_claude == []
    # the note was mirrored into the docs dir
    assert "the door code is 4821" in (
        cfg.knowledge.docs_path / "dictated-notes.md"
    ).read_text("utf-8")


def test_with_the_knowledge_base_off_recall_says_so(cfg, capsys):
    off = dataclasses.replace(cfg, knowledge=dataclasses.replace(cfg.knowledge, enabled=False))

    out = _converse(off, ["where did i park", "shut down"], capsys)

    assert "My knowledge base is switched off, sir." in _replies(out)
    assert not off.knowledge_db_path.exists()


# -- the orchestrator owns the interval task ----------------------------------------

class WatchedKnowledge:
    """Stands in for ``Knowledge``: only records what ``run()`` does with it."""

    def __init__(self) -> None:
        self.started = self.cancelled = False

    async def watch(self) -> None:
        self.started = True
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def test_run_starts_the_scan_task_and_stops_it_on_the_way_out(cfg):
    orch = app.build_text_orchestrator(cfg, lines=["shut down"])
    orch.knowledge = watched = WatchedKnowledge()

    asyncio.run(asyncio.wait_for(orch.run(), timeout=30.0))

    assert watched.started and watched.cancelled


def test_the_builders_hand_the_same_knowledge_base_to_skills_and_orchestrator(cfg):
    orch = app.build_text_orchestrator(cfg, lines=[])
    assert orch.knowledge is not None
    assert orch.registry.knowledge is orch.knowledge
    assert orch.registry.rebuilt().knowledge is orch.knowledge


# -- nothing here can reach Claude --------------------------------------------------

def test_the_knowledge_base_and_its_skills_never_import_anthropic():
    script = "\n".join([
        "import json, sys",
        "import jarvis.knowledge, jarvis.knowledge.store, jarvis.knowledge.ingest",
        "import jarvis.knowledge.answer",
        "from jarvis.skills.builtin import recall, remember, note, search",
        "print(json.dumps('anthropic' in sys.modules))",
    ])
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=True
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) is False


# -- `python -m jarvis knowledge scan|status` ------------------------------------------

@pytest.fixture
def fake_model(monkeypatch):
    """The CLI builds its own knowledge base; give it the fake embedder."""
    monkeypatch.setattr("jarvis.knowledge.fastembed_embedder", lambda _name: FakeEmbed())


def test_knowledge_scan_reports_what_changed(config, tmp_path, capsys, fake_model):
    cfg = with_knowledge_paths(config, tmp_path)
    docs = cfg.knowledge.docs_path
    docs.mkdir(parents=True)
    (docs / "coffee-manual.md").write_text(MANUAL, encoding="utf-8")

    assert app.knowledge_command(cfg, "scan") == 0
    out = capsys.readouterr().out
    assert "added" in out and "coffee-manual.md" in out
    assert "1 file(s)" in out and "2 chunk(s)" in out

    assert app.knowledge_command(cfg, "scan") == 0
    assert "no changes" in capsys.readouterr().out


def test_knowledge_status_lists_the_sources(config, tmp_path, capsys, fake_model):
    cfg = with_knowledge_paths(config, tmp_path)
    docs = cfg.knowledge.docs_path
    docs.mkdir(parents=True)
    (docs / "coffee-manual.md").write_text(MANUAL, encoding="utf-8")
    app.knowledge_command(cfg, "scan")
    capsys.readouterr()

    assert app.knowledge_command(cfg, "status") == 0
    out = capsys.readouterr().out
    assert "coffee-manual.md" in out
    assert "sqlite-vec" in out or "numpy" in out
    assert str(docs) in out


def test_knowledge_commands_say_when_it_is_off(config, tmp_path, capsys):
    cfg = with_knowledge_paths(config, tmp_path, enabled=False)
    assert app.knowledge_command(cfg, "status") == 1
    assert "enabled = false" in capsys.readouterr().out


def test_the_cli_knows_the_subcommand():
    from jarvis.__main__ import _build_parser

    args = _build_parser().parse_args(["knowledge", "scan"])
    assert (args.command, args.op) == ("knowledge", "scan")
