"""M5: the `[knowledge]` section of config.toml."""

from __future__ import annotations

from pathlib import Path

from jarvis.config import KnowledgeConfig, load_config


def _write(tmp_path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_defaults_are_the_prds(tmp_path):
    config = load_config(_write(tmp_path, ""))

    assert config.knowledge == KnowledgeConfig()
    assert config.knowledge.enabled is True
    assert config.knowledge.docs_path == Path.home() / "jarvis" / "knowledge"
    assert config.knowledge_db_path == config.data_dir / "knowledge" / "kb.sqlite"


def test_the_section_is_read(tmp_path):
    config = load_config(_write(tmp_path, f"""
[knowledge]
enabled = false
docs_dir = "{tmp_path}/docs"
scan_interval_s = 5
top_k = 2
min_score = 0.5
search_min_score = 0.7
chunk_chars = 400
chunk_overlap = 40
"""))

    assert config.knowledge == KnowledgeConfig(
        enabled=False,
        docs_dir=f"{tmp_path}/docs",
        scan_interval_s=5.0,
        top_k=2,
        min_score=0.5,
        search_min_score=0.7,
        chunk_chars=400,
        chunk_overlap=40,
    )
    assert config.knowledge.docs_path == tmp_path / "docs"


def test_values_that_would_hang_or_break_it_are_clamped(tmp_path):
    """`chunk_chars = 0` would never finish chunking; sqlite-vec refuses a k
    above 4096; an interval of 0 would scan without pause."""
    config = load_config(_write(tmp_path, """
[knowledge]
scan_interval_s = 0
top_k = 0
chunk_chars = 0
chunk_overlap = 5000
"""))

    assert config.knowledge.scan_interval_s == 1.0
    assert config.knowledge.top_k == 1
    assert config.knowledge.chunk_chars == 100
    assert config.knowledge.chunk_overlap == 99
