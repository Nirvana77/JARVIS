"""`config.example.toml` is the documentation people actually copy.

An example config that has drifted from the code is worse than none: it teaches
a setting that silently does nothing. So it is checked like code — it must
load, and every key in it must be a real field.
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import pytest

from jarvis.config import Config, load_config

EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.toml"

#: table name in the file -> attribute on Config
_SECTIONS = {
    "general": "general",
    "persona": "persona",
    "wake": "wake",
    "capture": "capture",
    "stt": "stt",
    "tts": "tts",
    "nlu": "nlu",
    "reasoner": "reasoner",
    "factory": "factory",
    "server": "server",
    "whisper": "whisper",
    "voder": "voder",
    "addressing": "addressing",
    "edge": "edge",
}


@pytest.fixture(scope="module")
def raw() -> dict:
    assert EXAMPLE.is_file(), f"{EXAMPLE} is missing"
    with EXAMPLE.open("rb") as fh:
        return tomllib.load(fh)


def test_the_example_config_loads():
    config = load_config(EXAMPLE)
    assert isinstance(config, Config)


def test_every_table_in_it_is_a_real_section(raw):
    unknown = sorted(set(raw) - set(_SECTIONS) - {"paths"})
    assert not unknown, f"config.example.toml has table(s) nothing reads: {unknown}"


def test_every_key_in_it_is_a_real_setting(raw):
    """A typo, or a setting that was renamed out from under the example."""
    config = load_config(EXAMPLE)
    for table, attribute in _SECTIONS.items():
        section = raw.get(table)
        if section is None:
            continue
        known = {f.name for f in dataclasses.fields(getattr(config, attribute))}
        if table == "edge":
            known.add("segment")  # [edge.segment] is a sub-table
        unknown = sorted(set(section) - known)
        assert not unknown, f"[{table}] has key(s) nothing reads: {unknown}"


def test_it_documents_every_setting_there_is(raw):
    """The other direction: a setting added to the code and never written down
    is one nobody will ever find."""
    config = load_config(EXAMPLE)
    missing: list[str] = []
    for table, attribute in _SECTIONS.items():
        documented = set(raw.get(table) or {})
        for field in dataclasses.fields(getattr(config, attribute)):
            if field.name not in documented:
                missing.append(f"[{table}] {field.name}")
    # A few are deliberately undocumented: internal ceilings nobody should be
    # tuning from a config file.
    allowed = {"[server] max_size", "[server] ping_interval_s", "[server] hello_timeout_s",
               "[edge] segment", "[whisper] timeout_s", "[voder] timeout_s"}
    assert not (set(missing) - allowed), sorted(set(missing) - allowed)


def test_it_ships_no_secrets_and_no_machine_specific_paths():
    body = EXAMPLE.read_text(encoding="utf-8")
    for leak in ("sk-ant-", "hf_", "/home/robin", "alsa_input.pci", "alsa_output.pci"):
        assert leak not in body, f"config.example.toml leaks {leak!r}"


def test_the_example_is_safe_by_default():
    """Copied verbatim and run, it must not put speech on the network in the
    clear: loopback, and no allow_insecure."""
    config = load_config(EXAMPLE)
    assert config.server.host == "127.0.0.1"
    assert config.server.allow_insecure is False
    assert config.server.tls_cert == ""


def test_the_cloudflared_example_exists_and_names_the_brain_port():
    example = EXAMPLE.parent / "services" / "cloudflared" / "config.example.yml"
    assert example.is_file()
    body = example.read_text(encoding="utf-8")
    assert "8765" in body
    assert "ingress:" in body
    assert "http_status:404" in body  # the required last rule
