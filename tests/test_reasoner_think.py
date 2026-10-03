"""M7 part A: qwen3-class models think before they answer unless told not to.

Measured on the cluster's Ollama (2026-10-03): the same mishearing guess took
0.25 s with ``"think": false`` and 12.5 s without it, past ``guess_timeout_s``.
So the reasoner sends the flag, and ``[reasoner] think`` is false by default.
"""

from __future__ import annotations

import dataclasses

from jarvis.config import ReasonerConfig, load_config
from jarvis.core import reasoner as reasoner_mod
from jarvis.core.reasoner import Reasoner


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _capture(monkeypatch):
    sent = {}

    def post(url, json=None, timeout=None):
        sent["url"], sent["body"] = url, json
        return _Resp({"response": " flip a coin "})

    monkeypatch.setattr(reasoner_mod.requests, "post", post)
    return sent


def test_think_defaults_to_off():
    assert ReasonerConfig().think is False


def test_the_think_flag_reaches_the_request(monkeypatch):
    sent = _capture(monkeypatch)
    r = Reasoner("http://ollama:11434", "qwen3:8b", think=False)
    assert r.generate("sys", "prompt") == "flip a coin"
    assert sent["body"]["think"] is False
    assert sent["url"] == "http://ollama:11434/api/generate"


def test_thinking_can_be_turned_back_on(monkeypatch):
    sent = _capture(monkeypatch)
    Reasoner("http://ollama:11434", "qwen3:8b", think=True).generate("sys", "prompt")
    assert sent["body"]["think"] is True


def test_from_config_carries_think(tmp_path):
    (tmp_path / "config.toml").write_text(
        '[reasoner]\nenabled = false\nmodel = "qwen3:8b"\nthink = true\n', encoding="utf-8"
    )
    config = load_config(tmp_path / "config.toml")
    assert config.reasoner.think is True
    r = Reasoner.from_config(config)
    assert r.think is True and r.model == "qwen3:8b"
    assert r.available is False  # disabled: never probed


def test_an_unreachable_reasoner_is_still_just_unavailable():
    config = dataclasses.replace(
        load_config(),
        reasoner=ReasonerConfig(enabled=True, base_url="http://127.0.0.1:9"),
    )
    assert Reasoner.from_config(config).available is False
