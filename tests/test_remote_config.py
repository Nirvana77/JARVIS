"""M3 config (decision 11): the [server], [whisper], [voder], [addressing] and
[edge] sections, and the segmenter overrides under [edge.segment].

Secrets are never in here — the edge tokens live in .env and are read straight
from the environment.
"""

from __future__ import annotations

import pytest

from jarvis.config import load_config
from jarvis.remote.addressing import DEFAULT_MODE, Mode


def _write(tmp_path, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return load_config(path)


def test_the_defaults_are_the_ones_the_plan_names(tmp_path):
    config = _write(tmp_path, "")
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 8765
    assert config.server.tls_cert == "" and config.server.tls_key == ""
    assert config.server.allow_insecure is False
    assert config.whisper.url == "http://127.0.0.1:3461"
    assert config.whisper.timeout_s == 20.0
    assert config.voder.url == "http://127.0.0.1:3462"
    assert config.voder.sample_rate == 16000
    assert config.addressing.default_mode == DEFAULT_MODE
    assert config.addressing.names == ("jarvis", "hey jarvis")
    # The plan (and Mike) say 2000. JARVIS halves it deliberately: the edge's
    # `speaking{on}` pauses the hold's countdown, so the window only has to
    # notice a continuation starting, not swallow one whole — see
    # `test_the_hold_only_has_to_notice_a_continuation_not_swallow_it`. It is
    # paid on every single turn, which is what makes it worth the trade.
    assert config.addressing.hold_ms == 1000
    assert config.edge.device_id == "edge"
    assert config.edge.reconnect_max_s == 30
    assert config.edge.ptt_gpio == 0


def test_the_sections_are_read_from_the_file(tmp_path):
    config = _write(
        tmp_path,
        """
        [server]
        host = "127.0.0.1"
        port = 9000
        tls_cert = "/etc/ssl/jarvis.pem"
        tls_key = "/etc/ssl/jarvis.key"
        allow_insecure = true

        [whisper]
        url = "off"
        timeout_s = 5.0

        [voder]
        url = "http://127.0.0.1:9999"
        sample_rate = 22050

        [addressing]
        default_mode = "always"
        names = ["jarvis", "hey jarvis", "javis"]
        hold_ms = 0

        [edge]
        server_url = "wss://jarvis.example.net:8765"
        device_id = "livingroom"
        tls_ca = "/etc/ssl/ca.pem"
        reconnect_max_s = 15
        ptt_gpio = 17
        """,
    )
    assert (config.server.host, config.server.port) == ("127.0.0.1", 9000)
    assert config.server.allow_insecure is True
    assert config.whisper.url == "off"
    assert config.voder.sample_rate == 22050
    assert config.addressing.default_mode == Mode.ALWAYS
    assert "javis" in config.addressing.names
    assert config.addressing.hold_ms == 0
    assert config.edge.server_url == "wss://jarvis.example.net:8765"
    assert config.edge.device_id == "livingroom"
    assert config.edge.ptt_gpio == 17


def test_an_unknown_mode_in_the_file_falls_back_to_the_default(tmp_path):
    config = _write(tmp_path, '[addressing]\ndefault_mode = "whenever"\n')
    assert config.addressing.default_mode == DEFAULT_MODE


def test_the_segmenter_can_be_tuned_per_room(tmp_path):
    config = _write(
        tmp_path,
        """
        [edge.segment]
        hangover_ms = 900
        margin_db = 12
        """,
    )
    opts = config.edge.segmenter_options()
    assert opts.hangover_ms == 900
    assert opts.margin_db == 12
    # everything else is Mike's measured default, untouched
    assert opts.pre_roll_ms == 300
    assert opts.max_segment_ms == 15_000
    assert opts.sample_rate == 16_000


def test_an_unknown_segmenter_knob_is_a_clear_error_not_a_silent_no_op(tmp_path):
    config = _write(tmp_path, "[edge.segment]\nhangover = 900\n")
    with pytest.raises(ValueError) as exc:
        config.edge.segmenter_options()
    assert "hangover" in str(exc.value)


def test_the_edge_token_comes_from_the_environment_not_the_file(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_EDGE_TOKEN", "s3cret")
    monkeypatch.setenv("JARVIS_EDGE_TOKENS", "livingroom:s3cret,kitchen:other")
    config = _write(tmp_path, "")
    assert config.edge_token == "s3cret"
    assert config.edge_tokens == {"livingroom": "s3cret", "kitchen": "other"}
    body = (tmp_path / "config.toml").read_text()
    assert "s3cret" not in body


def test_malformed_token_entries_are_skipped_not_crashed_on(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_EDGE_TOKENS", "livingroom:s3cret, nonsense ,kitchen:other:extra")
    config = _write(tmp_path, "")
    assert config.edge_tokens["livingroom"] == "s3cret"
    assert config.edge_tokens["kitchen"] == "other:extra"
    assert "nonsense" not in config.edge_tokens


def test_the_remote_state_directory_hangs_off_the_data_dir(tmp_path):
    config = _write(tmp_path, "")
    assert config.remote_dir.name == "remote"
    assert config.remote_dir.parent == config.data_dir
