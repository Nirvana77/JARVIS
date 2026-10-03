"""M7 part G: ``python -m jarvis learning …`` reads and undoes what was learned.

Pointed at a tmp ``data_dir``; never the repo's ``data/``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

from jarvis.__main__ import _build_parser
from jarvis.learning import Learning
from jarvis.learning import cli


def make(tmp_path, config):
    cfg = dataclasses.replace(config, data_dir=tmp_path)
    learning = Learning.from_config(cfg)
    return cfg, learning


def test_the_parser_knows_the_learning_commands():
    args = _build_parser().parse_args(["learning", "undo", "abc123"])
    assert (args.command, args.op, args.arg) == ("learning", "undo", "abc123")
    args = _build_parser().parse_args(["learning", "log", "--since", "2d"])
    assert args.since == "2d"
    assert _build_parser().parse_args(["learning"]).op == "status"


def test_status_counts_what_is_there(tmp_path, config, capsys):
    cfg, learning = make(tmp_path, config)
    learning.phrasings.add("flip a corn", "flip_a_coin", why="mishear")
    learning.state.disable("roll_a_die", "kept failing")
    assert cli.run(cfg, "status") == 0
    out = capsys.readouterr().out
    assert "1 learned phrasing" in out and "roll_a_die" in out


def test_log_prints_turns_since(tmp_path, config, capsys):
    cfg, learning = make(tmp_path, config)
    learning.log.append({"device": "watch", "heard": "flip a corn", "path": "mishear",
                         "skill": "flip_a_coin", "outcome": "ok"})
    assert cli.run(cfg, "log", since="1d") == 0
    out = capsys.readouterr().out
    assert "flip a corn" in out and "mishear" in out and "flip_a_coin" in out


def test_since_understands_days_and_dates():
    now = dt.datetime(2026, 10, 3, 12, 0)
    assert cli.parse_since("2d", now) == dt.datetime(2026, 10, 1, 12, 0)
    assert cli.parse_since("2026-09-30", now) == dt.datetime(2026, 9, 30)
    assert cli.parse_since(None, now) is None


def test_undo_removes_a_phrasing_for_good(tmp_path, config, capsys):
    cfg, learning = make(tmp_path, config)
    p = learning.phrasings.add("flip a corn", "flip_a_coin", why="mishear")
    assert cli.run(cfg, "undo", arg=p.id) == 0
    assert Learning.from_config(cfg).phrasings.examples() == []
    assert Learning.from_config(cfg).phrasings.is_rejected("flip a corn", "flip_a_coin")
    assert cli.run(cfg, "undo", arg="nope") == 1


def test_enable_switches_a_skill_back_on(tmp_path, config):
    cfg, learning = make(tmp_path, config)
    learning.state.disable("roll_a_die", "kept failing")
    assert cli.run(cfg, "enable", arg="roll_a_die") == 0
    assert not Learning.from_config(cfg).state.is_disabled("roll_a_die")
    assert cli.run(cfg, "enable", arg="roll_a_die") == 1  # was not off
