"""M7 parts B and C, the storage half: the interaction log, the learned
phrasings and the learning state (``jarvis/learning/``).

Everything here is pointed at ``tmp_path``. Nothing writes to the repo's
``data/``, which on the owner's machine is also the production pod's.
"""

from __future__ import annotations

import datetime as dt
import json
import threading

from jarvis.learning import Learning
from jarvis.learning.interactions import InteractionLog
from jarvis.learning.phrasings import Phrasings
from jarvis.learning.state import LearningState
from jarvis.nlu.corpus import build_corpus


class Clock:
    def __init__(self, when=dt.datetime(2026, 10, 3, 12, 0)):
        self.when = when

    def __call__(self):
        return self.when


# -- the interaction log --------------------------------------------------------

def test_a_turn_is_one_json_line_under_its_device_and_day(tmp_path):
    log = InteractionLog(tmp_path, now=Clock())
    rec = log.append({"device": "watch", "heard": "flip a coin", "label": "flip_a_coin",
                      "confidence": 0.9, "path": "direct", "outcome": "ok"})
    path = tmp_path / "watch" / "2026-10-03.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["heard"] == "flip a coin" and stored["id"] == rec["id"]
    assert stored["ts"].startswith("2026-10-03T12:00")


def test_records_come_back_oldest_first_and_by_device(tmp_path):
    clock = Clock()
    log = InteractionLog(tmp_path, now=clock)
    log.append({"device": "watch", "heard": "one"})
    clock.when += dt.timedelta(days=1)
    log.append({"device": "local", "heard": "two"})
    log.append({"device": "watch", "heard": "three"})
    assert [r["heard"] for r in log.records()] == ["one", "two", "three"]
    assert [r["heard"] for r in log.records(device="watch")] == ["one", "three"]
    since = dt.datetime(2026, 10, 4)
    assert [r["heard"] for r in log.records(since=since)] == ["two", "three"]


def test_ids_are_unique(tmp_path):
    log = InteractionLog(tmp_path, now=Clock())
    ids = {log.append({"device": "local", "heard": str(n)})["id"] for n in range(50)}
    assert len(ids) == 50


def test_an_odd_device_name_cannot_escape_the_folder(tmp_path):
    log = InteractionLog(tmp_path / "interactions", now=Clock())
    log.append({"device": "../../etc", "heard": "x"})
    assert not (tmp_path / "etc").exists()
    assert [r["device"] for r in log.records()] == ["local"]


def test_forget_deletes_that_device_only(tmp_path):
    log = InteractionLog(tmp_path, now=Clock())
    log.append({"device": "watch", "heard": "a"})
    log.append({"device": "local", "heard": "b"})
    assert log.forget("watch") == 1
    assert [r["heard"] for r in log.records()] == ["b"]
    assert not (tmp_path / "watch").exists()


def test_rotation_drops_days_past_keep_days(tmp_path):
    clock = Clock()
    log = InteractionLog(tmp_path, keep_days=2, now=clock)
    log.append({"device": "local", "heard": "old"})
    clock.when += dt.timedelta(days=3)
    log.append({"device": "local", "heard": "new"})
    assert log.rotate() == 1
    assert [r["heard"] for r in log.records()] == ["new"]


def test_a_log_without_a_folder_keeps_records_in_ram():
    log = InteractionLog(None, now=Clock())
    log.append({"device": "local", "heard": "a"})
    assert [r["heard"] for r in log.records()] == ["a"]
    assert log.forget("local") == 1
    assert log.records() == []


# -- learned phrasings ------------------------------------------------------------

def test_a_phrasing_is_kept_and_survives_a_reload(tmp_path):
    path = tmp_path / "phrasings.json"
    p = Phrasings(path, now=Clock())
    added = p.add("flip a corn", "flip_a_coin", why="mishear", turn="t1")
    assert added is not None
    again = Phrasings(path, now=Clock())
    assert again.examples() == [("flip a corn", "flip_a_coin")]
    assert again.entries()[0].why == "mishear" and again.entries()[0].turn == "t1"


def test_the_same_phrasing_is_not_added_twice(tmp_path):
    p = Phrasings(tmp_path / "p.json", now=Clock())
    assert p.add("Flip a corn.", "flip_a_coin", why="mishear")
    assert p.add("flip a corn", "flip_a_coin", why="mishear") is None
    assert len(p.entries()) == 1


def test_a_removed_phrasing_is_never_learned_again(tmp_path):
    p = Phrasings(tmp_path / "p.json", now=Clock())
    added = p.add("flip a corn", "flip_a_coin", why="mishear")
    assert p.remove(added.id).text == "flip a corn"
    assert p.examples() == []
    assert p.is_rejected("Flip a corn", "flip_a_coin")
    assert p.add("flip a corn", "flip_a_coin", why="mishear") is None
    # the same words for another label are a different lesson
    assert p.add("flip a corn", "set_timer", why="correction") is not None


def test_each_label_is_capped_and_the_oldest_goes_first(tmp_path):
    clock = Clock()
    p = Phrasings(tmp_path / "p.json", max_per_label=3, now=clock)
    for n in range(5):
        clock.when += dt.timedelta(minutes=1)
        p.add(f"coin {n}", "flip_a_coin", why="unsure")
    p.add("timer 0", "set_timer", why="unsure")
    coin = [t for t, label in p.examples() if label == "flip_a_coin"]
    assert coin == ["coin 2", "coin 3", "coin 4"]
    assert ("timer 0", "set_timer") in p.examples()


def test_pending_until_trained(tmp_path):
    p = Phrasings(tmp_path / "p.json", now=Clock())
    a = p.add("flip a corn", "flip_a_coin", why="mishear")
    b = p.add("toss it", "flip_a_coin", why="unsure")
    assert {x.id for x in p.pending()} == {a.id, b.id}
    p.mark_trained([a.id])
    assert [x.id for x in p.pending()] == [b.id]
    assert [x.id for x in Phrasings(tmp_path / "p.json").pending()] == [b.id]


def test_quarantine_takes_them_out_and_rejects_them(tmp_path):
    p = Phrasings(tmp_path / "p.json", now=Clock())
    a = p.add("flip a corn", "flip_a_coin", why="mishear")
    p.quarantine([a.id])
    assert p.examples() == []
    assert p.is_rejected("flip a corn", "flip_a_coin")


def test_two_writers_on_one_file_lose_nothing(tmp_path):
    """The dev brain and the pod share data/: each add re-reads the file
    under a lock, so one brain's add never overwrites the other's."""
    path = tmp_path / "p.json"
    a, b = Phrasings(path), Phrasings(path)

    def add_many(store, prefix):
        for n in range(20):
            store.add(f"{prefix} {n}", "flip_a_coin", why="unsure")

    threads = [threading.Thread(target=add_many, args=(s, k)) for s, k in ((a, "a"), (b, "b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(Phrasings(path).entries()) == 40


def test_a_corrupt_file_is_set_aside_not_fatal(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json", encoding="utf-8")
    p = Phrasings(path)
    assert p.examples() == []
    assert p.add("flip a corn", "flip_a_coin", why="mishear")
    assert list(tmp_path.glob("p.json.corrupt*"))


# -- the corpus gains a third source ---------------------------------------------

def test_learned_phrasings_are_part_of_the_corpus():
    from jarvis.skills.contract import SkillManifest

    coin = SkillManifest(name="flip_a_coin", description="Flip a coin.", examples=["flip a coin"])
    corpus = build_corpus(
        manifests=[coin],
        learned=[
            ("flip a corn", "flip_a_coin"),
            ("jarvis are you there?", "greeting"),  # already a seed pattern: no second copy
            ("roll it", "roll_a_die"),              # no such skill any more: left out
        ],
    )
    learned = [(e.text, e.label) for e in corpus if e.source == "learned"]
    assert learned == [("flip a corn", "flip_a_coin")]


# -- the learning state ------------------------------------------------------------

def test_builds_are_counted_per_day(tmp_path):
    clock = Clock()
    s = LearningState(tmp_path / "state.json", now=clock)
    s.count_build()
    s.count_build()
    assert s.builds_today() == 2
    clock.when += dt.timedelta(days=1)
    assert LearningState(tmp_path / "state.json", now=clock).builds_today() == 0


def test_repairs_are_counted_per_skill_per_day(tmp_path):
    s = LearningState(tmp_path / "state.json", now=Clock())
    s.count_repair("flip_a_coin")
    assert s.repairs_today("flip_a_coin") == 1
    assert s.repairs_today("roll_a_die") == 0


def test_a_disabled_skill_stays_disabled_until_enabled(tmp_path):
    s = LearningState(tmp_path / "state.json", now=Clock())
    s.disable("flip_a_coin", "kept failing")
    assert LearningState(tmp_path / "state.json").is_disabled("flip_a_coin")
    s.enable("flip_a_coin")
    assert not LearningState(tmp_path / "state.json").is_disabled("flip_a_coin")


def test_events_and_confusions(tmp_path):
    clock = Clock()
    s = LearningState(tmp_path / "state.json", now=clock)
    s.add_event("phrasing", "'flip a corn' means flip_a_coin")
    clock.when += dt.timedelta(days=1)
    s.add_event("skill", "roll_a_die")
    assert [e["kind"] for e in s.events(since=dt.date(2026, 10, 4))] == ["skill"]
    assert s.confusion("flip_a_coin", "set_timer") == 1
    assert s.confusion("flip_a_coin", "set_timer") == 2
    assert LearningState(tmp_path / "state.json").confusions() == {"flip_a_coin -> set_timer": 2}


def test_build_requests_are_remembered_for_the_day(tmp_path):
    clock = Clock()
    s = LearningState(tmp_path / "state.json", now=clock)
    s.add_request("roll a die", "roll a six-sided die", name="roll_a_die", status="building")
    s.set_request_status("roll_a_die", "failed")
    assert [(r["name"], r["status"]) for r in s.requests_today()] == [("roll_a_die", "failed")]
    clock.when += dt.timedelta(days=1)
    assert s.requests_today() == []


def test_a_builtin_failure_is_written_down_for_a_person(tmp_path):
    s = LearningState(tmp_path / "state.json", now=Clock())
    s.builtin_failure("weather", "what's the weather", {}, "KeyError: 'temp'")
    lines = (tmp_path / "builtin-failures.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["skill"] == "weather"


# -- the facade ------------------------------------------------------------------

def test_learning_from_config_lives_under_data_dir(tmp_path, config):
    import dataclasses

    cfg = dataclasses.replace(config, data_dir=tmp_path)
    learning = Learning.from_config(cfg)
    learning.log.append({"device": "local", "heard": "x"})
    learning.phrasings.add("flip a corn", "flip_a_coin", why="mishear")
    learning.state.count_build()
    assert list((tmp_path / "interactions" / "local").glob("*.jsonl"))
    assert (tmp_path / "learning" / "phrasings.json").is_file()
    assert (tmp_path / "learning" / "state.json").is_file()


def test_learning_in_memory_writes_nothing(tmp_path, config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    learning = Learning.in_memory(config)
    learning.log.append({"device": "local", "heard": "x"})
    learning.phrasings.add("flip a corn", "flip_a_coin", why="mishear")
    learning.state.count_build()
    assert learning.state.builds_today() == 1
    assert list(tmp_path.iterdir()) == []
