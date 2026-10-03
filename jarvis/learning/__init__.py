"""M7: learning from every turn (``PRD/milestone-7-learn-from-every-turn.md``).

Three stores, one facade:

  ``log``        every turn, ``<data>/interactions/<device>/<day>.jsonl``
  ``phrasings``  words learned from confirmed turns, the corpus's third source
  ``state``      daily caps, disabled skills, events, confusions

The orchestrator decides *what* is learned; this package only keeps it.
"""

from __future__ import annotations

from jarvis.learning.interactions import InteractionLog
from jarvis.learning.phrasings import Phrasing, Phrasings
from jarvis.learning.state import LearningState

__all__ = ["InteractionLog", "Learning", "LearningState", "Phrasing", "Phrasings"]


class Learning:
    def __init__(self, settings, log: InteractionLog, phrasings: Phrasings, state: LearningState):
        #: ``config.learning``
        self.settings = settings
        self.log = log
        self.phrasings = phrasings
        self.state = state

    @classmethod
    def from_config(cls, config) -> "Learning":
        settings = config.learning
        directory = config.interactions_dir if settings.log else None
        return cls(
            settings,
            InteractionLog(directory, keep_days=settings.keep_days),
            Phrasings(
                config.learning_dir / "phrasings.json", max_per_label=settings.max_per_label
            ),
            LearningState(config.learning_dir / "state.json"),
        )

    @classmethod
    def in_memory(cls, config) -> "Learning":
        """Nothing on disk: tests, and an orchestrator built without one."""
        settings = config.learning
        return cls(
            settings,
            InteractionLog(None, keep_days=settings.keep_days),
            Phrasings(None, max_per_label=settings.max_per_label),
            LearningState(None),
        )


def learned_examples(config) -> list[tuple[str, str]]:
    """What the corpus trains on from ``<data>/learning/phrasings.json``."""
    return Phrasings(config.learning_dir / "phrasings.json").examples()
