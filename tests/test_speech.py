"""Making an answer speakable (M3 decision 8).

Text written to be read has things in it that are unbearable said out loud: a
fenced code block, a forty-character path, a URL with a query string. The voder
gets the spoken version of the answer; the log keeps the whole of it.
"""

from __future__ import annotations

import pytest

from jarvis.core.speech import MAX_SPOKEN_CHARS, speakable, split_sentences


# -- markdown --------------------------------------------------------------


def test_markdown_emphasis_and_headings_are_stripped():
    assert speakable("**Done**, sir.") == "Done, sir."
    assert speakable("## The result\nIt is *forty-two*.") == "The result. It is forty-two."
    assert speakable("`config.toml` is the file") == "config.toml is the file"
    assert speakable("~~never mind~~ it worked") == "it worked"


def test_a_list_is_read_as_sentences():
    out = speakable("- first thing\n- second thing\n- third thing")
    assert "first thing" in out and "third thing" in out
    assert "-" not in out


def test_a_link_is_read_as_its_words_not_its_url():
    assert speakable("see [the docs](https://example.com/a/b?c=1) for more") == (
        "see the docs for more"
    )


# -- the things that are unbearable out loud -------------------------------


def test_a_code_block_becomes_one_phrase_however_many_there_are():
    out = speakable(
        "Here you go:\n```python\ndef f(x):\n    return x + 1\n```\nand that's it."
    )
    assert "def f" not in out
    assert out.count("code on the screen") == 1
    assert "Here you go" in out and "that's it" in out


def test_several_code_blocks_are_still_mentioned_once():
    out = speakable("one\n```\na\n```\ntwo\n```\nb\n```\nthree")
    assert out.count("code on the screen") == 1


def test_a_path_is_shortened_to_its_last_part():
    assert speakable("I wrote /home/robin/kevin/JARVIS/jarvis/core/speech.py") == (
        "I wrote speech.py"
    )
    assert speakable("check data/models/piper/en_GB-alan-medium.onnx") == (
        "check en_GB-alan-medium.onnx"
    )


def test_a_url_is_shortened_to_its_host():
    assert speakable("it's at https://github.com/anthropics/claude-code/issues/42") == (
        "it's at github.com"
    )
    assert speakable("see http://localhost:11434/api/tags") == "see localhost"


def test_a_bare_word_with_a_slash_in_it_is_left_alone():
    # not every slash is a path — "and/or" is a word
    assert speakable("and/or something") == "and/or something"


# -- length ----------------------------------------------------------------


def test_a_long_answer_is_cut_at_a_sentence_and_says_where_the_rest_is():
    long = " ".join(f"This is sentence number {i}." for i in range(1, 80))
    out = speakable(long)
    assert len(out) <= MAX_SPOKEN_CHARS + len(" The rest is in the log.")
    assert out.endswith("The rest is in the log.")
    # cut at a sentence end, not mid-word
    body = out[: -len(" The rest is in the log.")]
    assert body.endswith(".")


def test_a_short_answer_is_left_exactly_as_it_is():
    assert speakable("Of course, sir.") == "Of course, sir."
    assert "the log" not in speakable("Of course, sir.")


def test_an_answer_with_no_sentence_end_is_still_cut_somewhere_sane():
    out = speakable("word " * 400)
    assert len(out) <= MAX_SPOKEN_CHARS + len(" The rest is in the log.")
    assert "word word" in out


def test_empty_stays_empty():
    assert speakable("") == ""
    assert speakable(None) == ""
    assert speakable("   \n\n  ") == ""


# -- sentence splitting ----------------------------------------------------


def test_the_answer_is_split_so_the_first_sentence_can_be_spoken_at_once():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_an_abbreviation_does_not_end_a_sentence():
    assert split_sentences("It is 3.5 metres. That is all.") == [
        "It is 3.5 metres.", "That is all.",
    ]
    assert split_sentences("Mr. Stark is out.") == ["Mr. Stark is out."]


def test_a_single_sentence_is_one_part():
    assert split_sentences("Of course, sir") == ["Of course, sir"]
    assert split_sentences("") == []


def test_a_very_long_sentence_is_broken_up_so_speech_starts_sooner():
    one = "and then " * 60  # no sentence end at all
    parts = split_sentences(one.strip())
    assert len(parts) > 1
    assert all(len(p) <= 240 for p in parts)
    assert "".join(p if p.endswith(" ") else p + " " for p in parts).split() == one.split()


@pytest.mark.parametrize("text", ["One. Two.", "A single one", "x" * 500])
def test_splitting_never_loses_a_word(text):
    assert " ".join(split_sentences(text)).split() == text.split()
