"""Tests for textstat module."""

import pytest
from textstat import (
    count_words,
    count_chars,
    count_sentences,
    avg_word_length,
    top_words,
    reading_time,
    analyze,
)

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How valiantly did brave Zephyrus blow!"
)


class TestCountWords:
    def test_basic(self):
        assert count_words("hello world foo") == 3

    def test_empty(self):
        assert count_words("") == 0

    def test_extra_whitespace(self):
        assert count_words("  one   two  three  ") == 3

    def test_single_word(self):
        assert count_words("hello") == 1


class TestCountChars:
    def test_with_spaces(self):
        assert count_chars("abc de") == 6

    def test_without_spaces(self):
        assert count_chars("abc de", include_spaces=False) == 5

    def test_empty(self):
        assert count_chars("") == 0


class TestCountSentences:
    def test_periods(self):
        assert count_sentences("One. Two. Three.") == 3

    def test_mixed_punctuation(self):
        assert count_sentences("Hello! How are you? I am fine.") == 3

    def test_no_punctuation(self):
        assert count_sentences("just a plain line") == 1

    def test_empty(self):
        assert count_sentences("") == 0

    def test_sample(self):
        assert count_sentences(SAMPLE) == 3


class TestAvgWordLength:
    def test_uniform(self):
        # "cat dog pig" — all 3-letter words
        assert avg_word_length("cat dog pig") == 3.0

    def test_mixed(self):
        result = avg_word_length("hi hello")
        assert abs(result - 3.5) < 0.01

    def test_empty(self):
        assert avg_word_length("") == 0.0

    def test_ignores_punctuation(self):
        # punctuation stripped; only letters counted
        result = avg_word_length("hi, hello!")
        assert abs(result - 3.5) < 0.01


class TestTopWords:
    def test_returns_most_common(self):
        text = "cat cat cat dog dog bird"
        result = top_words(text, n=2)
        assert result[0] == ("cat", 3)
        assert result[1] == ("dog", 2)

    def test_stopwords_excluded(self):
        text = "the the the cat cat"
        result = top_words(text, n=5)
        words = [w for w, _ in result]
        assert "the" not in words
        assert "cat" in words

    def test_n_respected(self):
        text = "alpha beta gamma delta epsilon"
        result = top_words(text, n=3)
        assert len(result) <= 3

    def test_empty(self):
        assert top_words("") == []


class TestReadingTime:
    def test_zero(self):
        assert reading_time("") == 0.0

    def test_200_words(self):
        text = " ".join(["word"] * 200)
        assert reading_time(text, wpm=200) == 1.0

    def test_custom_wpm(self):
        text = " ".join(["word"] * 100)
        assert reading_time(text, wpm=50) == 2.0


class TestAnalyze:
    def test_returns_all_keys(self):
        result = analyze(SAMPLE)
        expected_keys = {
            "words", "chars", "chars_no_spaces",
            "sentences", "avg_word_length",
            "reading_time_min", "top_words",
        }
        assert expected_keys == set(result.keys())

    def test_word_count_correct(self):
        result = analyze("one two three four five")
        assert result["words"] == 5

    def test_sentence_count_correct(self):
        result = analyze("Hello. World!")
        assert result["sentences"] == 2

    def test_top_words_is_list(self):
        result = analyze(SAMPLE)
        assert isinstance(result["top_words"], list)