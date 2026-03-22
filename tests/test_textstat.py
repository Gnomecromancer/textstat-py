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
    count_syllables,
    total_syllables,
    flesch_reading_ease,
    flesch_kincaid_grade,
    lexical_diversity,
    count_complex_words,
    gunning_fog,
    sentence_lengths,
    sentence_stats,
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


class TestCountSyllables:
    def test_monosyllabic(self):
        assert count_syllables("cat") == 1
        assert count_syllables("dog") == 1
        assert count_syllables("fox") == 1

    def test_bisyllabic(self):
        assert count_syllables("hello") == 2
        assert count_syllables("garden") == 2

    def test_silent_e(self):
        # "make" has 2 vowel groups (a, e) but silent trailing e → 1 syllable
        assert count_syllables("make") == 1
        assert count_syllables("time") == 1

    def test_polysyllabic(self):
        assert count_syllables("beautiful") >= 3
        assert count_syllables("education") >= 4

    def test_empty(self):
        assert count_syllables("") == 0

    def test_minimum_one(self):
        # Any real word has at least 1 syllable
        assert count_syllables("rhythm") >= 1


class TestTotalSyllables:
    def test_simple(self):
        # "cat dog" = 1 + 1 = 2
        assert total_syllables("cat dog") == 2

    def test_empty(self):
        assert total_syllables("") == 0

    def test_ignores_punctuation(self):
        assert total_syllables("cat, dog.") == total_syllables("cat dog")


class TestFleschReadingEase:
    def test_empty(self):
        assert flesch_reading_ease("") == 0.0

    def test_returns_float(self):
        result = flesch_reading_ease(SAMPLE)
        assert isinstance(result, float)

    def test_simple_text_scores_higher(self):
        simple = "The cat sat. The dog ran. I am here."
        complex_text = (
            "The administration's unprecedented infrastructural deterioration "
            "necessitates comprehensive rehabilitation."
        )
        assert flesch_reading_ease(simple) > flesch_reading_ease(complex_text)

    def test_reasonable_range(self):
        # Standard English prose generally falls in 0–100
        result = flesch_reading_ease(SAMPLE)
        assert -20 <= result <= 120  # allow slight overshoot from heuristic


class TestFleschKincaidGrade:
    def test_empty(self):
        assert flesch_kincaid_grade("") == 0.0

    def test_returns_float(self):
        result = flesch_kincaid_grade(SAMPLE)
        assert isinstance(result, float)

    def test_simple_text_lower_grade(self):
        simple = "The cat sat. The dog ran. I am here."
        hard = (
            "The administration's unprecedented infrastructural deterioration "
            "necessitates comprehensive rehabilitation."
        )
        assert flesch_kincaid_grade(simple) < flesch_kincaid_grade(hard)

    def test_reasonable_range(self):
        result = flesch_kincaid_grade(SAMPLE)
        assert -5 <= result <= 25


class TestLexicalDiversity:
    def test_empty(self):
        assert lexical_diversity("") == 0.0

    def test_all_unique(self):
        # Every word is different → ratio = 1.0
        assert lexical_diversity("alpha beta gamma delta") == 1.0

    def test_all_same(self):
        # Same word repeated → ratio = 1/n (rounds to low value)
        result = lexical_diversity("cat cat cat cat")
        assert result == 0.25

    def test_range(self):
        result = lexical_diversity(SAMPLE)
        assert 0.0 <= result <= 1.0

    def test_case_insensitive(self):
        # "Cat" and "cat" should be treated as the same word
        assert lexical_diversity("Cat cat CAT") == pytest.approx(1 / 3, abs=0.01)


class TestCountComplexWords:
    def test_empty(self):
        assert count_complex_words("") == 0

    def test_no_complex(self):
        # all monosyllabic
        assert count_complex_words("cat dog fox ran") == 0

    def test_counts_polysyllabic(self):
        # "beautiful" (3 syl), "education" (4 syl)
        result = count_complex_words("beautiful education cat")
        assert result == 2

    def test_ignores_punctuation(self):
        assert count_complex_words("beautiful, education.") == count_complex_words("beautiful education")


class TestGunningFog:
    def test_empty(self):
        assert gunning_fog("") == 0.0

    def test_returns_float(self):
        result = gunning_fog(SAMPLE)
        assert isinstance(result, float)

    def test_complex_text_higher_fog(self):
        simple = "The cat sat. The dog ran. I am here."
        hard = (
            "Unprecedented infrastructural deterioration necessitates "
            "comprehensive rehabilitation immediately."
        )
        assert gunning_fog(hard) > gunning_fog(simple)

    def test_reasonable_range(self):
        # Typical prose: 6–17
        result = gunning_fog(SAMPLE)
        assert 0 <= result <= 30

    def test_formula_correctness(self):
        # One sentence, four words, one complex (3+ syl): "cat dog fox beautiful"
        # words/sentences = 4/1 = 4; complex/words = 1/4 = 0.25
        # fog = 0.4 * (4 + 100*0.25) = 0.4 * 29 = 11.6
        text = "cat dog fox beautiful"
        result = gunning_fog(text)
        assert abs(result - 11.6) < 0.5  # allow syllable heuristic variance


class TestSentenceLengths:
    def test_empty(self):
        assert sentence_lengths("") == []

    def test_single_sentence(self):
        assert sentence_lengths("one two three.") == [3]

    def test_multiple_sentences(self):
        lengths = sentence_lengths("one two. three four five. six.")
        assert lengths == [2, 3, 1]

    def test_mixed_punctuation(self):
        lengths = sentence_lengths("Hello! How are you? Fine.")
        assert len(lengths) == 3

    def test_no_punctuation(self):
        # Treated as one sentence
        assert sentence_lengths("one two three four") == [4]


class TestSentenceStats:
    def test_empty(self):
        result = sentence_stats("")
        assert result == {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}

    def test_single_sentence(self):
        result = sentence_stats("one two three.")
        assert result["min"] == 3
        assert result["max"] == 3
        assert result["mean"] == 3.0
        assert result["median"] == 3.0

    def test_two_sentences(self):
        # "one two." (2 words) and "three four five." (3 words)
        result = sentence_stats("one two. three four five.")
        assert result["min"] == 2
        assert result["max"] == 3
        assert result["mean"] == 2.5
        assert result["median"] == 2.5

    def test_varied_lengths(self):
        # 1, 3, 5 words → mean=3, median=3
        result = sentence_stats("go. one two three. alpha beta gamma delta epsilon.")
        assert result["min"] == 1
        assert result["max"] == 5
        assert result["mean"] == 3.0
        assert result["median"] == 3.0

    def test_returns_all_keys(self):
        result = sentence_stats(SAMPLE)
        assert set(result.keys()) == {"min", "max", "mean", "median"}


class TestAnalyze:
    def test_returns_all_keys(self):
        result = analyze(SAMPLE)
        expected_keys = {
            "words", "chars", "chars_no_spaces",
            "sentences", "avg_word_length",
            "reading_time_min", "top_words",
            "flesch_reading_ease", "flesch_kincaid_grade",
            "lexical_diversity", "gunning_fog", "sentence_stats",
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

    def test_readability_scores_present(self):
        result = analyze(SAMPLE)
        assert isinstance(result["flesch_reading_ease"], float)
        assert isinstance(result["flesch_kincaid_grade"], float)
        assert isinstance(result["lexical_diversity"], float)

    def test_gunning_fog_present(self):
        result = analyze(SAMPLE)
        assert isinstance(result["gunning_fog"], float)

    def test_sentence_stats_present(self):
        result = analyze(SAMPLE)
        ss = result["sentence_stats"]
        assert isinstance(ss, dict)
        assert "min" in ss and "max" in ss and "mean" in ss and "median" in ss
