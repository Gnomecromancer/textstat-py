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
    coleman_liau_index,
    automated_readability_index,
    grade_level_consensus,
    smog_index,
    hapax_legomena_ratio,
    count_paragraphs,
    paragraph_stats,
)

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How valiantly did brave Zephyrus blow!"
)

MULTI_PARA = (
    "The cat sat on the mat. The dog barked loudly.\n\n"
    "A quick brown fox jumps over the lazy dog. Pack my box with five dozen jugs.\n\n"
    "Simple short sentence."
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
        assert avg_word_length("cat dog pig") == 3.0

    def test_mixed(self):
        result = avg_word_length("hi hello")
        assert abs(result - 3.5) < 0.01

    def test_empty(self):
        assert avg_word_length("") == 0.0

    def test_ignores_punctuation(self):
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
        assert count_syllables("make") == 1
        assert count_syllables("time") == 1

    def test_polysyllabic(self):
        assert count_syllables("beautiful") >= 3
        assert count_syllables("education") >= 4

    def test_empty(self):
        assert count_syllables("") == 0

    def test_minimum_one(self):
        assert count_syllables("rhythm") >= 1


class TestTotalSyllables:
    def test_simple(self):
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
        result = flesch_reading_ease(SAMPLE)
        assert -20 <= result <= 120


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
        assert lexical_diversity("alpha beta gamma delta") == 1.0

    def test_all_same(self):
        result = lexical_diversity("cat cat cat cat")
        assert result == 0.25

    def test_range(self):
        result = lexical_diversity(SAMPLE)
        assert 0.0 <= result <= 1.0

    def test_case_insensitive(self):
        assert lexical_diversity("Cat cat CAT") == pytest.approx(1 / 3, abs=0.01)


class TestCountComplexWords:
    def test_empty(self):
        assert count_complex_words("") == 0

    def test_no_complex(self):
        assert count_complex_words("cat dog fox ran") == 0

    def test_counts_polysyllabic(self):
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
        result = gunning_fog(SAMPLE)
        assert 0 <= result <= 30

    def test_formula_correctness(self):
        text = "cat dog fox beautiful"
        result = gunning_fog(text)
        assert abs(result - 11.6) < 0.5


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
        result = sentence_stats("one two. three four five.")
        assert result["min"] == 2
        assert result["max"] == 3
        assert result["mean"] == 2.5
        assert result["median"] == 2.5

    def test_varied_lengths(self):
        result = sentence_stats("go. one two three. alpha beta gamma delta epsilon.")
        assert result["min"] == 1
        assert result["max"] == 5
        assert result["mean"] == 3.0
        assert result["median"] == 3.0

    def test_returns_all_keys(self):
        result = sentence_stats(SAMPLE)
        assert set(result.keys()) == {"min", "max", "mean", "median"}


class TestColemanLiauIndex:
    def test_empty(self):
        assert coleman_liau_index("") == 0.0

    def test_returns_float(self):
        result = coleman_liau_index(SAMPLE)
        assert isinstance(result, float)

    def test_complex_text_higher_grade(self):
        simple = "The cat sat. The dog ran. I am here."
        hard = (
            "The administration's unprecedented infrastructural deterioration "
            "necessitates comprehensive rehabilitation assessment."
        )
        assert coleman_liau_index(hard) > coleman_liau_index(simple)

    def test_reasonable_range(self):
        result = coleman_liau_index(SAMPLE)
        assert -5 <= result <= 25

    def test_formula_driven(self):
        short_words = "I go do it so am up."
        long_words = "Extraordinary administrative responsibilities necessitate comprehensive understanding."
        assert coleman_liau_index(long_words) > coleman_liau_index(short_words)


class TestAutomatedReadabilityIndex:
    def test_empty(self):
        assert automated_readability_index("") == 0.0

    def test_returns_float(self):
        result = automated_readability_index(SAMPLE)
        assert isinstance(result, float)

    def test_complex_text_higher_ari(self):
        simple = "The cat sat. The dog ran. I am here."
        hard = (
            "Unprecedented infrastructural deterioration necessitates "
            "comprehensive rehabilitation immediately."
        )
        assert automated_readability_index(hard) > automated_readability_index(simple)

    def test_reasonable_range(self):
        result = automated_readability_index(SAMPLE)
        assert -5 <= result <= 30

    def test_single_sentence_no_crash(self):
        result = automated_readability_index("Hello world today.")
        assert isinstance(result, float)


class TestGradeLevelConsensus:
    def test_empty(self):
        assert grade_level_consensus("") == 0.0

    def test_whitespace_only(self):
        assert grade_level_consensus("   ") == 0.0

    def test_returns_float(self):
        result = grade_level_consensus(SAMPLE)
        assert isinstance(result, float)

    def test_complex_text_higher_consensus(self):
        simple = "The cat sat. The dog ran. I am here."
        hard = (
            "Unprecedented infrastructural deterioration necessitates "
            "comprehensive rehabilitation immediately."
        )
        assert grade_level_consensus(hard) > grade_level_consensus(simple)

    def test_reasonable_range(self):
        result = grade_level_consensus(SAMPLE)
        assert -5 <= result <= 25

    def test_is_average_of_four_formulas(self):
        text = SAMPLE
        expected = round(
            (
                flesch_kincaid_grade(text)
                + gunning_fog(text)
                + coleman_liau_index(text)
                + automated_readability_index(text)
            )
            / 4,
            1,
        )
        assert grade_level_consensus(text) == expected


class TestSmogIndex:
    def test_empty(self):
        assert smog_index("") == 0.0

    def test_fewer_than_3_sentences_returns_zero(self):
        assert smog_index("One sentence only.") == 0.0
        assert smog_index("One. Two.") == 0.0

    def test_three_sentences_ok(self):
        result = smog_index(SAMPLE)
        assert isinstance(result, float)
        assert result > 0.0

    def test_complex_text_higher_smog(self):
        # Build 3+ sentence texts so SMOG is valid
        simple = "The cat sat. The dog ran. I am here. Go home now."
        hard = (
            "Unprecedented infrastructural deterioration necessitates comprehensive rehabilitation. "
            "Administrative responsibilities necessitate extraordinary understanding. "
            "Technological advancements revolutionize contemporary civilization. "
            "Extraordinary complications necessitate sophisticated interventions."
        )
        assert smog_index(hard) > smog_index(simple)

    def test_no_polysyllabic_words(self):
        # All monosyllabic: SMOG = 3 + sqrt(0) = 3.0
        text = "Cat sat. Dog ran. Fox hid. Pig ate. Hen flew."
        result = smog_index(text)
        assert result == pytest.approx(3.0, abs=0.1)

    def test_formula_correctness(self):
        # 3 sentences, exactly 3 complex words → poly * (30/3) = 3 * 10 = 30 → sqrt(30) ≈ 5.477
        # smog = 3 + 5.477 ≈ 8.48
        text = (
            "Cat sat on mat. "
            "Dog ran far away. "
            "Beautiful education necessitates dedication."
        )
        result = smog_index(text)
        # Allow variance due to syllable heuristic
        assert 6.0 <= result <= 12.0

    def test_returns_float(self):
        result = smog_index(SAMPLE)
        assert isinstance(result, float)


class TestHapaxLegomenaRatio:
    def test_empty(self):
        assert hapax_legomena_ratio("") == 0.0

    def test_all_unique_words(self):
        # Every word appears once → ratio = total_hapax / total_words = 4/4 = 1.0
        result = hapax_legomena_ratio("alpha beta gamma delta")
        assert result == 1.0

    def test_all_repeated(self):
        # "cat cat cat" — no hapax → ratio = 0.0
        result = hapax_legomena_ratio("cat cat cat")
        assert result == 0.0

    def test_mixed(self):
        # "cat cat dog" — "dog" is hapax (1 out of 3 tokens)
        result = hapax_legomena_ratio("cat cat dog")
        assert result == pytest.approx(1 / 3, abs=0.01)

    def test_case_insensitive(self):
        # "Cat" and "cat" are the same word → not a hapax
        result = hapax_legomena_ratio("Cat cat dog")
        assert result == pytest.approx(1 / 3, abs=0.01)

    def test_range(self):
        result = hapax_legomena_ratio(SAMPLE)
        assert 0.0 <= result <= 1.0

    def test_high_diversity_text(self):
        # SAMPLE has mostly unique words → ratio should be high
        result = hapax_legomena_ratio(SAMPLE)
        assert result > 0.5

    def test_returns_float(self):
        assert isinstance(hapax_legomena_ratio(SAMPLE), float)


class TestCountParagraphs:
    def test_empty(self):
        assert count_paragraphs("") == 0

    def test_single_block(self):
        assert count_paragraphs("One sentence. Two sentences.") == 1

    def test_two_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph."
        assert count_paragraphs(text) == 2

    def test_three_paragraphs(self):
        assert count_paragraphs(MULTI_PARA) == 3

    def test_whitespace_only_lines_ignored(self):
        text = "Para one.\n\n   \n\nPara two."
        assert count_paragraphs(text) == 2

    def test_multiple_blank_lines(self):
        text = "First.\n\n\n\nSecond."
        assert count_paragraphs(text) == 2


class TestParagraphStats:
    def test_empty(self):
        result = paragraph_stats("")
        assert result == {"count": 0, "min": 0, "max": 0, "mean": 0.0}

    def test_single_paragraph(self):
        result = paragraph_stats("one two three four five")
        assert result["count"] == 1
        assert result["min"] == 5
        assert result["max"] == 5
        assert result["mean"] == 5.0

    def test_two_paragraphs(self):
        text = "one two three.\n\none two three four five six."
        result = paragraph_stats(text)
        assert result["count"] == 2
        assert result["min"] == 3
        assert result["max"] == 6
        assert result["mean"] == 4.5

    def test_three_paragraphs(self):
        result = paragraph_stats(MULTI_PARA)
        assert result["count"] == 3
        assert result["min"] >= 1
        assert result["max"] >= result["min"]

    def test_returns_all_keys(self):
        result = paragraph_stats(SAMPLE)
        assert set(result.keys()) == {"count", "min", "max", "mean"}

    def test_mean_is_average(self):
        # Two paragraphs: 2 words and 4 words → mean = 3.0
        text = "one two.\n\nthree four five six."
        result = paragraph_stats(text)
        assert result["mean"] == 3.0


class TestAnalyze:
    def test_returns_all_keys(self):
        result = analyze(SAMPLE)
        expected_keys = {
            "words", "chars", "chars_no_spaces",
            "sentences", "avg_word_length",
            "reading_time_min", "top_words",
            "flesch_reading_ease", "flesch_kincaid_grade",
            "lexical_diversity", "gunning_fog", "sentence_stats",
            "coleman_liau", "automated_readability_index", "grade_level_consensus",
            "smog_index", "hapax_legomena_ratio", "paragraph_stats",
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

    def test_new_readability_metrics_present(self):
        result = analyze(SAMPLE)
        assert isinstance(result["coleman_liau"], float)
        assert isinstance(result["automated_readability_index"], float)
        assert isinstance(result["grade_level_consensus"], float)

    def test_smog_present(self):
        result = analyze(SAMPLE)
        assert isinstance(result["smog_index"], float)

    def test_hapax_legomena_present(self):
        result = analyze(SAMPLE)
        assert isinstance(result["hapax_legomena_ratio"], float)
        assert 0.0 <= result["hapax_legomena_ratio"] <= 1.0

    def test_paragraph_stats_present(self):
        result = analyze(SAMPLE)
        ps = result["paragraph_stats"]
        assert isinstance(ps, dict)
        assert set(ps.keys()) == {"count", "min", "max", "mean"}