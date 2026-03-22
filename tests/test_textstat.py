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
    sentiment_polarity,
    sentiment_label,
    text_density,
    word_frequency_distribution,
    herdan_c,
    yule_k,
    mattr,
    vocabulary_richness,
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
        simple = "The cat sat. The dog ran. I am here. Go home now."
        hard = (
            "Unprecedented infrastructural deterioration necessitates comprehensive rehabilitation. "
            "Administrative responsibilities necessitate extraordinary understanding. "
            "Technological advancements revolutionize contemporary civilization. "
            "Extraordinary complications necessitate sophisticated interventions."
        )
        assert smog_index(hard) > smog_index(simple)

    def test_no_polysyllabic_words(self):
        text = "Cat sat. Dog ran. Fox hid. Pig ate. Hen flew."
        result = smog_index(text)
        assert result == pytest.approx(3.0, abs=0.1)

    def test_formula_correctness(self):
        text = (
            "Cat sat on mat. "
            "Dog ran far away. "
            "Beautiful education necessitates dedication."
        )
        result = smog_index(text)
        assert 6.0 <= result <= 12.0

    def test_returns_float(self):
        result = smog_index(SAMPLE)
        assert isinstance(result, float)


class TestHapaxLegomenaRatio:
    def test_empty(self):
        assert hapax_legomena_ratio("") == 0.0

    def test_all_unique_words(self):
        result = hapax_legomena_ratio("alpha beta gamma delta")
        assert result == 1.0

    def test_all_repeated(self):
        result = hapax_legomena_ratio("cat cat cat")
        assert result == 0.0

    def test_mixed(self):
        result = hapax_legomena_ratio("cat cat dog")
        assert result == pytest.approx(1 / 3, abs=0.01)

    def test_case_insensitive(self):
        result = hapax_legomena_ratio("Cat cat dog")
        assert result == pytest.approx(1 / 3, abs=0.01)

    def test_range(self):
        result = hapax_legomena_ratio(SAMPLE)
        assert 0.0 <= result <= 1.0

    def test_high_diversity_text(self):
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
        text = "one two.\n\nthree four five six."
        result = paragraph_stats(text)
        assert result["mean"] == 3.0


class TestSentimentPolarity:
    def test_empty(self):
        assert sentiment_polarity("") == 0.0

    def test_positive_text(self):
        result = sentiment_polarity("This is great and wonderful and amazing!")
        assert result > 0.0

    def test_negative_text(self):
        result = sentiment_polarity("This is terrible and awful and horrible.")
        assert result < 0.0

    def test_neutral_text(self):
        result = sentiment_polarity("The fox jumped over the log on Tuesday.")
        assert result == 0.0

    def test_negation_flips_positive(self):
        pos = sentiment_polarity("This is good.")
        neg = sentiment_polarity("This is not good.")
        assert pos > 0.0
        assert neg < 0.0

    def test_negation_flips_negative(self):
        bad = sentiment_polarity("This is terrible.")
        not_bad = sentiment_polarity("This is not terrible.")
        assert bad < 0.0
        assert not_bad > 0.0

    def test_range(self):
        result = sentiment_polarity(SAMPLE)
        assert -1.0 <= result <= 1.0

    def test_returns_float(self):
        assert isinstance(sentiment_polarity(SAMPLE), float)

    def test_mixed_sentiment_near_zero(self):
        text = "good bad good bad good bad"
        result = sentiment_polarity(text)
        assert abs(result) < 0.1


class TestSentimentLabel:
    def test_positive(self):
        assert sentiment_label("wonderful amazing great fantastic love") == "positive"

    def test_negative(self):
        assert sentiment_label("terrible awful horrible hate bad") == "negative"

    def test_neutral(self):
        assert sentiment_label("the fox sat on a log") == "neutral"

    def test_empty(self):
        assert sentiment_label("") == "neutral"

    def test_returns_string(self):
        assert isinstance(sentiment_label(SAMPLE), str)

    def test_valid_labels(self):
        for text in [SAMPLE, "good", "bad", "the cat"]:
            assert sentiment_label(text) in {"positive", "negative", "neutral"}


class TestTextDensity:
    def test_empty(self):
        assert text_density("") == 0.0

    def test_all_stopwords(self):
        result = text_density("the a an and or but")
        assert result == 0.0

    def test_no_stopwords(self):
        result = text_density("cat dog fox runs jumps")
        assert result == 1.0

    def test_mixed(self):
        # "the cat sat" — "the" is stopword, "cat" and "sat" are content
        result = text_density("the cat sat")
        assert result == pytest.approx(2 / 3, abs=0.01)

    def test_range(self):
        result = text_density(SAMPLE)
        assert 0.0 <= result <= 1.0

    def test_content_rich_text_higher_density(self):
        stopword_heavy = "the the the and and or or but but"
        content_rich = "fox jumps runs leaps bounds sprints"
        assert text_density(content_rich) > text_density(stopword_heavy)


class TestWordFrequencyDistribution:
    def test_empty(self):
        result = word_frequency_distribution("")
        assert result == {"total_tokens": 0, "unique_types": 0, "top_10_pct": 0.0, "zipf_fit": 0.0}

    def test_returns_all_keys(self):
        result = word_frequency_distribution(SAMPLE)
        assert set(result.keys()) == {"total_tokens", "unique_types", "top_10_pct", "zipf_fit"}

    def test_total_tokens_correct(self):
        result = word_frequency_distribution("cat dog fox cat")
        assert result["total_tokens"] == 4

    def test_unique_types_correct(self):
        result = word_frequency_distribution("cat dog fox cat")
        assert result["unique_types"] == 3

    def test_top_10_pct_range(self):
        result = word_frequency_distribution(SAMPLE)
        assert 0.0 <= result["top_10_pct"] <= 1.0

    def test_top_10_pct_all_unique(self):
        result = word_frequency_distribution("alpha beta gamma delta epsilon")
        assert result["top_10_pct"] == 1.0

    def test_few_words_zipf_zero(self):
        result = word_frequency_distribution("cat dog cat")
        assert result["zipf_fit"] == 0.0

    def test_zipf_fit_range(self):
        result = word_frequency_distribution(SAMPLE)
        assert -1.0 <= result["zipf_fit"] <= 1.0

    def test_zipf_fit_negative_for_natural_text(self):
        long_text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How valiantly did brave Zephyrus blow the horn today. "
            "The cat sat on the mat and watched the sun set slowly. "
            "Every good boy does fine and every fine boy does good deeds. "
        )
        result = word_frequency_distribution(long_text)
        assert result["zipf_fit"] < 0.0


class TestHerdanC:
    def test_empty(self):
        assert herdan_c("") == 0.0

    def test_single_token(self):
        assert herdan_c("cat") == 0.0

    def test_all_same(self):
        # all same word → V=1, log(1)=0 → returns 0.0
        assert herdan_c("cat cat cat cat cat") == 0.0

    def test_all_unique(self):
        # all unique → V=N → log(V)/log(N) = 1.0
        result = herdan_c("alpha beta gamma delta epsilon")
        assert result == pytest.approx(1.0, abs=0.001)

    def test_range(self):
        result = herdan_c(SAMPLE)
        assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        assert isinstance(herdan_c(SAMPLE), float)

    def test_richer_text_higher_c(self):
        # text with more unique words per token should have higher Herdan's C
        repeated = "cat cat cat cat cat dog dog dog dog dog"
        diverse = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        assert herdan_c(diverse) > herdan_c(repeated)

    def test_longer_repeated_text(self):
        # repeated tokens should give low C
        text = " ".join(["word"] * 50)
        result = herdan_c(text)
        assert result < 0.3


class TestYuleK:
    def test_empty(self):
        assert yule_k("") == 0.0

    def test_all_same_word(self):
        # all tokens same → V(m)=1 type with freq=N → K = 10^4*(N^2 - N)/N^2
        # as N grows this approaches 10000; for small N it's smaller but positive
        result = yule_k("cat cat cat cat cat")
        assert result > 0.0

    def test_all_unique(self):
        # all unique → each type appears once → V(1)=N, all m=1
        # numerator = 1*1*N - N = 0 → K = 0.0
        result = yule_k("alpha beta gamma delta epsilon")
        assert result == 0.0

    def test_returns_float(self):
        assert isinstance(yule_k(SAMPLE), float)

    def test_repeated_text_higher_k(self):
        # more repetition → higher K
        repeated = " ".join(["cat"] * 20 + ["dog"] * 20)
        diverse = " ".join([f"word{i}" for i in range(40)])
        assert yule_k(repeated) > yule_k(diverse)

    def test_non_negative(self):
        # K is non-negative for any real text
        for text in [SAMPLE, "cat dog", "the the the cat"]:
            assert yule_k(text) >= 0.0

    def test_single_word(self):
        result = yule_k("hello")
        assert result == 0.0


class TestMattr:
    def test_empty(self):
        assert mattr("") == 0.0

    def test_all_same(self):
        result = mattr("cat cat cat cat cat", window=3)
        assert result == pytest.approx(1 / 3, abs=0.01)

    def test_all_unique_short(self):
        # text shorter than window → uses full TTR
        result = mattr("alpha beta gamma delta", window=100)
        assert result == 1.0

    def test_all_unique_equals_window(self):
        # exactly window unique words → TTR=1.0 for that window
        words = " ".join([f"w{i}" for i in range(10)])
        result = mattr(words, window=10)
        assert result == 1.0

    def test_range(self):
        result = mattr(SAMPLE)
        assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        assert isinstance(mattr(SAMPLE), float)

    def test_window_sliding(self):
        # 6 words: a b c a b c — window=3 → TTR for [a,b,c]=1.0, [b,c,a]=1.0, [c,a,b]=1.0, [a,b,c]=1.0
        result = mattr("a b c a b c", window=3)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_repeated_reduces_mattr(self):
        diverse = " ".join([f"word{i}" for i in range(200)])
        repeated = " ".join(["cat", "dog"] * 100)
        assert mattr(diverse, window=50) > mattr(repeated, window=50)


class TestVocabularyRichness:
    def test_empty(self):
        result = vocabulary_richness("")
        assert result == {"ttr": 0.0, "herdan_c": 0.0, "yule_k": 0.0, "mattr": 0.0}

    def test_returns_all_keys(self):
        result = vocabulary_richness(SAMPLE)
        assert set(result.keys()) == {"ttr", "herdan_c", "yule_k", "mattr"}

    def test_ttr_matches_lexical_diversity(self):
        from textstat import lexical_diversity
        text = SAMPLE
        assert vocabulary_richness(text)["ttr"] == lexical_diversity(text)

    def test_herdan_c_matches(self):
        text = SAMPLE
        assert vocabulary_richness(text)["herdan_c"] == herdan_c(text)

    def test_yule_k_matches(self):
        text = SAMPLE
        assert vocabulary_richness(text)["yule_k"] == yule_k(text)

    def test_mattr_matches(self):
        text = SAMPLE
        assert vocabulary_richness(text)["mattr"] == mattr(text)

    def test_all_values_floats(self):
        result = vocabulary_richness(SAMPLE)
        for v in result.values():
            assert isinstance(v, float)

    def test_ranges(self):
        result = vocabulary_richness(SAMPLE)
        assert 0.0 <= result["ttr"] <= 1.0
        assert 0.0 <= result["herdan_c"] <= 1.0
        assert result["yule_k"] >= 0.0
        assert 0.0 <= result["mattr"] <= 1.0


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
            "sentiment_polarity", "sentiment_label", "text_density",
            "word_frequency_distribution", "vocabulary_richness",
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

    def test_sentiment_in_analyze(self):
        result = analyze("This is a wonderful day!")
        assert isinstance(result["sentiment_polarity"], float)
        assert result["sentiment_label"] in {"positive", "negative", "neutral"}

    def test_text_density_in_analyze(self):
        result = analyze(SAMPLE)
        assert isinstance(result["text_density"], float)
        assert 0.0 <= result["text_density"] <= 1.0

    def test_word_frequency_distribution_in_analyze(self):
        result = analyze(SAMPLE)
        wfd = result["word_frequency_distribution"]
        assert isinstance(wfd, dict)
        assert "zipf_fit" in wfd
        assert "total_tokens" in wfd

    def test_vocabulary_richness_in_analyze(self):
        result = analyze(SAMPLE)
        vr = result["vocabulary_richness"]
        assert isinstance(vr, dict)
        assert set(vr.keys()) == {"ttr", "herdan_c", "yule_k", "mattr"}
        assert all(isinstance(v, float) for v in vr.values())