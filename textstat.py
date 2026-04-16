"""Text statistics: word count, sentence count, top words, reading time, readability."""

import re
import sys
import math
import argparse
from collections import Counter
from typing import Optional


def count_words(text: str) -> int:
    """Count whitespace-separated tokens."""
    return len(text.split())


def count_chars(text: str, include_spaces: bool = True) -> int:
    """Count characters, optionally excluding whitespace."""
    if include_spaces:
        return len(text)
    return sum(1 for c in text if not c.isspace())


def count_sentences(text: str) -> int:
    """Count sentences by splitting on .  !  ? sequences."""
    sentences = re.split(r'[.!?]+', text.strip())
    return sum(1 for s in sentences if s.strip())


def avg_word_length(text: str) -> float:
    """Return mean character length of words (letters only)."""
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def top_words(text: str, n: int = 10) -> list:
    """Return the n most common lowercase words as (word, count) pairs."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "it", "as", "be", "by", "this", "that",
        "was", "are", "were", "from", "not", "have", "has", "had", "i",
        "you", "he", "she", "we", "they", "its",
    }
    filtered = [w for w in words if w not in stopwords and len(w) > 1]
    return Counter(filtered).most_common(n)


def reading_time(text: str, wpm: int = 200) -> float:
    """Estimated reading time in minutes at the given words-per-minute rate."""
    words = count_words(text)
    return round(words / wpm, 2)


def count_syllables(word: str) -> int:
    """Estimate syllable count for a single English word using vowel-group heuristic."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = len(re.findall(r"[aeiou]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in "aeiou":
        count += 1
    return max(1, count)


def total_syllables(text: str) -> int:
    """Count total syllables across all words in text."""
    words = re.findall(r"[A-Za-z']+", text)
    return sum(count_syllables(w) for w in words)


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease score (0–100; higher = easier to read)."""
    words = count_words(text)
    sentences = count_sentences(text)
    syllables = total_syllables(text)
    if words == 0 or sentences == 0:
        return 0.0
    score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    return round(score, 2)


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid Grade Level — US school grade needed to understand the text."""
    words = count_words(text)
    sentences = count_sentences(text)
    syllables = total_syllables(text)
    if words == 0 or sentences == 0:
        return 0.0
    grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    return round(grade, 2)


def lexical_diversity(text: str) -> float:
    """Type-token ratio: unique words / total words (0.0–1.0)."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 3)


def count_complex_words(text: str) -> int:
    """Count words with 3 or more syllables (polysyllabic words used in Gunning Fog)."""
    words = re.findall(r"[A-Za-z']+", text)
    return sum(1 for w in words if count_syllables(w) >= 3)


def gunning_fog(text: str) -> float:
    """Gunning Fog Index — estimated years of schooling to understand the text."""
    words = count_words(text)
    sentences = count_sentences(text)
    complex_words = count_complex_words(text)
    if words == 0 or sentences == 0:
        return 0.0
    score = 0.4 * ((words / sentences) + 100 * (complex_words / words))
    return round(score, 2)


def sentence_lengths(text: str) -> list:
    """Return a list of word counts for each sentence in text."""
    parts = re.split(r'[.!?]+', text.strip())
    return [len(p.split()) for p in parts if p.strip()]


def sentence_stats(text: str) -> dict:
    """Return min/max/mean/median sentence length (in words)."""
    lengths = sentence_lengths(text)
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    n = len(lengths)
    s = sorted(lengths)
    mid = n // 2
    median = float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / n, 2),
        "median": median,
    }


def coleman_liau_index(text: str) -> float:
    """Coleman-Liau Index — grade level based on letters and sentences per 100 words."""
    words = count_words(text)
    sentences = count_sentences(text)
    if words == 0:
        return 0.0
    letters = sum(1 for c in text if c.isalpha())
    L = (letters / words) * 100
    S = (sentences / words) * 100
    score = 0.0588 * L - 0.296 * S - 15.8
    return round(score, 2)


def automated_readability_index(text: str) -> float:
    """Automated Readability Index (ARI) — grade level from chars/words and words/sentences."""
    words = count_words(text)
    sentences = count_sentences(text)
    if words == 0 or sentences == 0:
        return 0.0
    chars = sum(1 for c in text if c.isalpha())
    score = 4.71 * (chars / words) + 0.5 * (words / sentences) - 21.43
    return round(score, 2)


def grade_level_consensus(text: str) -> float:
    """Mean grade level from all four grade-level formulas (FK, Fog, Coleman-Liau, ARI)."""
    if not text.strip():
        return 0.0
    scores = [
        flesch_kincaid_grade(text),
        gunning_fog(text),
        coleman_liau_index(text),
        automated_readability_index(text),
    ]
    return round(sum(scores) / len(scores), 1)


def smog_index(text: str) -> float:
    """SMOG Grade — Simple Measure of Gobbledygook.

    Returns 0.0 for texts with fewer than 3 sentences (insufficient data).
    """
    sentences = count_sentences(text)
    if sentences < 3:
        return 0.0
    poly = count_complex_words(text)
    score = 3 + (poly * (30 / sentences)) ** 0.5
    return round(score, 2)


def hapax_legomena_ratio(text: str) -> float:
    """Proportion of words that appear exactly once (hapax legomena / total words)."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    freq = Counter(words)
    hapax = sum(1 for count in freq.values() if count == 1)
    return round(hapax / len(words), 3)


def count_paragraphs(text: str) -> int:
    """Count non-empty paragraphs (blocks separated by one or more blank lines)."""
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return sum(1 for p in paragraphs if p.strip())


def paragraph_stats(text: str) -> dict:
    """Return word-count statistics per paragraph: count/min/max/mean."""
    paragraphs = [p for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
    if not paragraphs:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    lengths = [count_words(p) for p in paragraphs]
    n = len(lengths)
    return {
        "count": n,
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / n, 2),
    }


# ---------------------------------------------------------------------------
# Sentiment polarity (lexicon-based, no external deps)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = frozenset({
    "good", "great", "excellent", "wonderful", "amazing", "fantastic", "love",
    "happy", "joy", "beautiful", "best", "brilliant", "superb", "perfect",
    "positive", "nice", "enjoy", "pleased", "glad", "delightful", "awesome",
    "outstanding", "magnificent", "splendid", "marvelous", "terrific", "fine",
    "pleasant", "fortunate", "favorable", "win", "success", "triumph",
})

_NEGATIVE_WORDS = frozenset({
    "bad", "terrible", "awful", "horrible", "hate", "sad", "ugly", "worst",
    "poor", "wrong", "failure", "fail", "loss", "lost", "negative", "nasty",
    "dreadful", "disgrace", "disaster", "problem", "trouble", "difficult",
    "unfortunate", "unfavorable", "painful", "suffer", "miserable", "evil",
    "damage", "broken", "useless", "inferior", "mediocre", "flawed",
})

_NEGATORS = frozenset({
    "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
    "hardly", "scarcely", "barely", "without",
})


def sentiment_polarity(text: str) -> float:
    """Simple lexicon-based sentiment polarity score in range [-1.0, 1.0].

    Counts positive and negative words from a built-in lexicon, handles
    single-word negation (e.g. "not good" flips polarity), and normalises
    by total matched words.  Returns 0.0 for empty or neutral text.
    """
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0

    score = 0
    total = 0
    for i, word in enumerate(words):
        negated = i > 0 and words[i - 1] in _NEGATORS
        if word in _POSITIVE_WORDS:
            score += -1 if negated else 1
            total += 1
        elif word in _NEGATIVE_WORDS:
            score += 1 if negated else -1
            total += 1

    if total == 0:
        return 0.0
    return round(score / total, 3)


def sentiment_label(text: str) -> str:
    """Return 'positive', 'negative', or 'neutral' based on polarity score."""
    p = sentiment_polarity(text)
    if p > 0.05:
        return "positive"
    if p < -0.05:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Text density
# ---------------------------------------------------------------------------

def text_density(text: str) -> float:
    """Content-word density: ratio of non-stopword words to total words.

    Higher density indicates more information-rich prose (fewer filler words).
    Returns 0.0 for empty text.
    """
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "it", "as", "be", "by", "this", "that",
        "was", "are", "were", "from", "not", "have", "has", "had", "i",
        "you", "he", "she", "we", "they", "its",
    }
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    content = sum(1 for w in words if w not in stopwords)
    return round(content / len(words), 3)


# ---------------------------------------------------------------------------
# Zipf / frequency distribution
# ---------------------------------------------------------------------------

def word_frequency_distribution(text: str) -> dict:
    """Return word frequency distribution stats.

    Keys:
        ``total_tokens``   — total word tokens
        ``unique_types``   — number of distinct word types
        ``top_10_pct``     — fraction of tokens covered by the 10 most frequent words
        ``zipf_fit``       — Pearson correlation of log(rank) vs log(freq); near -1.0
                            means frequency follows Zipf's law closely
    Returns zeroed dict for texts with fewer than 5 unique word types.
    """
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return {"total_tokens": 0, "unique_types": 0, "top_10_pct": 0.0, "zipf_fit": 0.0}

    freq = Counter(words)
    unique = len(freq)
    total = len(words)

    top10 = sum(c for _, c in freq.most_common(10))
    top_10_pct = round(top10 / total, 3)

    if unique < 5:
        return {
            "total_tokens": total,
            "unique_types": unique,
            "top_10_pct": top_10_pct,
            "zipf_fit": 0.0,
        }

    # Pearson correlation between log(rank) and log(frequency)
    ranked = [c for _, c in freq.most_common()]
    log_ranks = [math.log(r + 1) for r in range(unique)]
    log_freqs = [math.log(f) for f in ranked]

    n = unique
    mean_r = sum(log_ranks) / n
    mean_f = sum(log_freqs) / n
    cov = sum((log_ranks[i] - mean_r) * (log_freqs[i] - mean_f) for i in range(n))
    std_r = math.sqrt(sum((x - mean_r) ** 2 for x in log_ranks))
    std_f = math.sqrt(sum((x - mean_f) ** 2 for x in log_freqs))

    if std_r == 0 or std_f == 0:
        zipf_fit = 0.0
    else:
        zipf_fit = round(cov / (std_r * std_f), 3)

    return {
        "total_tokens": total,
        "unique_types": unique,
        "top_10_pct": top_10_pct,
        "zipf_fit": zipf_fit,
    }


# ---------------------------------------------------------------------------
# Vocabulary richness metrics
# ---------------------------------------------------------------------------

def herdan_c(text: str) -> float:
    """Herdan's C — log-based vocabulary richness, less biased by text length than TTR.

    C = log(V) / log(N) where V = unique word types, N = total tokens.
    Range (0, 1]; closer to 1 means richer vocabulary relative to text length.
    Returns 0.0 for texts with fewer than 2 tokens.
    """
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    n = len(words)
    if n < 2:
        return 0.0
    v = len(set(words))
    if v < 2:
        return 0.0
    return round(math.log(v) / math.log(n), 3)


def yule_k(text: str) -> float:
    """Yule's K — vocabulary richness measure robust to text length.

    K = 10^4 * (sum(m^2 * V(m)) - N) / N^2
    where V(m) = number of word types occurring exactly m times, N = total tokens.
    Lower K means richer vocabulary (words are distributed more evenly).
    Returns 0.0 for empty text.
    """
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    n = len(words)
    if n == 0:
        return 0.0
    freq = Counter(words)
    freq_of_freq = Counter(freq.values())
    numerator = sum(m * m * fm for m, fm in freq_of_freq.items()) - n
    return round(1e4 * numerator / (n * n), 2)


def mattr(text: str, window: int = 100) -> float:
    """Moving Average Type-Token Ratio (MATTR).

    Slides a window of `window` tokens over the text, computes TTR per window,
    and returns the mean.  More stable than global TTR for long texts.
    Falls back to global TTR when text is shorter than the window size.
    Returns 0.0 for empty text.
    """
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    n = len(words)
    if n == 0:
        return 0.0
    if n <= window:
        return round(len(set(words)) / n, 3)
    ttrs = [len(set(words[i:i + window])) / window for i in range(n - window + 1)]
    return round(sum(ttrs) / len(ttrs), 3)


def vocabulary_richness(text: str) -> dict:
    """Bundle of vocabulary richness metrics beyond simple TTR.

    Keys:
        ``ttr``      — type-token ratio (same as lexical_diversity)
        ``herdan_c`` — log-based richness, more stable across text lengths
        ``yule_k``   — Yule's K; lower value = richer vocabulary
        ``mattr``    — moving-average TTR (window=100 tokens)
    """
    return {
        "ttr": lexical_diversity(text),
        "herdan_c": herdan_c(text),
        "yule_k": yule_k(text),
        "mattr": mattr(text),
    }


# ---------------------------------------------------------------------------
# N-gram analysis
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list:
    """Lowercase word tokens (letters and apostrophes only)."""
    return re.findall(r"[A-Za-z']+", text.lower())


def top_ngrams(text: str, n: int = 2, k: int = 10) -> list:
    """Return the k most common n-grams as list of (ngram_tuple, count) pairs.

    Args:
        text: Input text.
        n:    N-gram size (2=bigrams, 3=trigrams, etc.).
        k:    Maximum number of results to return.

    Returns an empty list when the text has fewer than n tokens.
    """
    tokens = _tokenize(text)
    if len(tokens) < n:
        return []
    ngrams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams).most_common(k)


def ngram_diversity(text: str, n: int = 2) -> float:
    """Unique n-grams / total n-gram positions (n-gram type-token ratio).

    Values near 1.0 indicate the text avoids repeating phrases.
    Returns 0.0 when the text has fewer than n tokens.
    """
    tokens = _tokenize(text)
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0
    ngrams = [tuple(tokens[i: i + n]) for i in range(total)]
    return round(len(set(ngrams)) / total, 3)


def ngram_stats(text: str) -> dict:
    """Bundle of n-gram statistics.

    Keys:
        ``top_bigrams``       — up to 5 most frequent bigrams as (tuple, count) pairs
        ``top_trigrams``      — up to 5 most frequent trigrams
        ``bigram_diversity``  — unique bigrams / total bigram positions
        ``trigram_diversity`` — unique trigrams / total trigram positions
    """
    return {
        "top_bigrams": top_ngrams(text, n=2, k=5),
        "top_trigrams": top_ngrams(text, n=3, k=5),
        "bigram_diversity": ngram_diversity(text, n=2),
        "trigram_diversity": ngram_diversity(text, n=3),
    }


# ---------------------------------------------------------------------------
# Writing quality signals
# ---------------------------------------------------------------------------

_BE_FORMS = frozenset({"am", "is", "are", "was", "were", "be", "been", "being"})
_PAST_PARTICIPLE_SUFFIXES = ("ed", "en", "wn", "rn", "ne", "lt", "pt", "nt")
_LY_STOPWORDS = frozenset({
    "only", "early", "likely", "family", "daily", "really", "already",
    "nearly", "clearly", "simply", "finally", "usually", "roughly",
})


def passive_voice_ratio(text: str) -> float:
    """Fraction of sentences containing a passive construction.

    Detects ``be`` verb followed (within 3 tokens) by a past participle
    (word ending in -ed, -en, -wn, etc.).  Handles common patterns:
    "was written", "is being done", "has been deleted".

    Returns a value in [0.0, 1.0]; 0.0 for empty or single-word text.
    """
    parts = [p.strip() for p in re.split(r'[.!?]+', text.strip()) if p.strip()]
    if not parts:
        return 0.0
    passive_count = 0
    for sentence in parts:
        tokens = re.findall(r"[a-z']+", sentence.lower())
        for i, tok in enumerate(tokens):
            if tok in _BE_FORMS:
                window = tokens[i + 1: i + 4]
                for w in window:
                    if any(w.endswith(sfx) for sfx in _PAST_PARTICIPLE_SUFFIXES) and len(w) > 3:
                        passive_count += 1
                        break
    return round(passive_count / len(parts), 3)


def adverb_density(text: str) -> float:
    """Fraction of words that are likely adverbs (end in -ly, not stopwords).

    High adverb density (> 0.05) often signals weak writing — adverbs patching
    weak verbs ("ran quickly" vs "sprinted").  Returns 0.0 for empty text.
    """
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    adverbs = [w for w in words if w.endswith("ly") and w not in _LY_STOPWORDS and len(w) > 3]
    return round(len(adverbs) / len(words), 3)


def analyze(text: str) -> dict:
    """Return a combined stats dict for the given text."""
    return {
        "words": count_words(text),
        "chars": count_chars(text),
        "chars_no_spaces": count_chars(text, include_spaces=False),
        "sentences": count_sentences(text),
        "avg_word_length": round(avg_word_length(text), 2),
        "reading_time_min": reading_time(text),
        "top_words": top_words(text, 5),
        "flesch_reading_ease": flesch_reading_ease(text),
        "flesch_kincaid_grade": flesch_kincaid_grade(text),
        "lexical_diversity": lexical_diversity(text),
        "gunning_fog": gunning_fog(text),
        "sentence_stats": sentence_stats(text),
        "coleman_liau": coleman_liau_index(text),
        "automated_readability_index": automated_readability_index(text),
        "grade_level_consensus": grade_level_consensus(text),
        "smog_index": smog_index(text),
        "hapax_legomena_ratio": hapax_legomena_ratio(text),
        "paragraph_stats": paragraph_stats(text),
        "sentiment_polarity": sentiment_polarity(text),
        "sentiment_label": sentiment_label(text),
        "text_density": text_density(text),
        "word_frequency_distribution": word_frequency_distribution(text),
        "vocabulary_richness": vocabulary_richness(text),
        "ngram_stats": ngram_stats(text),
        "passive_voice_ratio": passive_voice_ratio(text),
        "adverb_density": adverb_density(text),
    }


def _format_report(stats: dict, source: str) -> str:
    lines = [f"=== Text Statistics: {source} ==="]
    lines.append(f"  Words            : {stats['words']}")
    lines.append(f"  Characters       : {stats['chars']}  ({stats['chars_no_spaces']} without spaces)")
    lines.append(f"  Sentences        : {stats['sentences']}")
    lines.append(f"  Avg word length  : {stats['avg_word_length']}")
    lines.append(f"  Reading time     : {stats['reading_time_min']} min")
    lines.append(f"  Flesch ease      : {stats['flesch_reading_ease']}  (0=hard, 100=easy)")
    lines.append(f"  FK grade level   : {stats['flesch_kincaid_grade']}")
    lines.append(f"  Gunning Fog      : {stats['gunning_fog']}  (years of schooling)")
    lines.append(f"  SMOG index       : {stats['smog_index']}  (needs 3+ sentences)")
    lines.append(f"  Coleman-Liau     : {stats['coleman_liau']}")
    lines.append(f"  ARI              : {stats['automated_readability_index']}")
    lines.append(f"  Grade consensus  : {stats['grade_level_consensus']}  (avg of 4 formulas)")
    lines.append(f"  Lexical diversity: {stats['lexical_diversity']}  (unique/total words)")
    lines.append(f"  Hapax legomena   : {stats['hapax_legomena_ratio']}  (once-only words / total)")
    lines.append(f"  Text density     : {stats['text_density']}  (content words / total)")
    lines.append(f"  Sentiment        : {stats['sentiment_label']}  (polarity={stats['sentiment_polarity']})")
    wfd = stats["word_frequency_distribution"]
    lines.append(
        f"  Freq dist        : top10={wfd['top_10_pct']}  zipf_fit={wfd['zipf_fit']}"
        f"  ({wfd['unique_types']} types / {wfd['total_tokens']} tokens)"
    )
    vr = stats["vocabulary_richness"]
    lines.append(
        f"  Vocab richness   : ttr={vr['ttr']}  herdan_c={vr['herdan_c']}"
        f"  yule_k={vr['yule_k']}  mattr={vr['mattr']}"
    )
    ss = stats["sentence_stats"]
    lines.append(
        f"  Sentence lengths : min={ss['min']}  max={ss['max']}  "
        f"mean={ss['mean']}  median={ss['median']}  words"
    )
    ps = stats["paragraph_stats"]
    lines.append(
        f"  Paragraphs       : {ps['count']}  (min={ps['min']}  max={ps['max']}  mean={ps['mean']} words)"
    )
    ng = stats["ngram_stats"]
    if ng["top_bigrams"]:
        top_bi = ", ".join(" ".join(g) for g, _ in ng["top_bigrams"][:3])
        lines.append(
            f"  N-grams          : bigram_div={ng['bigram_diversity']}  "
            f"trigram_div={ng['trigram_diversity']}  top_bigrams=[{top_bi}]"
        )
    if stats["top_words"]:
        top = ", ".join(f"{w}({c})" for w, c in stats["top_words"])
        lines.append(f"  Top words        : {top}")
    lines.append(f"  Passive voice    : {stats['passive_voice_ratio']}  (fraction of sentences)")
    lines.append(f"  Adverb density   : {stats['adverb_density']}  (>0.05 may signal weak verbs)")
    return "\n".join(lines)


def _compare_report(s1: dict, s2: dict, name1: str, name2: str) -> str:
    _COMPARE_KEYS = [
        ("words", "Words"),
        ("sentences", "Sentences"),
        ("reading_time_min", "Reading time (min)"),
        ("flesch_reading_ease", "Flesch ease"),
        ("grade_level_consensus", "Grade level"),
        ("lexical_diversity", "Lexical diversity"),
        ("sentiment_polarity", "Sentiment polarity"),
        ("passive_voice_ratio", "Passive voice ratio"),
        ("adverb_density", "Adverb density"),
        ("text_density", "Text density"),
    ]
    w = 26
    lines = [
        f"{'Metric':<{w}}  {'A: ' + name1:<20}  {'B: ' + name2:<20}  Delta",
        "-" * (w + 50),
    ]
    for key, label in _COMPARE_KEYS:
        v1, v2 = s1.get(key, 0), s2.get(key, 0)
        try:
            delta = round(float(v2) - float(v1), 3)
            delta_str = f"+{delta}" if delta > 0 else str(delta)
        except (TypeError, ValueError):
            delta_str = "n/a"
        lines.append(f"{label:<{w}}  {str(v1):<20}  {str(v2):<20}  {delta_str}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze text statistics from a file or stdin."
    )
    parser.add_argument("file", nargs="?", help="Text file to analyze (default: stdin)")
    parser.add_argument("--compare", metavar="FILE2", help="Compare FILE against FILE2")
    parser.add_argument("--wpm", type=int, default=200, help="Reading speed (words/min)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args(argv)

    if args.compare:
        if not args.file:
            print("error: --compare requires a base file argument", file=sys.stderr)
            return 1
        try:
            with open(args.file, encoding="utf-8") as fh:
                text1 = fh.read()
            with open(args.compare, encoding="utf-8") as fh:
                text2 = fh.read()
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        s1, s2 = analyze(text1), analyze(text2)
        print(_compare_report(s1, s2, args.file, args.compare))
        return 0

    if args.file:
        try:
            with open(args.file, encoding="utf-8") as fh:
                text = fh.read()
            source = args.file
        except FileNotFoundError:
            print(f"error: file not found: {args.file}", file=sys.stderr)
            return 1
    else:
        if sys.stdin.isatty():
            print("error: provide a file or pipe text via stdin", file=sys.stderr)
            return 1
        text = sys.stdin.read()
        source = "stdin"

    stats = analyze(text)
    stats["reading_time_min"] = reading_time(text, wpm=args.wpm)

    if args.json:
        import json
        print(json.dumps(stats, default=list))
    else:
        print(_format_report(stats, source))
    return 0


if __name__ == "__main__":
    sys.exit(main())