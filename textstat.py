"""Text statistics: word count, sentence count, top words, reading time, readability."""

import re
import sys
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
    # silent trailing 'e' — only subtract if word has more than one syllable group
    if word.endswith("e") and count > 1:
        count -= 1
    # 'le' ending after a consonant counts as a syllable even without a vowel group
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
    """Gunning Fog Index — estimated years of schooling to understand the text.

    Formula: 0.4 * ((words/sentences) + 100 * (complex_words/words))
    Typical values: 6 (plain English) to 17+ (academic/legal).
    """
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
    """Coleman-Liau Index — grade level based on letters and sentences per 100 words.

    Formula: 0.0588 * L - 0.296 * S - 15.8
    L = avg letters per 100 words; S = avg sentences per 100 words.
    """
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
    """Automated Readability Index (ARI) — grade level from chars/words and words/sentences.

    Formula: 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43
    """
    words = count_words(text)
    sentences = count_sentences(text)
    if words == 0 or sentences == 0:
        return 0.0
    chars = sum(1 for c in text if c.isalpha())
    score = 4.71 * (chars / words) + 0.5 * (words / sentences) - 21.43
    return round(score, 2)


def grade_level_consensus(text: str) -> float:
    """Mean grade level from all four grade-level formulas (FK, Fog, Coleman-Liau, ARI).

    Returns 0.0 for empty text.
    """
    if not text.strip():
        return 0.0
    scores = [
        flesch_kincaid_grade(text),
        gunning_fog(text),
        coleman_liau_index(text),
        automated_readability_index(text),
    ]
    return round(sum(scores) / len(scores), 1)


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
    lines.append(f"  Coleman-Liau     : {stats['coleman_liau']}")
    lines.append(f"  ARI              : {stats['automated_readability_index']}")
    lines.append(f"  Grade consensus  : {stats['grade_level_consensus']}  (avg of 4 formulas)")
    lines.append(f"  Lexical diversity: {stats['lexical_diversity']}  (unique/total words)")
    ss = stats["sentence_stats"]
    lines.append(
        f"  Sentence lengths : min={ss['min']}  max={ss['max']}  "
        f"mean={ss['mean']}  median={ss['median']}  words"
    )
    if stats["top_words"]:
        top = ", ".join(f"{w}({c})" for w, c in stats["top_words"])
        lines.append(f"  Top words        : {top}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze text statistics from a file or stdin."
    )
    parser.add_argument("file", nargs="?", help="Text file to analyze (default: stdin)")
    parser.add_argument("--wpm", type=int, default=200, help="Reading speed (words/min)")
    args = parser.parse_args(argv)

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
    print(_format_report(stats, source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
