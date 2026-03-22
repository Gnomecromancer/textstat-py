"""Text statistics: word count, sentence count, top words, reading time."""

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
    }


def _format_report(stats: dict, source: str) -> str:
    lines = [f"=== Text Statistics: {source} ==="]
    lines.append(f"  Words            : {stats['words']}")
    lines.append(f"  Characters       : {stats['chars']}  ({stats['chars_no_spaces']} without spaces)")
    lines.append(f"  Sentences        : {stats['sentences']}")
    lines.append(f"  Avg word length  : {stats['avg_word_length']}")
    lines.append(f"  Reading time     : {stats['reading_time_min']} min")
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