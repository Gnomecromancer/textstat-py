# textstat

A pure-Python text analysis library. Zero dependencies beyond the standard library.

```bash
pip install textstat-py
```

## Features

**Readability:** Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, Coleman-Liau, ARI, SMOG, grade-level consensus

**Vocabulary richness:** lexical diversity, MATTR, Herdan's C, Yule's K, hapax legomena ratio

**Structure:** word/char/sentence/paragraph counts, reading time, sentence length stats, text density

**Sentiment:** polarity score (−1 to +1), sentiment label

**N-grams:** top bigrams/trigrams, n-gram diversity

## Quick start

```python
from textstat import analyze, flesch_reading_ease, grade_level_consensus

text = "The quick brown fox jumps over the lazy dog. It did so with considerable grace."

print(flesch_reading_ease(text))    # e.g. 72.4
print(grade_level_consensus(text))  # e.g. 6.1

stats = analyze(text)               # all metrics in one dict
print(stats['sentiment_label'])     # neutral
print(stats['reading_time_min'])    # 0.05
```

## CLI

```bash
textstat document.txt          # analyze a file
cat file.txt | textstat        # pipe text
textstat --json document.txt   # JSON output
```

## API

| Function | Returns | Description |
|----------|---------|-------------|
| `analyze(text)` | dict | All metrics in one call |
| `flesch_reading_ease(text)` | float | 0–100, higher = easier |
| `flesch_kincaid_grade(text)` | float | US grade level |
| `gunning_fog(text)` | float | Years of education needed |
| `coleman_liau_index(text)` | float | Grade level by characters |
| `automated_readability_index(text)` | float | Grade level |
| `smog_index(text)` | float | Grade level (polysyllables) |
| `grade_level_consensus(text)` | float | Mean of grade-level scores |
| `lexical_diversity(text)` | float | Type-token ratio |
| `mattr(text, window=100)` | float | Moving-average TTR |
| `herdan_c(text)` | float | Herdan's C |
| `yule_k(text)` | float | Yule's K |
| `hapax_legomena_ratio(text)` | float | Fraction of once-occurring words |
| `sentiment_polarity(text)` | float | −1 (negative) to +1 (positive) |
| `sentiment_label(text)` | str | "positive" / "neutral" / "negative" |
| `reading_time(text, wpm=200)` | float | Estimated minutes to read |
| `top_words(text, n=10)` | list | Most frequent non-stop words |
| `top_ngrams(text, n=2, k=10)` | list | Most frequent n-grams |
| `sentence_stats(text)` | dict | min/max/mean/stdev sentence length |
| `paragraph_stats(text)` | dict | Paragraph count and length stats |
| `vocabulary_richness(text)` | dict | All richness metrics combined |
| `ngram_stats(text)` | dict | Bigram and trigram diversity |

## Requirements

Python 3.8+. No third-party dependencies.

## License

MIT
