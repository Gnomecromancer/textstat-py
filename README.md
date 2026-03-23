# textstat

Text analysis for Python. Readability scores, vocabulary stats, sentiment, n-grams — no dependencies.

```
pip install textstat-py
```

## Usage

```python
from textstat import analyze, flesch_reading_ease, grade_level_consensus

text = open("essay.txt").read()

print(flesch_reading_ease(text))    # 68.4
print(grade_level_consensus(text))  # 9.2

stats = analyze(text)
# stats is a flat dict with everything:
# reading_time_min, sentiment_label, vocabulary_richness, sentence_stats, ...
```

## CLI

```
textstat document.txt
cat file.txt | textstat
textstat --json report.txt
```

## Functions

**Readability**
- `flesch_reading_ease(text)` — 0–100
- `flesch_kincaid_grade(text)` — US grade level
- `gunning_fog(text)` — years of education
- `coleman_liau_index(text)`
- `automated_readability_index(text)`
- `smog_index(text)`
- `grade_level_consensus(text)` — average across all grade metrics

**Vocabulary**
- `lexical_diversity(text)` — type-token ratio
- `mattr(text, window=100)` — moving-average TTR
- `herdan_c(text)`, `yule_k(text)`
- `hapax_legomena_ratio(text)` — fraction of words appearing once
- `vocabulary_richness(text)` — all of the above as a dict

**Counts & structure**
- `count_words(text)`, `count_sentences(text)`, `count_paragraphs(text)`
- `reading_time(text, wpm=200)`
- `sentence_stats(text)`, `paragraph_stats(text)`

**Sentiment**
- `sentiment_polarity(text)` — −1 to +1
- `sentiment_label(text)` — "positive" / "neutral" / "negative"

**N-grams**
- `top_ngrams(text, n=2, k=10)`
- `ngram_diversity(text, n=2)`
- `ngram_stats(text)`

**Misc**
- `top_words(text, n=10)`
- `word_frequency_distribution(text)`
- `text_density(text)`

## Requirements

Python 3.8+

## License

MIT
