# textstat-py

Text analysis for Python. **Zero dependencies.**

NLTK is 40MB and requires a corpus download just to tokenize a sentence. textblob pulls in NLTK. spaCy needs a 50MB model file before it'll tell you anything. For most text analysis tasks — readability scores, vocabulary stats, sentiment, writing quality signals — none of that weight is necessary.

```
pip install textstat-py
```

## What it does

```
$ textstat essay.txt
=== Text Statistics: essay.txt ===
  Words            : 1243
  Sentences        : 67
  Reading time     : 6.2 min
  Flesch ease      : 58.4  (0=hard, 100=easy)
  FK grade level   : 11.2
  Grade consensus  : 10.8  (avg of 4 formulas)
  Lexical diversity: 0.71  (unique/total words)
  Sentiment        : neutral  (polarity=0.02)
  Passive voice    : 0.18  (fraction of sentences)
  Adverb density   : 0.031  (>0.05 may signal weak verbs)
  Top words        : data(18), model(14), training(11), loss(9), layer(7)
```

Compare two versions of the same document:

```
$ textstat --compare draft.txt final.txt
Metric                      A: draft.txt          B: final.txt          Delta
------------------------------------------------------------------------
Words                       1891                  1243                  -648
Reading time (min)          9.46                  6.21                  -3.25
Flesch ease                 44.1                  58.4                  +14.3
Grade level                 13.2                  10.8                  -2.4
Passive voice ratio         0.31                  0.18                  -0.13
Adverb density              0.071                 0.031                 -0.04
```

## Install

```bash
pip install textstat-py
```

Python 3.8+. No dependencies. Single file.

## CLI

```bash
textstat document.txt          # full report
textstat --json document.txt   # JSON output
textstat --wpm 250 document.txt  # custom reading speed
textstat --compare before.txt after.txt  # side-by-side diff
cat text.txt | textstat        # stdin
```

## Python API

```python
from textstat import analyze, flesch_reading_ease, grade_level_consensus

text = open("essay.txt").read()

# Quick scores
print(flesch_reading_ease(text))    # 58.4
print(grade_level_consensus(text))  # 10.8

# Full analysis dict
stats = analyze(text)
print(stats["passive_voice_ratio"])  # 0.18
print(stats["adverb_density"])       # 0.031
print(stats["top_words"])            # [("data", 18), ("model", 14), ...]
```

## Functions

**Readability**
- `flesch_reading_ease(text)` — 0–100, higher = easier
- `flesch_kincaid_grade(text)` — US grade level
- `gunning_fog(text)` — years of education needed
- `coleman_liau_index(text)`
- `automated_readability_index(text)`
- `smog_index(text)`
- `grade_level_consensus(text)` — mean across all grade formulas

**Writing quality**
- `passive_voice_ratio(text)` — fraction of sentences with passive constructions
- `adverb_density(text)` — fraction of words that are -ly adverbs (>0.05 is a signal)

**Vocabulary**
- `lexical_diversity(text)` — type-token ratio
- `mattr(text, window=100)` — moving-average TTR, stable for long texts
- `herdan_c(text)`, `yule_k(text)` — length-robust vocabulary richness
- `hapax_legomena_ratio(text)` — fraction of words appearing exactly once
- `vocabulary_richness(text)` — all of the above as a dict

**Counts & structure**
- `count_words(text)`, `count_sentences(text)`, `count_paragraphs(text)`
- `reading_time(text, wpm=200)`
- `sentence_stats(text)` — min/max/mean/median sentence length
- `paragraph_stats(text)` — word counts per paragraph

**Sentiment**
- `sentiment_polarity(text)` — −1.0 to +1.0, lexicon-based, no model needed
- `sentiment_label(text)` — "positive" / "neutral" / "negative"

**N-grams**
- `top_ngrams(text, n=2, k=10)` — most frequent n-grams
- `ngram_diversity(text, n=2)` — unique n-grams / total positions
- `ngram_stats(text)` — bigrams + trigrams bundled

**Misc**
- `top_words(text, n=10)` — most frequent non-stopword words
- `word_frequency_distribution(text)` — total tokens, unique types, Zipf fit
- `text_density(text)` — content words / total words

## License

MIT
