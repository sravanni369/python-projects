"""Lexicon-based sentiment analyser.

Scores text as positive, negative, or neutral by looking each word up in a
small opinion lexicon, then applying negation and intensifier rules.

No third-party libraries. Run: python sentiment_analyser.py
"""

import re
from collections import Counter

POSITIVE = {
    "good": 1.0, "great": 1.5, "excellent": 2.0, "amazing": 2.0, "love": 1.8,
    "wonderful": 1.8, "fantastic": 2.0, "happy": 1.3, "best": 1.7, "nice": 0.8,
    "helpful": 1.2, "fast": 0.7, "clean": 0.7, "works": 0.8, "recommend": 1.4,
}

NEGATIVE = {
    "bad": -1.0, "terrible": -2.0, "awful": -2.0, "hate": -1.8, "worst": -2.0,
    "poor": -1.2, "slow": -0.8, "broken": -1.5, "useless": -1.7, "buggy": -1.3,
    "disappointing": -1.5, "crash": -1.4, "confusing": -1.0, "waste": -1.6,
}

LEXICON = {**POSITIVE, **NEGATIVE}

NEGATIONS = {"not", "no", "never", "none", "cannot", "cant", "didnt", "dont",
             "isnt", "wasnt", "wouldnt", "shouldnt", "couldnt", "wont"}

INTENSIFIERS = {"very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3,
                "too": 1.2, "absolutely": 1.7, "slightly": 0.5, "somewhat": 0.6}

NEGATION_SCOPE = 3


def tokenize(text):
    """Split text into lowercase word tokens, dropping punctuation."""
    return re.findall(r"[a-z']+", text.lower().replace("'", ""))


def score_tokens(tokens):
    """Score tokens, applying negation flips and intensifier multipliers.

    Returns (total_score, hits) where hits lists (word, applied_score).
    """
    total = 0.0
    hits = []

    for i, word in enumerate(tokens):
        if word not in LEXICON:
            continue

        value = LEXICON[word]

        # An intensifier immediately before the word scales it.
        if i > 0 and tokens[i - 1] in INTENSIFIERS:
            value *= INTENSIFIERS[tokens[i - 1]]

        # A negation within the preceding few words flips and dampens it, but
        # only if no other lexicon word sits in between — a negation applies to
        # the first opinion word it reaches and is spent there. Without this,
        # "not bad, quite nice" would wrongly negate "nice" as well.
        window = tokens[max(0, i - NEGATION_SCOPE):i]
        if any(w in NEGATIONS for w in window) and not any(w in LEXICON for w in window):
            value *= -0.75

        total += value
        hits.append((word, round(value, 2)))

    return total, hits


def classify(score):
    """Map a raw score to a label using a neutral dead zone."""
    if score > 0.5:
        return "positive"
    if score < -0.5:
        return "negative"
    return "neutral"


def analyse(text):
    """Analyse one piece of text and return a report dict."""
    tokens = tokenize(text)
    score, hits = score_tokens(tokens)
    return {
        "text": text,
        "tokens": len(tokens),
        "score": round(score, 2),
        "label": classify(score),
        "hits": hits,
    }


def summarise(reports):
    """Count labels across many reports."""
    return Counter(r["label"] for r in reports)


if __name__ == "__main__":
    reviews = [
        "This app is really great and the support team was helpful",
        "Terrible experience, the checkout is broken and slow",
        "It arrived on Tuesday in a cardboard box",
        "Not good at all, honestly a waste of money",
        "The interface is not bad, actually quite nice",
    ]

    reports = [analyse(r) for r in reviews]

    for r in reports:
        print(f"[{r['label']:>8}] score={r['score']:>6}  {r['text']}")
        if r["hits"]:
            print(f"           matched: {r['hits']}")

    print()
    for label, n in summarise(reports).most_common():
        print(f"{label}: {n}")
