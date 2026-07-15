"""Tests for sentiment_analyser."""

from sentiment_analyser import analyse, classify, score_tokens, tokenize


def test_tokenize_strips_punctuation_and_case():
    assert tokenize("Terrible experience, the checkout!") == [
        "terrible", "experience", "the", "checkout"
    ]


def test_tokenize_joins_contractions():
    # The apostrophe is removed before splitting, so "don't" becomes one token
    # that matches the "dont" entry in NEGATIONS.
    assert tokenize("I don't care") == ["i", "dont", "care"]


def test_unknown_words_score_zero():
    score, hits = score_tokens(tokenize("it arrived in a cardboard box"))
    assert score == 0.0
    assert hits == []


def test_positive_word_scores_positive():
    assert analyse("the support was helpful")["label"] == "positive"


def test_negative_word_scores_negative():
    assert analyse("the checkout is broken")["label"] == "negative"


def test_intensifier_multiplies_preceding_word():
    plain = analyse("this is great")["score"]
    boosted = analyse("this is really great")["score"]
    assert boosted > plain
    assert boosted == 2.1  # great 1.5 * really 1.4


def test_negation_flips_positive_to_negative():
    assert analyse("not good at all, a waste of money")["label"] == "negative"


def test_negation_flips_negative_to_positive():
    assert analyse("the interface is not bad, quite nice")["label"] == "positive"


def test_negation_dampens_rather_than_fully_inverting():
    # "not good" should read negative, but weaker than an outright "terrible".
    not_good = analyse("not good")["score"]
    terrible = analyse("terrible")["score"]
    assert not_good < 0
    assert not_good > terrible


def test_negation_outside_window_does_not_apply():
    # NEGATION_SCOPE is 3, so a negation five words back must not reach.
    assert analyse("not sure how but anyway it was great")["label"] == "positive"


def test_classify_dead_zone_is_neutral():
    assert classify(0.5) == "neutral"
    assert classify(-0.5) == "neutral"
    assert classify(0.51) == "positive"
    assert classify(-0.51) == "negative"


def test_empty_text_is_neutral():
    report = analyse("")
    assert report["label"] == "neutral"
    assert report["tokens"] == 0


def test_hits_report_the_words_that_drove_the_score():
    hits = dict(analyse("the checkout is broken and slow")["hits"])
    assert hits == {"broken": -1.5, "slow": -0.8}


def test_known_limitation_double_negative_is_wrong():
    # A lexicon method cannot handle this: "don't hate" flips hate to positive,
    # which happens to be right here, but the mechanism is naive. This test
    # documents the actual behaviour rather than pretending it reasons.
    assert analyse("I don't hate it")["score"] == 1.35  # hate -1.8 * -0.75
