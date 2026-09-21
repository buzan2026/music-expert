"""Recommendation engine — bootstrap scoring + SGDClassifier online learning."""
# Phase bootstrap (< 20 verdicts):
#   score = (1 / log(playcount + 2)) * (0.7 + 0.3 * random())
#
# Phase nominale (>= 20 verdicts):
#   SGDClassifier(loss='log_loss'), partial_fit() on each verdict.
#   Features: listeners_log, playcount_log, year, tags_onehot, seed_onehot, similarity
#
# Pick strategy: 70% top-score, 30% max uncertainty (|score - 0.5| minimal)
