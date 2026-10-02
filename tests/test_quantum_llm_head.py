"""The classical side of IonQ's quantum LLM head (Result 91).

Only the parts that do not need the ML stack or a download: the split must be the
paper's (sizes, balance, disjointness), and model choice must use validation,
never test.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp036_quantum_llm_head import best_on_validation, paper_split


def corpus_labels():
    """SetFit/sst2's size and class balance: 9 613 sentences, 4 963 positive."""
    labels = np.zeros(9613, dtype=int)
    labels[:4963] = 1
    return labels


def test_split_matches_the_paper():
    train, val, test = paper_split(corpus_labels(), seed=0)
    assert len(train) == 435 and len(val) == 77
    assert len(test) == 9613 - 512 == 9101


def test_split_is_balanced_and_disjoint():
    labels = corpus_labels()
    train, val, test = paper_split(labels, seed=3)
    chosen = np.concatenate([train, val])
    assert labels[chosen].sum() == 256
    assert len(set(train) | set(val) | set(test)) == len(labels)
    assert not (set(train) & set(test)) and not (set(val) & set(test))


def test_seeds_give_different_splits():
    a, _, _ = paper_split(corpus_labels(), seed=0)
    b, _, _ = paper_split(corpus_labels(), seed=1)
    assert set(a) != set(b)


class Fixed:
    """A 'model' with a set validation score and a set test score."""

    def __init__(self, val, test):
        self.val, self.test = val, test

    def fit(self, features, labels):
        return self

    def score(self, features, labels):
        return self.val if len(labels) == 2 else self.test


def test_choice_is_made_on_validation_not_test():
    features, labels = np.zeros((9, 1)), np.zeros(9)
    train, val, test = np.arange(3), np.arange(3, 5), np.arange(5, 9)
    # the second model would win on test; the choice must not see that
    models = [Fixed(val=0.9, test=0.5), Fixed(val=0.6, test=0.99)]
    assert best_on_validation(models, features, labels, train, val, test) == pytest.approx(0.5)
