"""Experiment 036 -- does IonQ's "quantum-enhanced LLM fine-tuning" beat classical?

IonQ's arXiv:2504.08732 ("Quantum Large Language Model Fine-Tuning") puts a
simulated parameterised-circuit head on sentence-transformer embeddings for SST-2
sentiment in a low-data regime and reports **92.70%** for the best quantum head
(14 qubits, 886 parameters) against **89.56%** for the best classical baseline,
"up to 3.14%". No seeds, error bars or significance test are reported, and the
quantum figure is the best of a hyperparameter screen.

The quantum head is under-specified in the paper (how the 768-dimensional
embedding reaches the qubits is not stated), so it is **not** reproduced here,
and nothing below is a refutation in the rubric's A' sense. What this measures
is the classical side of the comparison, on the paper's own data, model and
split protocol:

1. **Frozen embeddings** (``paraphrase-mpnet-base-v2``, as the paper names) with
   classical heads selected on validation, over many seeds -- does the paper's
   89.56% reproduce, and is it the ceiling for classical heads on that
   representation?
2. **SetFit** -- the few-shot method the paper's pipeline is named after, which
   contrastively fine-tunes the sentence transformer on the same 435 training
   sentences -- followed by an ordinary logistic-regression head. If that lands
   at the quantum head's number, the claimed margin is the gap between a frozen
   and a fine-tuned *representation*, which a classical pipeline closes without
   any quantum component.

The paper's text is ambiguous on exactly that point: it says the base model's
weights were kept frozen, and also that the embeddings were "output by SetFit".

**Protocol, as the paper states it:** SetFit/sst2 (6 920 + 872 + 1 821 = 9 613
sentences), 256 positive and 256 negative sampled, split 85/15 into train and
validation, model chosen on validation, accuracy reported on the remaining
9 101.

Needs the optional ML stack: ``pip install --index-url
https://download.pytorch.org/whl/cpu torch`` then ``pip install
sentence-transformers datasets setfit``. SetFit costs ~20 CPU-minutes per seed.

Run:  python -m experiments.exp036_quantum_llm_head [--seeds 20 --setfit-seeds 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qres.bench import RESULTS_DIR

#: the paper's numbers, for the comparison
PAPER_QUANTUM = 92.70
PAPER_CLASSICAL = 89.56

MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
DATASET = "SetFit/sst2"
PER_CLASS = 256
TRAIN_FRACTION = 0.85

#: SetFit contrastive pairs per training sentence; its default is 20, and 5
#: already reaches the number this experiment is about, at a quarter the cost
SETFIT_ITERATIONS = 5

CACHE = RESULTS_DIR / "raw" / "exp036_embeddings.npz"


def paper_split(labels, seed):
    """256 + 256 sampled, 85/15 train/validation, everything else is test."""
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    chosen = np.concatenate([rng.choice(positive, PER_CLASS, replace=False),
                             rng.choice(negative, PER_CLASS, replace=False)])
    rng.shuffle(chosen)
    cut = int(round(TRAIN_FRACTION * len(chosen)))
    test = np.setdiff1d(np.arange(len(labels)), chosen)
    return chosen[:cut], chosen[cut:], test


def heads(seed):
    """Classical heads and their grids; every one is chosen on validation."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC

    return {
        "logistic regression (769 params)": [
            LogisticRegression(C=c, max_iter=5000) for c in (0.01, 0.1, 1, 10, 100)],
        "SVC, RBF kernel": [
            SVC(C=c, gamma=g) for c in (0.1, 1, 10, 100) for g in ("scale", 0.1, 1.0)],
        "MLP 1x192 (148k params)": [
            MLPClassifier((192,), alpha=a, max_iter=800, random_state=seed)
            for a in (1e-4, 1e-2, 1)],
        "PCA-14 + MLP 48 (~870 params)": [
            make_pipeline(PCA(14, random_state=seed),
                          MLPClassifier((48,), alpha=a, max_iter=2000, random_state=seed))
            for a in (1e-4, 1e-2, 1)],
    }


def best_on_validation(models, features, labels, train, val, test):
    chosen = max(models, key=lambda m: m.fit(features[train], labels[train])
                 .score(features[val], labels[val]))
    return float(chosen.score(features[test], labels[test]))


def load_corpus():
    from datasets import load_dataset

    data = load_dataset(DATASET)
    texts, labels = [], []
    for split in ("train", "validation", "test"):
        texts += list(data[split]["text"])
        labels += list(data[split]["label"])
    return texts, np.array(labels)


def frozen_embeddings(texts):
    if CACHE.exists():
        return np.load(CACHE)["embeddings"]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device="cpu")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False,
                              convert_to_numpy=True).astype(np.float32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, embeddings=embeddings)
    return embeddings


def setfit_arm(texts, labels, seed, iterations=SETFIT_ITERATIONS):
    """Contrastive fine-tuning on the 435 training sentences, then classical heads."""
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    from sklearn.linear_model import LogisticRegression

    train, val, test = paper_split(labels, seed)
    model = SetFitModel.from_pretrained(MODEL)
    dataset = Dataset.from_dict({"text": [texts[i] for i in train],
                                 "label": [int(labels[i]) for i in train]})
    Trainer(model=model, args=TrainingArguments(batch_size=16, num_iterations=iterations,
                                                num_epochs=1, seed=seed),
            train_dataset=dataset).train()
    tuned = model.model_body.encode(texts, batch_size=64, show_progress_bar=False,
                                    convert_to_numpy=True)
    logistic = [LogisticRegression(C=c, max_iter=5000) for c in (0.01, 0.1, 1, 10, 100)]
    return best_on_validation(logistic, tuned, labels, train, val, test)


def summary(values):
    values = 100 * np.asarray(values)
    return {"mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max()), "n": int(len(values))}


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--setfit-seeds", type=int, default=5)
    args = ap.parse_args()

    print("=== experiment 036 :: the classical side of IonQ's quantum LLM head ===")
    print(f"paper: quantum {PAPER_QUANTUM}%, best classical {PAPER_CLASSICAL}%\n")

    texts, labels = load_corpus()
    print(f"{len(texts)} sentences; test split is {len(texts) - 2 * PER_CLASS}\n")
    embeddings = frozen_embeddings(texts)

    print("--- 1. classical heads on FROZEN embeddings, chosen on validation ---")
    frozen = {}
    for seed in range(args.seeds):
        train, val, test = paper_split(labels, seed)
        for name, models in heads(seed).items():
            frozen.setdefault(name, []).append(
                best_on_validation(models, embeddings, labels, train, val, test))
    for name, values in frozen.items():
        s = summary(values)
        print(f"  {name:<34} {s['mean']:6.2f} +- {s['std']:4.2f}   max {s['max']:6.2f}"
              f"   (n={s['n']})")

    print("\n--- 2. SetFit fine-tuning, then logistic regression ---")
    tuned = []
    for seed in range(args.setfit_seeds):
        started = time.perf_counter()
        tuned.append(setfit_arm(texts, labels, seed))
        print(f"  seed {seed}: {100 * tuned[-1]:6.2f}%  ({time.perf_counter() - started:.0f}s)",
              flush=True)
    s = summary(tuned)
    print(f"  SetFit + logistic regression       {s['mean']:6.2f} +- {s['std']:4.2f}"
          f"   max {s['max']:6.2f}   (n={s['n']})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp036_quantum_llm_head.json"
    with open(path, "w") as fh:
        json.dump({"paper": {"quantum": PAPER_QUANTUM, "classical": PAPER_CLASSICAL},
                   "frozen": {k: {"accuracies": v, **summary(v)} for k, v in frozen.items()},
                   "setfit_logistic": {"accuracies": tuned, **summary(tuned),
                                       "iterations": SETFIT_ITERATIONS}},
                  fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
