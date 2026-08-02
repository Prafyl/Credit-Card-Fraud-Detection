"""Reproduce the LightGBM hyperparameter search behind models.TUNED_LIGHTGBM.

Run:  python src/tune.py

Method (leakage-safe): fit each candidate on TRAIN, score PR-AUC on VALIDATION, keep the
best. The test set is never touched here — train.py reports the final numbers on it.

This is a plain randomised search rather than a grid: with 6 hyperparameters a full grid is
thousands of fits, while 40 random draws find a near-best config in a couple of minutes and
are just as easy to explain.
"""

import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from data import RANDOM_STATE, get_splits

N_CANDIDATES = 40

# Search space. Note num_leaves stays small: the winning pattern is many shallow trees.
SEARCH_SPACE = {
    "n_estimators": [300, 500, 800, 1500, 2000],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 20, 30, 50],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.0, 0.5, 1.0, 2.0],
}


def score(params: dict, Xtr, y_train, Xva, y_val) -> float:
    model = LGBMClassifier(
        **params, class_weight="balanced", n_jobs=-1, verbose=-1,
        random_state=RANDOM_STATE
    )
    model.fit(Xtr, y_train)
    return average_precision_score(y_val, model.predict_proba(Xva)[:, 1])


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

    # Scale once, reuse for every candidate (the scaler has no hyperparameters to tune).
    scaler = StandardScaler().set_output(transform="pandas")
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)

    rng = np.random.default_rng(RANDOM_STATE)
    results = []
    start = time.time()

    print(f"Searching {N_CANDIDATES} random LightGBM configs (val PR-AUC)...\n")
    for i in range(N_CANDIDATES):
        params = {k: rng.choice(v).item() for k, v in SEARCH_SPACE.items()}
        ap = score(params, Xtr, y_train, Xva, y_val)
        results.append((ap, params))
        print(f"  [{i:2d}] PR-AUC {ap:.4f}")

    results.sort(reverse=True, key=lambda r: r[0])
    print(f"\nDone in {time.time() - start:.0f}s. Best config (copy into models.TUNED_LIGHTGBM):\n")
    best_ap, best_params = results[0]
    print(f"  val PR-AUC {best_ap:.4f}")
    for key, value in best_params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
