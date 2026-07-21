"""Compare four ways of handling the 0.167% fraud rate, for the champion LightGBM model.

Run:  python src/resampling_experiment.py

Reuses the exact split (data.get_splits), the exact tuned hyperparameters
(models.TUNED_LIGHTGBM), and the exact val-tuned/test-scored protocol
(train.best_threshold, train.evaluate) as the main comparison in train.py. Only the
resampling step changes across the four pipelines, so any difference in results is
attributable to resampling alone. The four candidates:

  * Class weighting     - the current champion's approach. No resampling; LightGBM's own
    class_weight="balanced" tells it frauds count more during training.
  * Random undersampling - drop legit rows until the classes are equal.
  * Random oversampling  - duplicate fraud rows until the classes are equal.
  * SMOTE                - synthesise new, slightly-varied fraud rows instead of
    duplicating them.

Every sampler lives INSIDE its pipeline (fit-time only) — the same rule models.py already
follows — so no resampled or synthetic row ever leaks into validation or test.

This script only writes to reports/. It does not touch models/, metadata.json, or the
served API in any way.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler

from data import RANDOM_STATE, get_splits
from models import TUNED_LIGHTGBM
from plotting import FRAUD, INK, LEGIT, ROOT, SURFACE, save
from train import best_threshold, evaluate  # noqa: F401 (best_threshold used inside evaluate)

REPORTS = ROOT / "reports"


def _lgbm(**extra) -> LGBMClassifier:
    return LGBMClassifier(**TUNED_LIGHTGBM, n_jobs=-1, verbose=-1,
                           random_state=RANDOM_STATE, **extra)


# Each factory returns a fresh, unfitted pipeline — called once per strategy so no fitted
# state leaks between them.
STRATEGIES = {
    "Class weighting": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", _lgbm(class_weight="balanced")),
    ]),
    "Random undersampling": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("sample", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("clf", _lgbm()),
    ]),
    "Random oversampling": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("sample", RandomOverSampler(random_state=RANDOM_STATE)),
        ("clf", _lgbm()),
    ]),
    "SMOTE": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("sample", SMOTE(random_state=RANDOM_STATE)),
        ("clf", _lgbm()),
    ]),
}


def describe_resampling(pipeline: Pipeline, X_train, y_train) -> str:
    """Run just the pre-classifier steps, so the printed log shows what each sampler
    actually did to the training set (e.g. how many legit rows undersampling discarded)."""
    if len(pipeline.steps) == 2:  # scale + clf only -> no sampler in this pipeline
        return (f"no resampling - kept all {len(y_train):,} rows "
                f"({int(y_train.sum())} fraud, {int((y_train == 0).sum()):,} legit)")
    _, y_resampled = pipeline[:-1].fit_resample(X_train, y_train)
    y_resampled = np.asarray(y_resampled)
    return (f"{len(y_resampled):,} rows after resampling "
            f"({int(y_resampled.sum()):,} fraud, "
            f"{int((y_resampled == 0).sum()):,} legit)")


def plot_pr_curves(curves: dict[str, tuple]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = [LEGIT, FRAUD, "#c98a2b", "#7a5cc4"]
    for (name, (precision, recall)), colour in zip(curves.items(), palette):
        ax.plot(recall, precision, label=name, color=colour, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall by resampling strategy (LightGBM, test set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    save(fig, "09_resampling_pr_curves")


def plot_confusion_matrices(matrices: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
    cmap = LinearSegmentedColormap.from_list("surface_legit", [SURFACE, LEGIT])
    for ax, (name, cm) in zip(axes.ravel(), matrices.items()):
        ax.imshow(cm, cmap=cmap, vmin=0)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else INK, fontsize=11)
        ax.set_xticks([0, 1], ["Pred legit", "Pred fraud"])
        ax.set_yticks([0, 1], ["True legit", "True fraud"])
        ax.set_title(name, fontsize=11)
        ax.grid(False)
    fig.suptitle("Confusion matrix by resampling strategy (test set, tuned threshold)")
    fig.tight_layout()
    save(fig, "10_resampling_confusion_matrices")


def main() -> None:
    print("Loading data (same split as train.py)...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

    results, pr_curves, matrices = [], {}, {}

    for name, build in STRATEGIES.items():
        print(f"\n{name}")
        print(f"  {describe_resampling(build(), X_train, y_train)}")

        pipeline = build()
        pipeline.set_output(transform="pandas")
        pipeline.fit(X_train, y_train)

        val_proba = pipeline.predict_proba(X_val)[:, 1]
        test_proba = pipeline.predict_proba(X_test)[:, 1]

        row = evaluate(name, y_val, val_proba, y_test, test_proba)
        results.append(row)
        print(f"  PR-AUC {row['PR_AUC']:.3f}  recall {row['recall']:.3f}  "
              f"precision {row['precision']:.3f}")

        precision, recall, _ = precision_recall_curve(y_test, test_proba)
        pr_curves[name] = (precision, recall)
        y_pred = (test_proba >= row["threshold"]).astype(int)
        matrices[name] = confusion_matrix(y_test, y_pred)

    table = pd.DataFrame(results).sort_values("PR_AUC", ascending=False).reset_index(drop=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "resampling_comparison.csv", index=False)

    print("\nResampling comparison (sorted by PR-AUC):\n")
    show = table.drop(columns=["threshold"]).copy()
    for col in ["PR_AUC", "ROC_AUC", "precision", "recall", "f1"]:
        show[col] = show[col].map(lambda v: f"{v:.3f}")
    print(show.to_string(index=False))

    plot_pr_curves(pr_curves)
    plot_confusion_matrices(matrices)


if __name__ == "__main__":
    main()
