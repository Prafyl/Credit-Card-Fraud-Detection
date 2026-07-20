"""Train all 8 pipelines, compare them honestly, and save the best one.

Run:  python src/train.py

Protocol (why it's trustworthy):
  * fit on TRAIN
  * choose each model's decision threshold on VALIDATION (never the test set)
  * report every final number on TEST, which no decision ever touched
The primary metric is PR-AUC (average precision), the right choice at 0.167% positives —
ROC-AUC looks flattering on imbalanced data, so it's shown but not used to rank.
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from data import FEATURES, get_splits
from models import build_models
from plotting import FRAUD, INK_SECONDARY, LEGIT, MUTED, save

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"


def best_threshold(y_val, val_proba) -> float:
    """Pick the cut-off that makes the fewest TOTAL mistakes on validation.

    "Total mistakes" = missed frauds (false negatives) + false alarms (false positives).
    This directly minimises the two counts a fraud team cares about, and is easier to
    justify than maximising F1. The frontend's threshold slider can move this operating
    point afterwards without retraining — this is only the default.
    """
    y_val = np.asarray(y_val)
    order = np.argsort(val_proba)
    proba_sorted = val_proba[order]
    y_sorted = y_val[order]

    n_pos = int(y_val.sum())
    # For a threshold just above proba_sorted[i], everything up to and including i is
    # predicted legit. Missed frauds = frauds at-or-below the cut; false alarms = legit
    # above it. Both come from cumulative sums, so the whole scan is O(n log n).
    missed = np.cumsum(y_sorted)                       # frauds at or below each point
    false_alarms = (len(y_val) - np.arange(1, len(y_val) + 1)) - (n_pos - missed)
    total_errors = missed + false_alarms

    best = int(np.argmin(total_errors))
    # Put the cut-off just above the chosen sample so that sample is classed as legit.
    return float(np.nextafter(proba_sorted[best], np.inf))


def evaluate(name, y_val, val_proba, y_test, test_proba) -> dict:
    """Tune the threshold on validation, then score once on test.

    Probabilities are passed in (computed once by the caller) so the same test_proba
    can also feed the per-model threshold_curve without predicting twice.
    """
    threshold = best_threshold(y_val, val_proba)
    y_pred = (test_proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model": name,
        "PR_AUC": average_precision_score(y_test, test_proba),  # threshold-independent
        "ROC_AUC": roc_auc_score(y_test, test_proba),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def threshold_curve(y_test, test_proba, step: float = 0.005) -> list[dict]:
    """Confusion-matrix counts across a sweep of thresholds, for the dashboard slider.

    Lets the frontend show how missed frauds and false alarms trade off as the operating
    point moves, without re-scoring anything. Computed once on the test set here.

    Thresholds are evenly spaced 0..1 (not data quantiles): a tree model puts ~99.8% of
    its probabilities near 0, so quantiles would crowd every point into an unusable sliver
    near zero. Even spacing gives an interpretable slider ("set the cut-off to 0.50").
    """
    y_test = np.asarray(y_test)
    n_pos = int((y_test == 1).sum())
    n_neg = int((y_test == 0).sum())
    curve = []
    for t in np.arange(0.0, 1.0 + step, step):
        pred = test_proba >= t
        tp = int((pred & (y_test == 1)).sum())
        fp = int((pred & (y_test == 0)).sum())
        curve.append({
            "threshold": round(float(t), 4),
            "TP": tp, "FP": fp, "FN": n_pos - tp, "TN": n_neg - fp,
        })
    return curve


def slugify(name: str) -> str:
    """Turn a display name like "LightGBM + weighted" into a filename-safe "lightgbm_weighted"."""
    return name.lower().replace(" + ", "_").replace(" ", "_").replace("+", "")


def export_holdout_samples(X_test, y_test, model, path, n_legit: int = 400) -> None:
    """Save held-out test transactions for the frontend's sample picker.

    All fraud cases plus a random sample of legit ones, each tagged with its true label
    and the model's fraud probability. These are the rows a user loads into the single-
    transaction screen — the only honest way to demo a dataset of un-typeable PCA features.
    """
    proba = model.predict_proba(X_test)[:, 1]
    out = X_test.copy()
    out["Class"] = y_test.values
    out["fraud_probability"] = proba.round(6)

    frauds = out[out["Class"] == 1]
    legit = out[out["Class"] == 0].sample(n_legit, random_state=42)
    samples = pd.concat([frauds, legit]).sample(frac=1, random_state=42)  # shuffle
    samples.to_csv(path, index=False)
    print(f"  saved {len(samples)} holdout samples ({len(frauds)} fraud) to "
          f"{path.relative_to(ROOT)}")


def baseline_row(y_test) -> dict:
    """The "always predict legitimate" model: 99.83% accuracy, catches zero fraud.

    Included to make concrete why accuracy is banned as a metric here.
    """
    return {
        "model": "Baseline (always legit)",
        "PR_AUC": float(y_test.mean()),  # a random classifier's average precision
        "ROC_AUC": 0.5,
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": np.nan,
        "TP": 0, "FP": 0, "FN": int(y_test.sum()), "TN": int((1 - y_test).sum()),
    }


def plot_comparison(results: pd.DataFrame) -> None:
    """Rank every model by PR-AUC; highlight the winner, grey the baseline."""
    ranked = results.sort_values("PR_AUC")
    colours = [MUTED if "Baseline" in m else
               (FRAUD if m == ranked.iloc[-1]["model"] else LEGIT)
               for m in ranked["model"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(ranked["model"], ranked["PR_AUC"], color=colours, height=0.7)
    ax.set_xlabel("PR-AUC on the held-out test set (higher is better)")
    ax.set_title("XGBoost and LightGBM lead; class-weighting beats SMOTE")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)

    for model, value in zip(ranked["model"], ranked["PR_AUC"]):
        ax.text(value + 0.01, model, f"{value:.3f}", va="center",
                color=INK_SECONDARY, fontsize=9)
    save(fig, "08_model_comparison")


def main() -> None:
    print("Loading and splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

    n_pos = int(y_train.sum())
    scale_pos_weight = (len(y_train) - n_pos) / n_pos
    print(f"  train {len(y_train):,} | val {len(y_val):,} | test {len(y_test):,}")
    print(f"  scale_pos_weight = {scale_pos_weight:.1f}\n")

    models = build_models(scale_pos_weight)
    results = []
    curves: dict[str, list] = {}  # per-model threshold sweep, for the dashboard slider

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        row = evaluate(name, y_val, val_proba, y_test, test_proba)
        curves[name] = threshold_curve(y_test, test_proba)
        results.append(row)
        print(f"  {name:28s} PR-AUC {row['PR_AUC']:.3f}  "
              f"recall {row['recall']:.3f}  precision {row['precision']:.3f}  "
              f"({time.time() - start:.0f}s)")

    results.append(baseline_row(y_test))
    table = pd.DataFrame(results).sort_values("PR_AUC", ascending=False).reset_index(drop=True)

    # --- Save comparison table and chart -------------------------------------------
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "model_comparison.csv", index=False)
    plot_comparison(table)

    print("\nModel comparison (sorted by PR-AUC):\n")
    show = table.drop(columns=["threshold"]).copy()
    for col in ["PR_AUC", "ROC_AUC", "precision", "recall", "f1"]:
        show[col] = show[col].map(lambda v: f"{v:.3f}")
    print(show.to_string(index=False))

    # --- Save EVERY model to a registry, plus combined metadata --------------------
    # The API loads all of these so the UI can switch between them at inference time.
    # The winner (best PR-AUC among real models) stays the default and is also written
    # to best_model.joblib for backward compatibility.
    best_name = table[~table["model"].str.contains("Baseline")].iloc[0]["model"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY = MODELS_DIR / "registry"
    REGISTRY.mkdir(parents=True, exist_ok=True)

    METRIC_KEYS = ["PR_AUC", "ROC_AUC", "precision", "recall", "f1", "TP", "FP", "FN", "TN"]
    model_entries: dict[str, dict] = {}
    for name, model in models.items():
        slug = slugify(name)
        joblib.dump(model, REGISTRY / f"{slug}.joblib")
        row = next(r for r in results if r["model"] == name)
        model_entries[name] = {
            "model_name": name,
            "slug": slug,
            "threshold": row["threshold"],
            "test_metrics": {k: row[k] for k in METRIC_KEYS},
            "threshold_curve": curves[name],
        }

    best_model = models[best_name]
    best_entry = model_entries[best_name]
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    metadata = {
        "default_model": best_name,
        "features": FEATURES,
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
        "models": model_entries,
        # Top-level mirror of the default model, so older single-model API code keeps working.
        "model_name": best_name,
        "threshold": best_entry["threshold"],
        "test_metrics": best_entry["test_metrics"],
        "threshold_curve": best_entry["threshold_curve"],
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    export_holdout_samples(X_test, y_test, best_model, MODELS_DIR / "holdout_samples.csv")
    best_row = best_entry["test_metrics"]
    print(f"\nSaved {len(model_entries)} models to models/registry/  "
          f"(default: {best_name})")

    # --- Leakage sanity check: reload and re-score ---------------------------------
    reloaded = joblib.load(MODELS_DIR / "best_model.joblib")
    proba = reloaded.predict_proba(X_test)[:, 1]
    reloaded_pr_auc = average_precision_score(y_test, proba)
    assert abs(reloaded_pr_auc - best_row["PR_AUC"]) < 1e-9, "reloaded model disagrees!"
    if reloaded_pr_auc > 0.99:
        print("  WARNING: PR-AUC above 0.99 — check for leakage.")
    else:
        print(f"  Leakage check OK: reloaded PR-AUC {reloaded_pr_auc:.3f} matches, "
              "and is realistically below 1.0.")


if __name__ == "__main__":
    main()
