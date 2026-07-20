"""Deeper statistical analysis of the already-trained models, for the report.

Run:  python src/deep_analysis.py            (full run, several minutes)
      python src/deep_analysis.py --fast     (skips cross-validation + learning curve)

Everything the main comparison reports comes from a single fixed train/val/test split with
no error bars. That is enough to pick a winner, but not enough to say the winner is
*significantly* better, whether its probabilities mean anything, or what it would cost a
bank to run. This script answers those questions.

It RELOADS the eight pipelines from models/registry/ rather than retraining them, and
reuses the same split (data.get_splits) and the same tuned thresholds (models/metadata.json)
as train.py, so every number here is directly comparable to model_comparison.csv.

Nothing here writes to models/ or changes the served API. Outputs are CSVs and figures
11-19 under reports/.
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from data import FEATURES, TARGET, get_splits, load_data
from plotting import DIVERGING, FRAUD, INK_SECONDARY, LEGIT, MUTED, ROOT, save

REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"
REGISTRY = MODELS_DIR / "registry"

RANDOM_STATE = 42
N_BOOTSTRAP = 1000        # resamples for the PR-AUC confidence interval
FP_REVIEW_COST = 3.0      # EUR to manually review one false alarm (assumption, stated in report)

# Palette for multi-model charts. Distinct hues, dark enough to read on the light figure
# surface used by plotting.py.
SERIES = ["#2a78d6", "#e34948", "#c98a2b", "#7a5cc4",
          "#2f9e8f", "#b8478f", "#5b8c2a", "#8a6a4f"]


# ======================================================================================
# Loading
# ======================================================================================

def load_artifacts():
    """Reload the fitted pipelines and the metadata written by train.py."""
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    entries = meta["models"]
    models = {name: joblib.load(REGISTRY / f"{e['slug']}.joblib")
              for name, e in entries.items()}
    return meta, entries, models


def score_all(models: dict, X_test) -> dict[str, np.ndarray]:
    """Fraud probability on the test set for every model, computed once and reused."""
    probas = {}
    for name, model in models.items():
        start = time.time()
        probas[name] = model.predict_proba(X_test)[:, 1]
        print(f"  scored {name:32s} ({time.time() - start:.1f}s)")
    return probas


# ======================================================================================
# 1. Extended metrics
# ======================================================================================

def confusion_counts(y_true, proba, threshold):
    pred = proba >= threshold
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    tn = int((~pred & (y_true == 0)).sum())
    return tp, fp, fn, tn


def extended_metrics(y_test, probas, entries) -> pd.DataFrame:
    """Everything derivable from the confusion matrix that the main table omits.

    The headline table reports precision/recall/F1. A fraud team also cares about how many
    alerts per 1,000 transactions it has to staff for, and MCC is the one summary number
    that stays honest under extreme imbalance (it uses all four cells).
    """
    rows = []
    for name, proba in probas.items():
        t = entries[name]["threshold"]
        tp, fp, fn, tn = confusion_counts(y_test, proba, t)
        pred = (proba >= t).astype(int)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        npv = tn / (tn + fn) if tn + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0

        rows.append({
            "model": name,
            "threshold": t,
            "PR_AUC": average_precision_score(y_test, proba),
            "ROC_AUC": roc_auc_score(y_test, proba),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "NPV": npv,
            "FPR": fpr,
            "balanced_accuracy": (recall + specificity) / 2,
            "youden_J": recall + specificity - 1,
            "MCC": matthews_corrcoef(y_test, pred),
            "alerts_per_1000": (tp + fp) / len(y_test) * 1000,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    return pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).reset_index(drop=True)


# ======================================================================================
# 2. Confidence intervals
# ======================================================================================

def wilson_interval(successes: int, total: int, conf: float = 0.95):
    """Wilson score interval for a proportion.

    Preferred over the textbook normal approximation because precision and recall here are
    based on very few positives (95 frauds), where the normal interval can run past 1.0.
    """
    if total == 0:
        return (np.nan, np.nan)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - conf) / 2)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson(successes: int, total: int, conf: float = 0.95):
    """Exact binomial interval, reported alongside Wilson as a conservative cross-check."""
    if total == 0:
        return (np.nan, np.nan)
    alpha = 1 - conf
    lo = beta.ppf(alpha / 2, successes, total - successes + 1) if successes > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, successes + 1, total - successes) if successes < total else 1.0
    return (float(lo), float(hi))


def bootstrap_pr_auc(y_true, proba, n_resamples: int, rng) -> tuple:
    """Percentile bootstrap CI for PR-AUC.

    PR-AUC has no closed-form standard error, so the interval is obtained by resampling the
    test set with replacement. Stratification is not imposed: the point is to capture the
    variability that comes from having only 95 frauds in the sample.
    """
    n = len(y_true)
    y_true = np.asarray(y_true)
    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        y_boot = y_true[idx]
        if y_boot.sum() == 0:            # degenerate resample, retry
            scores[i] = np.nan
            continue
        scores[i] = average_precision_score(y_boot, proba[idx])
    scores = scores[~np.isnan(scores)]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def confidence_intervals(y_test, probas, entries) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name, proba in probas.items():
        t = entries[name]["threshold"]
        tp, fp, fn, tn = confusion_counts(y_test, proba, t)
        prec_lo, prec_hi = wilson_interval(tp, tp + fp)
        rec_lo, rec_hi = wilson_interval(tp, tp + fn)
        cp_rec_lo, cp_rec_hi = clopper_pearson(tp, tp + fn)

        start = time.time()
        pr_lo, pr_hi = bootstrap_pr_auc(y_test, proba, N_BOOTSTRAP, rng)
        rows.append({
            "model": name,
            "PR_AUC": average_precision_score(y_test, proba),
            "PR_AUC_lo95": pr_lo, "PR_AUC_hi95": pr_hi,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "precision_lo95": prec_lo, "precision_hi95": prec_hi,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "recall_lo95": rec_lo, "recall_hi95": rec_hi,
            "recall_cp_lo95": cp_rec_lo, "recall_cp_hi95": cp_rec_hi,
        })
        print(f"  {name:32s} PR-AUC 95% CI [{pr_lo:.3f}, {pr_hi:.3f}] "
              f"({time.time() - start:.0f}s)")
    return pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).reset_index(drop=True)


def plot_ci_forest(ci: pd.DataFrame) -> None:
    """Forest plot: point estimate plus 95% bootstrap interval, ranked."""
    ranked = ci.sort_values("PR_AUC")
    y = np.arange(len(ranked))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hlines(y, ranked["PR_AUC_lo95"], ranked["PR_AUC_hi95"],
              color=MUTED, linewidth=2)
    ax.plot(ranked["PR_AUC"], y, "o", color=LEGIT, markersize=7)
    ax.plot(ranked["PR_AUC"].iloc[-1], y[-1], "o", color=FRAUD, markersize=9)
    ax.set_yticks(y, ranked["model"])
    ax.set_xlabel("PR-AUC with 95% bootstrap confidence interval")
    ax.set_title("The top models' intervals overlap heavily")
    ax.grid(axis="y", visible=False)
    save(fig, "11_pr_auc_confidence_intervals")


# ======================================================================================
# 3. Significance testing (McNemar)
# ======================================================================================

def mcnemar_tests(y_test, probas, entries) -> pd.DataFrame:
    """Pairwise McNemar exact test on the models' correctness patterns.

    Two models are compared on the SAME transactions, so their errors are paired and an
    unpaired test would be wrong. McNemar looks only at the discordant cases: transactions
    where exactly one of the two models is right. Under the null "both models are equally
    accurate", those discordant cases split 50/50, which is an exact binomial test.

    b = A right, B wrong;  c = A wrong, B right.  p from binomtest(b, b+c, 0.5).
    """
    correct = {}
    for name, proba in probas.items():
        pred = (proba >= entries[name]["threshold"]).astype(int)
        correct[name] = (pred == np.asarray(y_test))

    rows = []
    for a, b_name in combinations(correct, 2):
        ca, cb = correct[a], correct[b_name]
        b = int((ca & ~cb).sum())     # a right, b wrong
        c = int((~ca & cb).sum())     # a wrong, b right
        n_disc = b + c
        p = binomtest(b, n_disc, 0.5).pvalue if n_disc > 0 else 1.0
        rows.append({
            "model_a": a, "model_b": b_name,
            "a_right_b_wrong": b, "a_wrong_b_right": c,
            "n_discordant": n_disc, "p_value": p,
        })

    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    # Holm-Bonferroni correction: 28 pairwise tests inflate the false-positive rate, and
    # Holm controls the family-wise error rate without being as blunt as plain Bonferroni.
    m = len(out)
    adj, running_max = [], 0.0
    for i, p in enumerate(out["p_value"]):
        val = min(1.0, (m - i) * p)
        running_max = max(running_max, val)   # enforce monotonicity
        adj.append(running_max)
    out["p_holm"] = adj
    out["significant_5pct"] = out["p_holm"] < 0.05
    return out


def plot_mcnemar_heatmap(tests: pd.DataFrame, order: list) -> None:
    n = len(order)
    grid = np.full((n, n), np.nan)
    idx = {name: i for i, name in enumerate(order)}
    for _, r in tests.iterrows():
        i, j = idx[r["model_a"]], idx[r["model_b"]]
        grid[i, j] = grid[j, i] = r["p_holm"]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(grid, cmap=DIVERGING.reversed(), vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color=INK_SECONDARY)
    short = [m.replace("LogisticRegression", "LogReg").replace("RandomForest", "RF")
             for m in order]
    ax.set_xticks(range(n), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), short, fontsize=8)
    ax.set_title("McNemar test, Holm-adjusted p-values\n(low = the two models differ)")
    ax.grid(visible=False)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label("adjusted p-value")
    bar.outline.set_visible(False)
    save(fig, "12_mcnemar_significance")


# ======================================================================================
# 4. Calibration
# ======================================================================================

def calibration_table(y_test, probas, n_bins: int = 10) -> pd.DataFrame:
    """Are the predicted probabilities honest? Bin them and compare to observed frequency.

    A model can rank perfectly (high PR-AUC) and still be badly calibrated, which matters
    if the score is ever used as an expected loss rather than just a ranking.
    Expected Calibration Error is the bin-count-weighted mean gap between the two.
    """
    y_test = np.asarray(y_test)
    edges = np.linspace(0, 1, n_bins + 1)
    frames = []
    for name, proba in probas.items():
        bins = np.clip(np.digitize(proba, edges) - 1, 0, n_bins - 1)
        model_rows, ece = [], 0.0
        for b in range(n_bins):
            mask = bins == b
            count = int(mask.sum())
            if count == 0:
                continue
            mean_pred = float(proba[mask].mean())
            observed = float(y_test[mask].mean())
            ece += count / len(y_test) * abs(mean_pred - observed)
            model_rows.append({
                "model": name, "bin": b,
                "bin_lo": edges[b], "bin_hi": edges[b + 1],
                "count": count, "mean_predicted": mean_pred, "observed_rate": observed,
            })
        # ECE and Brier are per-model scalars; repeated on each bin row so the CSV is
        # self-contained and one groupby("model").first() recovers them.
        brier = brier_score_loss(y_test, proba)
        for r in model_rows:
            r["ECE"] = ece
            r["brier"] = brier
        frames.extend(model_rows)
    return pd.DataFrame(frames)


def plot_calibration(y_test, probas) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, label="Perfectly calibrated")
    for (name, proba), colour in zip(probas.items(), SERIES):
        edges = np.linspace(0, 1, 11)
        bins = np.clip(np.digitize(proba, edges) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            mask = bins == b
            if mask.sum() < 5:
                continue
            xs.append(proba[mask].mean())
            ys.append(np.asarray(y_test)[mask].mean())
        ax.plot(xs, ys, "o-", color=colour, linewidth=1.6, markersize=4,
                label=name.replace("LogisticRegression", "LogReg")
                          .replace("RandomForest", "RF"))
    ax.set_xlabel("Mean predicted fraud probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Reliability curves: predicted vs actual")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "13_calibration_curves")


# ======================================================================================
# 5. ROC and PR curves
# ======================================================================================

def plot_roc_pr(y_test, probas) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5))
    for (name, proba), colour in zip(probas.items(), SERIES):
        label = (name.replace("LogisticRegression", "LogReg")
                     .replace("RandomForest", "RF"))
        fpr, tpr, _ = roc_curve(y_test, proba)
        left.plot(fpr, tpr, color=colour, linewidth=1.7, label=label)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        right.plot(recall, precision, color=colour, linewidth=1.7, label=label)

    left.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
    left.set_xlabel("False positive rate")
    left.set_ylabel("True positive rate")
    left.set_title("ROC: flattering, all curves hug the corner")

    right.axhline(float(np.mean(y_test)), linestyle="--", color=MUTED, linewidth=1)
    right.set_xlabel("Recall")
    right.set_ylabel("Precision")
    right.set_title("Precision-recall: where the models actually separate")
    right.set_ylim(0, 1.02)
    right.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "14_roc_and_pr_curves")


# ======================================================================================
# 6. Feature importance
# ======================================================================================

def feature_importance(champion, champion_name, X_test, y_test) -> pd.DataFrame:
    """Two views of importance for the champion.

    Gain is what the model itself used when splitting. Permutation importance measures what
    actually degrades held-out PR-AUC when a column is shuffled, which is the more honest
    of the two because it is measured on data the model never saw.
    """
    clf = champion.named_steps["clf"]
    booster = clf.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    print("  running permutation importance (29 features x 10 repeats)...")
    start = time.time()
    perm = permutation_importance(
        champion, X_test, y_test, scoring="average_precision",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    print(f"    done ({time.time() - start:.0f}s)")

    out = pd.DataFrame({
        "feature": FEATURES,
        "gain": gain,
        "split_count": split,
        "permutation_mean": perm.importances_mean,
        "permutation_std": perm.importances_std,
    })
    out["gain_pct"] = out["gain"] / out["gain"].sum() * 100
    out["model"] = champion_name
    return out.sort_values("permutation_mean", ascending=False).reset_index(drop=True)


def plot_feature_importance(imp: pd.DataFrame) -> None:
    top = imp.head(15).iloc[::-1]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)

    left.barh(top["feature"], top["permutation_mean"],
              xerr=top["permutation_std"], color=LEGIT, height=0.7,
              error_kw={"ecolor": MUTED, "elinewidth": 1})
    left.set_xlabel("Drop in test PR-AUC when shuffled")
    left.set_title("Permutation importance (held-out)")
    left.grid(axis="y", visible=False)

    right.barh(top["feature"], top["gain_pct"], color=LEGIT, height=0.7)
    right.set_xlabel("Share of total split gain (%)")
    right.set_title("LightGBM gain (in-model)")
    right.grid(axis="y", visible=False)

    fig.suptitle("Top 15 features for the champion model",
                 fontsize=13, fontweight="semibold")
    fig.tight_layout()
    save(fig, "15_feature_importance")


# ======================================================================================
# 7. Cost analysis
# ======================================================================================

def cost_analysis(y_test, proba, amounts, tuned_threshold) -> pd.DataFrame:
    """What does each operating point cost in euros?

    Counting a missed fraud and a false alarm as equally bad (what train.py's threshold
    search does) is a modelling convenience, not a business fact. Here a missed fraud costs
    the transaction amount that was lost, and a false alarm costs a fixed manual review.
    That turns the threshold into a money decision.
    """
    y_test = np.asarray(y_test)
    amounts = np.asarray(amounts)
    rows = []
    for t in np.arange(0.0, 1.0 + 0.005, 0.005):
        pred = proba >= t
        missed = (~pred) & (y_test == 1)
        false_alarm = pred & (y_test == 0)
        fraud_loss = float(amounts[missed].sum())
        review_cost = float(false_alarm.sum() * FP_REVIEW_COST)
        rows.append({
            "threshold": round(float(t), 4),
            "missed_frauds": int(missed.sum()),
            "false_alarms": int(false_alarm.sum()),
            "fraud_loss_eur": fraud_loss,
            "review_cost_eur": review_cost,
            "total_cost_eur": fraud_loss + review_cost,
        })
    out = pd.DataFrame(rows)
    out["is_cost_optimal"] = out["total_cost_eur"] == out["total_cost_eur"].min()
    out["is_tuned_threshold"] = np.isclose(out["threshold"], tuned_threshold, atol=0.0025)
    return out


def plot_cost(cost: pd.DataFrame, tuned_threshold: float) -> None:
    best = cost.loc[cost["total_cost_eur"].idxmin()]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(cost["threshold"], cost["fraud_loss_eur"], color=FRAUD,
            linewidth=1.8, label="Fraud losses (missed frauds)")
    ax.plot(cost["threshold"], cost["review_cost_eur"], color=LEGIT,
            linewidth=1.8, label=f"Review costs (EUR {FP_REVIEW_COST:.0f} per false alarm)")
    ax.plot(cost["threshold"], cost["total_cost_eur"], color=INK_SECONDARY,
            linewidth=2.4, label="Total cost")

    ax.axvline(best["threshold"], color=INK_SECONDARY, linestyle=":", linewidth=1.4)
    ax.text(best["threshold"], ax.get_ylim()[1] * 0.92,
            f" cost-optimal {best['threshold']:.3f}\n EUR {best['total_cost_eur']:,.0f}",
            fontsize=9, color=INK_SECONDARY, va="top")
    ax.axvline(tuned_threshold, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(tuned_threshold, ax.get_ylim()[1] * 0.55,
            f" error-count\n threshold {tuned_threshold:.3f}",
            fontsize=9, color=MUTED, va="top")

    gap = cost.loc[cost["is_tuned_threshold"], "total_cost_eur"].iloc[0] - best["total_cost_eur"]
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Cost on the test set (EUR)")
    ax.set_title(f"Total cost is flat near the optimum: the tuned threshold\n"
                 f"costs only EUR {gap:.0f} more than the cheapest point")
    ax.legend(fontsize=9)
    save(fig, "16_cost_analysis")


# ======================================================================================
# 8. Cross-validation
# ======================================================================================

def cross_validation(top_names, X_all, y_all, n_splits: int = 5) -> pd.DataFrame:
    """Refit the leading configurations across 5 stratified folds.

    The headline numbers come from one split. This asks whether the ranking survives being
    re-run on different data, by reporting a standard deviation next to each mean.
    """
    from models import build_models

    n_pos = int(y_all.sum())
    scale_pos_weight = (len(y_all) - n_pos) / n_pos
    available = build_models(scale_pos_weight)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name in top_names:
        if name not in available:
            print(f"    skipping {name}: not in the model grid")
            continue
        for fold, (tr, te) in enumerate(cv.split(X_all, y_all)):
            start = time.time()
            # A fresh pipeline per fold, so no fitted state carries over between folds.
            model = build_models(scale_pos_weight)[name]
            model.fit(X_all.iloc[tr], y_all.iloc[tr])
            proba = model.predict_proba(X_all.iloc[te])[:, 1]
            score = average_precision_score(y_all.iloc[te], proba)
            rows.append({"model": name, "fold": fold, "PR_AUC": score})
            print(f"    {name:28s} fold {fold + 1}/{n_splits}  PR-AUC {score:.3f}  "
                  f"({time.time() - start:.0f}s)")
    return pd.DataFrame(rows)


def plot_cv(cv: pd.DataFrame) -> None:
    order = (cv.groupby("model")["PR_AUC"].mean().sort_values().index.tolist())
    data = [cv.loc[cv["model"] == m, "PR_AUC"].values for m in order]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor(LEGIT)
        patch.set_alpha(0.35)
        patch.set_edgecolor(INK_SECONDARY)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(INK_SECONDARY)
    ax.set_yticks(range(1, len(order) + 1), order)
    ax.set_xlabel("PR-AUC across 5 stratified folds")
    ax.set_title("Fold-to-fold spread overlaps the gaps between models")
    ax.grid(axis="y", visible=False)
    save(fig, "17_cross_validation")


# ======================================================================================
# 9. Learning curve
# ======================================================================================

def learning_curve_analysis(champion_name, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Would more data help? Refit the champion on growing subsets of the training set."""
    from models import build_models

    fractions = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    rng = np.random.default_rng(RANDOM_STATE)
    n_pos = int(y_train.sum())
    rows = []
    for frac in fractions:
        n = int(len(y_train) * frac)
        idx = rng.choice(len(y_train), n, replace=False)
        Xs, ys = X_train.iloc[idx], y_train.iloc[idx]
        if ys.sum() < 5:
            continue
        model = build_models((len(ys) - int(ys.sum())) / max(1, int(ys.sum())))[champion_name]
        start = time.time()
        model.fit(Xs, ys)
        score = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
        rows.append({"fraction": frac, "n_train": n, "n_frauds": int(ys.sum()),
                     "test_PR_AUC": score})
        print(f"    n={n:>7,} ({frac:>5.0%})  frauds={int(ys.sum()):>3}  "
              f"PR-AUC {score:.3f}  ({time.time() - start:.0f}s)")
    return pd.DataFrame(rows)


def plot_learning_curve(lc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lc["n_train"], lc["test_PR_AUC"], "o-", color=LEGIT, linewidth=2)
    for _, r in lc.iterrows():
        ax.text(r["n_train"], r["test_PR_AUC"] + 0.012,
                f"{r['test_PR_AUC']:.3f}", ha="center", fontsize=8, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlabel("Training transactions (log scale)")
    ax.set_ylabel("Test PR-AUC")
    ax.set_title("Returns flatten well before the full training set")
    save(fig, "18_learning_curve")


# ======================================================================================
# 10. Lift and cumulative gains
# ======================================================================================

def lift_table(y_test, proba, n_deciles: int = 10) -> pd.DataFrame:
    """If an analyst could only review the top X% of scored transactions, what would they catch?"""
    y_test = np.asarray(y_test)
    order = np.argsort(-proba)
    y_sorted = y_test[order]
    total_frauds = int(y_test.sum())
    n = len(y_test)

    rows = []
    for d in range(1, n_deciles + 1):
        cut = int(n * d / n_deciles)
        caught = int(y_sorted[:cut].sum())
        rows.append({
            "decile": d,
            "share_reviewed": d / n_deciles,
            "n_reviewed": cut,
            "frauds_caught": caught,
            "cumulative_recall": caught / total_frauds,
            "lift": (caught / cut) / (total_frauds / n),
        })
    return pd.DataFrame(rows)


def plot_lift(lift: pd.DataFrame) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.concatenate([[0], lift["share_reviewed"] * 100])
    y = np.concatenate([[0], lift["cumulative_recall"] * 100])
    left.plot(x, y, "o-", color=LEGIT, linewidth=2, label="Model")
    left.plot([0, 100], [0, 100], "--", color=MUTED, linewidth=1, label="Random review")
    left.set_xlabel("Share of transactions reviewed (%)")
    left.set_ylabel("Share of frauds caught (%)")
    left.set_title("Cumulative gains")
    left.legend(fontsize=9)

    right.bar(lift["decile"], lift["lift"], color=LEGIT, width=0.65)
    right.axhline(1, color=MUTED, linestyle="--", linewidth=1)
    right.set_xlabel("Decile of transactions, ranked by fraud score")
    right.set_ylabel("Lift over random")
    right.set_title("Lift by decile")
    right.set_yscale("log")
    right.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "19_lift_and_gains")


# ======================================================================================
# 11. Dataset profile
# ======================================================================================

def dataset_profile(df: pd.DataFrame) -> pd.DataFrame:
    """The EDA numbers as data, so the report can cite them instead of reading a chart."""
    frauds = df[df[TARGET] == 1]
    legit = df[df[TARGET] == 0]
    hour = (df["Time"] // 3600) % 24
    rate_by_hour = df.groupby(hour)[TARGET].mean() * 100
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)

    stats = [
        ("rows_after_dedup", len(df)),
        ("duplicates_removed", 1081),
        ("fraud_count", int(df[TARGET].sum())),
        ("legit_count", int((df[TARGET] == 0).sum())),
        ("fraud_rate_pct", df[TARGET].mean() * 100),
        ("imbalance_ratio", (df[TARGET] == 0).sum() / max(1, int(df[TARGET].sum()))),
        ("missing_values", int(df.isna().sum().sum())),
        ("n_features_modelled", len(FEATURES)),
        ("amount_median_legit", legit["Amount"].median()),
        ("amount_median_fraud", frauds["Amount"].median()),
        ("amount_mean_legit", legit["Amount"].mean()),
        ("amount_mean_fraud", frauds["Amount"].mean()),
        ("amount_p95_legit", legit["Amount"].quantile(0.95)),
        ("amount_p95_fraud", frauds["Amount"].quantile(0.95)),
        ("amount_max_fraud", frauds["Amount"].max()),
        ("peak_fraud_hour", int(rate_by_hour.idxmax())),
        ("peak_fraud_rate_pct", rate_by_hour.max()),
        ("min_fraud_rate_pct", rate_by_hour.min()),
        ("top_feature", corr.index[0]),
        ("top_feature_abs_corr", corr.iloc[0]),
        ("second_feature", corr.index[1]),
        ("second_feature_abs_corr", corr.iloc[1]),
    ]
    return pd.DataFrame(stats, columns=["statistic", "value"])


# ======================================================================================
# Main
# ======================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="skip cross-validation and the learning curve")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading split and trained models...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
    meta, entries, models = load_artifacts()
    champion_name = meta["default_model"]
    print(f"  champion: {champion_name}")
    print(f"  test set: {len(y_test):,} rows, {int(y_test.sum())} frauds\n")

    print("Scoring the test set with all 8 models...")
    probas = score_all(models, X_test)

    # --- sanity check against the published numbers -----------------------------------
    tp, fp, fn, tn = confusion_counts(y_test, probas[champion_name],
                                      entries[champion_name]["threshold"])
    pr = average_precision_score(y_test, probas[champion_name])
    print(f"\nReproduction check on {champion_name}:")
    print(f"  PR-AUC {pr:.4f} (expected 0.8277)   TP/FP/FN/TN {tp}/{fp}/{fn}/{tn} "
          f"(expected 70/2/25/56649)")
    assert abs(pr - 0.8277) < 5e-4, "PR-AUC does not match model_comparison.csv"
    assert (tp, fp, fn, tn) == (70, 2, 25, 56649), "confusion matrix does not match"
    print("  OK - reloaded models reproduce the reported results exactly.\n")

    # --- 1. extended metrics ----------------------------------------------------------
    print("[1/11] Extended metrics")
    ext = extended_metrics(y_test, probas, entries)
    ext.to_csv(REPORTS / "extended_metrics.csv", index=False)
    print(ext[["model", "MCC", "specificity", "balanced_accuracy",
               "alerts_per_1000"]].round(4).to_string(index=False))

    # --- 2. confidence intervals ------------------------------------------------------
    print(f"\n[2/11] Bootstrap confidence intervals ({N_BOOTSTRAP} resamples each)")
    ci = confidence_intervals(y_test, probas, entries)
    ci.to_csv(REPORTS / "confidence_intervals.csv", index=False)
    plot_ci_forest(ci)

    # --- 3. significance --------------------------------------------------------------
    print("\n[3/11] McNemar pairwise significance tests")
    tests = mcnemar_tests(y_test, probas, entries)
    tests.to_csv(REPORTS / "significance_tests.csv", index=False)
    n_sig = int(tests["significant_5pct"].sum())
    print(f"  {n_sig} of {len(tests)} pairs differ significantly after Holm correction")
    plot_mcnemar_heatmap(tests, list(ext["model"]))

    # --- 4. calibration ---------------------------------------------------------------
    print("\n[4/11] Calibration")
    cal = calibration_table(y_test, probas)
    cal.to_csv(REPORTS / "calibration.csv", index=False)
    summary = cal.groupby("model")[["ECE", "brier"]].first().sort_values("ECE")
    print(summary.round(5).to_string())
    plot_calibration(y_test, probas)

    # --- 5. ROC and PR curves ---------------------------------------------------------
    print("\n[5/11] ROC and precision-recall curves")
    plot_roc_pr(y_test, probas)

    # --- 6. feature importance --------------------------------------------------------
    print("\n[6/11] Feature importance")
    imp = feature_importance(models[champion_name], champion_name, X_test, y_test)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)
    print(imp.head(5)[["feature", "permutation_mean", "gain_pct"]].round(4).to_string(index=False))
    plot_feature_importance(imp)

    # --- 7. cost analysis --------------------------------------------------------------
    print("\n[7/11] Cost analysis")
    amounts = X_test["Amount"].values
    cost = cost_analysis(y_test, probas[champion_name], amounts,
                         entries[champion_name]["threshold"])
    cost.to_csv(REPORTS / "cost_analysis.csv", index=False)
    best = cost.loc[cost["total_cost_eur"].idxmin()]
    tuned_row = cost.loc[cost["is_tuned_threshold"]].iloc[0]
    print(f"  cost-optimal threshold {best['threshold']:.3f} -> EUR {best['total_cost_eur']:,.0f}")
    print(f"  tuned threshold        {tuned_row['threshold']:.3f} -> "
          f"EUR {tuned_row['total_cost_eur']:,.0f}")
    plot_cost(cost, entries[champion_name]["threshold"])

    # --- 8/9. slow sections -----------------------------------------------------------
    if args.fast:
        print("\n[8/11] Cross-validation  SKIPPED (--fast)")
        print("[9/11] Learning curve     SKIPPED (--fast)")
    else:
        print("\n[8/11] 5-fold cross-validation on the top 4 configurations")
        X_all = pd.concat([X_train, X_val])
        y_all = pd.concat([y_train, y_val])
        top4 = list(ext["model"].head(4))
        cv = cross_validation(top4, X_all, y_all)
        cv.to_csv(REPORTS / "cv_results.csv", index=False)
        agg = cv.groupby("model")["PR_AUC"].agg(["mean", "std"]).sort_values("mean",
                                                                            ascending=False)
        print(agg.round(4).to_string())
        plot_cv(cv)

        print("\n[9/11] Learning curve for the champion")
        lc = learning_curve_analysis(champion_name, X_train, y_train, X_test, y_test)
        lc.to_csv(REPORTS / "learning_curve.csv", index=False)
        plot_learning_curve(lc)

    # --- 10. lift ---------------------------------------------------------------------
    print("\n[10/11] Lift and cumulative gains")
    lift = lift_table(y_test, probas[champion_name])
    lift.to_csv(REPORTS / "lift_table.csv", index=False)
    print(lift[["decile", "share_reviewed", "cumulative_recall", "lift"]]
          .head(3).round(4).to_string(index=False))
    plot_lift(lift)

    # --- 11. dataset profile ----------------------------------------------------------
    print("\n[11/11] Dataset profile")
    profile = dataset_profile(load_data())
    profile.to_csv(REPORTS / "dataset_profile.csv", index=False)
    print(profile.to_string(index=False))

    print(f"\nDone in {time.time() - t0:.0f}s. CSVs and figures 11-19 written to reports/.")


if __name__ == "__main__":
    main()
