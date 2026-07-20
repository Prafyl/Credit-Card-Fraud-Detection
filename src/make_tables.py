"""Generate every LaTeX table in the report from the analysis CSVs.

Run:  python src/make_tables.py

The report never hard-codes a number. Each table below is written to
reports/latex/tables/*.tex and pulled into the document with \\input{}, so the figures in
the text cannot drift away from what the code actually produced. Re-run this after any
re-training or re-analysis and the report updates itself.

Also copies reports/figures/*.png into reports/latex/figures/ so the LaTeX project is a
self-contained folder that can be zipped and uploaded to Overleaf as-is.
"""

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
LATEX = REPORTS / "latex"
TABLES = LATEX / "tables"
FIGURES_SRC = REPORTS / "figures"
FIGURES_DST = LATEX / "figures"

SHORT = {
    "LogisticRegression": "LogReg",
    "RandomForest": "Forest",
}


def shorten(name: str) -> str:
    """Model names are long; LaTeX tables are narrow."""
    for full, short in SHORT.items():
        name = name.replace(full, short)
    return name.replace(" + ", " + ")


def escape(text: str) -> str:
    """Escape the LaTeX specials that appear in our data (mainly _ and %)."""
    return (str(text).replace("\\", r"\textbackslash{}")
                     .replace("_", r"\_")
                     .replace("%", r"\%")
                     .replace("&", r"\&"))


def write_table(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(body, encoding="utf-8")
    print(f"  wrote tables/{name}.tex")


def tabular(df: pd.DataFrame, colspec: str, headers: list[str],
            formatters: dict, highlight_first: bool = False) -> str:
    """Render a plain LaTeX tabular.

    Deliberately uses only \\hline and \\textbf, which are part of base LaTeX. The report
    is built on the department's projectreport.cls and that file is not ours to modify, so
    nothing here may depend on a package the class does not already load. That rules out
    booktabs (\\toprule etc.) and colortbl (\\rowcolor), which earlier versions used; the
    winning row is now marked in bold instead of being shaded.

    Kept manual rather than using DataFrame.to_latex so column formatting and the winner
    highlight stay explicit.
    """
    lines = [r"\begin{tabular}{" + colspec + "}", r"\hline"]
    lines.append(" & ".join(rf"\textbf{{{h}}}" for h in headers) + r" \\")
    lines.append(r"\hline")

    for i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for col in df.columns:
            fmt = formatters.get(col, str)
            value = fmt(row[col])
            if highlight_first and i == 0:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


# ======================================================================================

def table_model_comparison() -> None:
    df = pd.read_csv(REPORTS / "model_comparison.csv")
    df = df[["model", "PR_AUC", "ROC_AUC", "precision", "recall", "f1", "TP", "FP", "FN"]]
    write_table("model_comparison", tabular(
        df, "l r r r r r r r r",
        ["Model", "PR-AUC", "ROC-AUC", "Prec.", "Recall", "F1", "TP", "FP", "FN"],
        {
            "model": lambda v: escape(shorten(v)),
            "PR_AUC": lambda v: f"{v:.3f}",
            "ROC_AUC": lambda v: f"{v:.3f}",
            "precision": lambda v: f"{v:.3f}",
            "recall": lambda v: f"{v:.3f}",
            "f1": lambda v: f"{v:.3f}",
            "TP": lambda v: f"{int(v)}",
            "FP": lambda v: f"{int(v)}",
            "FN": lambda v: f"{int(v)}",
        },
        highlight_first=True,
    ))


def table_extended_metrics() -> None:
    df = pd.read_csv(REPORTS / "extended_metrics.csv")
    df = df[["model", "MCC", "specificity", "NPV", "balanced_accuracy",
             "youden_J", "alerts_per_1000"]]
    write_table("extended_metrics", tabular(
        df, "l r r r r r r",
        ["Model", "MCC", "Spec.", "NPV", "Bal. acc.", "Youden $J$", "Alerts/1k"],
        {
            "model": lambda v: escape(shorten(v)),
            "MCC": lambda v: f"{v:.3f}",
            "specificity": lambda v: f"{v:.5f}",
            "NPV": lambda v: f"{v:.5f}",
            "balanced_accuracy": lambda v: f"{v:.3f}",
            "youden_J": lambda v: f"{v:.3f}",
            "alerts_per_1000": lambda v: f"{v:.3f}",
        },
        highlight_first=True,
    ))


def table_confidence_intervals() -> None:
    df = pd.read_csv(REPORTS / "confidence_intervals.csv")
    out = pd.DataFrame({
        "model": df["model"],
        "pr": df.apply(lambda r: f"{r.PR_AUC:.3f} [{r.PR_AUC_lo95:.3f}, {r.PR_AUC_hi95:.3f}]",
                       axis=1),
        "prec": df.apply(
            lambda r: f"{r.precision:.3f} [{r.precision_lo95:.3f}, {r.precision_hi95:.3f}]",
            axis=1),
        "rec": df.apply(
            lambda r: f"{r.recall:.3f} [{r.recall_lo95:.3f}, {r.recall_hi95:.3f}]", axis=1),
    })
    write_table("confidence_intervals", tabular(
        out, "l l l l",
        ["Model", "PR-AUC [95\\% CI]", "Precision [95\\% CI]", "Recall [95\\% CI]"],
        {"model": lambda v: escape(shorten(v)), "pr": str, "prec": str, "rec": str},
        highlight_first=True,
    ))


def table_resampling() -> None:
    df = pd.read_csv(REPORTS / "resampling_comparison.csv")
    train_rows = {
        "Class weighting": "170{,}235",
        "Random oversampling": "339{,}902",
        "SMOTE": "339{,}902",
        "Random undersampling": "568",
    }
    df["train_rows"] = df["model"].map(train_rows).fillna("--")
    df = df[["model", "PR_AUC", "ROC_AUC", "precision", "recall", "f1", "train_rows"]]
    write_table("resampling_comparison", tabular(
        df, "l r r r r r r",
        ["Strategy", "PR-AUC", "ROC-AUC", "Prec.", "Recall", "F1", "Train rows"],
        {
            "model": lambda v: escape(v),
            "PR_AUC": lambda v: f"{v:.3f}",
            "ROC_AUC": lambda v: f"{v:.3f}",
            "precision": lambda v: f"{v:.3f}",
            "recall": lambda v: f"{v:.3f}",
            "f1": lambda v: f"{v:.3f}",
            "train_rows": str,
        },
        highlight_first=True,
    ))


def table_significance() -> None:
    """The 10 most nearly-significant pairs; the full 28 go in the appendix."""
    df = pd.read_csv(REPORTS / "significance_tests.csv").head(10)
    out = df[["model_a", "model_b", "n_discordant", "p_value", "p_holm"]]
    write_table("significance_tests", tabular(
        out, "l l r r r",
        # \mathrm, not amsmath's \text: projectreport.cls does not load amsmath and is
        # not ours to modify.
        ["Model A", "Model B", "Discordant", "$p$", "$p_{\\mathrm{Holm}}$"],
        {
            "model_a": lambda v: escape(shorten(v)),
            "model_b": lambda v: escape(shorten(v)),
            "n_discordant": lambda v: f"{int(v)}",
            "p_value": lambda v: f"{v:.4f}",
            "p_holm": lambda v: f"{v:.3f}",
        },
    ))


def table_calibration() -> None:
    df = pd.read_csv(REPORTS / "calibration.csv")
    summary = (df.groupby("model")[["ECE", "brier"]].first()
                 .sort_values("ECE").reset_index())
    write_table("calibration", tabular(
        summary, "l r r",
        ["Model", "ECE", "Brier score"],
        {
            "model": lambda v: escape(shorten(v)),
            "ECE": lambda v: f"{v:.5f}",
            "brier": lambda v: f"{v:.5f}",
        },
        highlight_first=True,
    ))


def table_cross_validation() -> None:
    path = REPORTS / "cv_results.csv"
    if not path.exists():
        print("  skipping cv_results (run deep_analysis.py without --fast)")
        return
    df = pd.read_csv(path)
    agg = (df.groupby("model")["PR_AUC"].agg(["mean", "std", "min", "max"])
             .sort_values("mean", ascending=False).reset_index())
    write_table("cross_validation", tabular(
        agg, "l r r r r",
        ["Model", "Mean PR-AUC", "Std.", "Min", "Max"],
        {
            "model": lambda v: escape(shorten(v)),
            "mean": lambda v: f"{v:.4f}",
            "std": lambda v: f"{v:.4f}",
            "min": lambda v: f"{v:.4f}",
            "max": lambda v: f"{v:.4f}",
        },
        highlight_first=True,
    ))


def table_learning_curve() -> None:
    path = REPORTS / "learning_curve.csv"
    if not path.exists():
        print("  skipping learning_curve (run deep_analysis.py without --fast)")
        return
    df = pd.read_csv(path)
    write_table("learning_curve", tabular(
        df, "r r r r",
        ["Fraction", "Training rows", "Frauds", "Test PR-AUC"],
        {
            "fraction": lambda v: f"{v:.0%}".replace("%", r"\%"),
            "n_train": lambda v: f"{int(v):,}".replace(",", "{,}"),
            "n_frauds": lambda v: f"{int(v)}",
            "test_PR_AUC": lambda v: f"{v:.3f}",
        },
    ))


def table_feature_importance() -> None:
    df = pd.read_csv(REPORTS / "feature_importance.csv").head(12)
    out = df[["feature", "permutation_mean", "permutation_std", "gain_pct", "split_count"]]
    write_table("feature_importance", tabular(
        out, "l r r r r",
        ["Feature", "Perm. importance", "Std.", "Gain \\%", "Splits"],
        {
            "feature": lambda v: escape(v),
            "permutation_mean": lambda v: f"{v:.4f}",
            "permutation_std": lambda v: f"{v:.4f}",
            "gain_pct": lambda v: f"{v:.2f}",
            "split_count": lambda v: f"{int(v)}",
        },
    ))


def table_cost() -> None:
    df = pd.read_csv(REPORTS / "cost_analysis.csv")
    best = df.loc[df["total_cost_eur"].idxmin()]
    tuned = df.loc[df["is_tuned_threshold"]].iloc[0]
    # A few representative operating points plus the two that matter.
    picks = [0.005, 0.05, 0.25, 0.5, 0.75, 0.95]
    rows = [df.iloc[(df["threshold"] - p).abs().idxmin()] for p in picks]
    out = pd.DataFrame(rows + [best, tuned]).drop_duplicates(subset="threshold")
    out = out.sort_values("threshold")
    out = out[["threshold", "missed_frauds", "false_alarms",
               "fraud_loss_eur", "review_cost_eur", "total_cost_eur"]]
    write_table("cost_analysis", tabular(
        out, "r r r r r r",
        ["Threshold", "Missed", "False alarms", "Fraud loss (EUR)",
         "Review (EUR)", "Total (EUR)"],
        {
            "threshold": lambda v: f"{v:.3f}",
            "missed_frauds": lambda v: f"{int(v)}",
            "false_alarms": lambda v: f"{int(v)}",
            "fraud_loss_eur": lambda v: f"{v:,.0f}".replace(",", "{,}"),
            "review_cost_eur": lambda v: f"{v:,.0f}".replace(",", "{,}"),
            "total_cost_eur": lambda v: f"{v:,.0f}".replace(",", "{,}"),
        },
    ))


def table_lift() -> None:
    df = pd.read_csv(REPORTS / "lift_table.csv")
    out = df[["decile", "n_reviewed", "frauds_caught", "cumulative_recall", "lift"]]
    write_table("lift", tabular(
        out, "r r r r r",
        ["Decile", "Reviewed", "Frauds caught", "Cum. recall", "Lift"],
        {
            "decile": lambda v: f"{int(v)}",
            "n_reviewed": lambda v: f"{int(v):,}".replace(",", "{,}"),
            "frauds_caught": lambda v: f"{int(v)}",
            "cumulative_recall": lambda v: f"{v:.3f}",
            "lift": lambda v: f"{v:.2f}",
        },
    ))


def table_dataset_profile() -> None:
    df = pd.read_csv(REPORTS / "dataset_profile.csv")
    labels = {
        "rows_after_dedup": "Transactions after de-duplication",
        "duplicates_removed": "Exact duplicate rows removed",
        "fraud_count": "Fraudulent transactions",
        "legit_count": "Legitimate transactions",
        "fraud_rate_pct": "Fraud rate (\\%)",
        "imbalance_ratio": "Imbalance ratio (legit : fraud)",
        "missing_values": "Missing values",
        "n_features_modelled": "Features used for modelling",
        "amount_median_legit": "Median amount, legitimate (EUR)",
        "amount_median_fraud": "Median amount, fraud (EUR)",
        "amount_mean_legit": "Mean amount, legitimate (EUR)",
        "amount_mean_fraud": "Mean amount, fraud (EUR)",
        "amount_p95_legit": "95th percentile amount, legitimate (EUR)",
        "amount_p95_fraud": "95th percentile amount, fraud (EUR)",
        "amount_max_fraud": "Largest fraudulent amount (EUR)",
        "peak_fraud_hour": "Hour of day with highest fraud rate",
        "peak_fraud_rate_pct": "Peak hourly fraud rate (\\%)",
        "min_fraud_rate_pct": "Lowest hourly fraud rate (\\%)",
        "top_feature": "Strongest single feature",
        "top_feature_abs_corr": "Its absolute correlation with the label",
        "second_feature": "Second strongest feature",
        "second_feature_abs_corr": "Its absolute correlation with the label",
    }

    def fmt(stat, value):
        if stat in {"top_feature", "second_feature"}:
            return str(value)
        try:
            v = float(value)
        except (TypeError, ValueError):
            return escape(value)
        if float(v).is_integer() and abs(v) >= 1000:
            return f"{int(v):,}".replace(",", "{,}")
        if float(v).is_integer():
            return f"{int(v)}"
        return f"{v:.3f}"

    out = pd.DataFrame({
        "label": [labels.get(s, escape(s)) for s in df["statistic"]],
        "value": [fmt(s, v) for s, v in zip(df["statistic"], df["value"])],
    })
    write_table("dataset_profile", tabular(
        out, "l r", ["Statistic", "Value"],
        {"label": str, "value": str},
    ))


def table_splits() -> None:
    """Static, but generated so the split sizes live in exactly one place."""
    rows = pd.DataFrame([
        ["Training", "170{,}235", "284", "0.167", "Fit model parameters"],
        ["Validation", "56{,}745", "94", "0.166", "Choose the decision threshold"],
        ["Test", "56{,}746", "95", "0.167", "Report final numbers (used once)"],
    ], columns=["split", "rows", "frauds", "rate", "purpose"])
    write_table("splits", tabular(
        rows, "l r r r l",
        ["Split", "Rows", "Frauds", "Rate (\\%)", "Purpose"],
        {c: str for c in rows.columns},
    ))


def table_hyperparameters() -> None:
    rows = pd.DataFrame([
        ["n\\_estimators", "1500", "Number of boosting rounds"],
        ["learning\\_rate", "0.1", "Shrinkage applied to each tree"],
        ["num\\_leaves", "15", "Max leaves per tree (controls complexity)"],
        ["min\\_child\\_samples", "30", "Min samples in a leaf"],
        ["colsample\\_bytree", "0.7", "Feature subsample per tree"],
        ["reg\\_lambda", "0.0", "L2 regularisation"],
        ["class\\_weight", "balanced", "Re-weights the loss by inverse class frequency"],
        ["random\\_state", "42", "Seed, fixed for reproducibility"],
    ], columns=["param", "value", "meaning"])
    write_table("hyperparameters", tabular(
        rows, "l r l", ["Hyperparameter", "Value", "Role"],
        {c: str for c in rows.columns},
    ))


def copy_figures() -> None:
    FIGURES_DST.mkdir(parents=True, exist_ok=True)
    n = 0
    for png in sorted(FIGURES_SRC.glob("*.png")):
        shutil.copy2(png, FIGURES_DST / png.name)
        n += 1
    print(f"  copied {n} figures into latex/figures/")


def main() -> None:
    LATEX.mkdir(parents=True, exist_ok=True)
    print("Generating LaTeX tables from the analysis CSVs...")
    table_model_comparison()
    table_extended_metrics()
    table_confidence_intervals()
    table_resampling()
    table_significance()
    table_calibration()
    table_cross_validation()
    table_learning_curve()
    table_feature_importance()
    table_cost()
    table_lift()
    table_dataset_profile()
    table_splits()
    table_hyperparameters()
    copy_figures()
    print("\nDone. reports/latex/ is ready to zip and upload to Overleaf.")


if __name__ == "__main__":
    main()
