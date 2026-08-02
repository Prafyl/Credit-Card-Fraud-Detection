"""Exploratory data analysis: seven figures written to reports/figures/.

Run:  python src/eda.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import FEATURES, TARGET, load_data
from plotting import DIVERGING, FRAUD, INK_SECONDARY, LEGIT, MUTED, save


def plot_class_balance(df: pd.DataFrame) -> None:
    """1. The headline: frauds are 0.167% of the data, so accuracy is a useless metric."""
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["Legitimate", "Fraud"], counts, color=[LEGIT, FRAUD], width=0.55)

    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            n * 1.15,
            f"{n:,}\n{n / len(df):.3%}",
            ha="center",
            va="bottom",
            color=INK_SECONDARY,
        )

    ax.set_yscale("log")  # linear scale would render the fraud bar invisible
    ax.set_ylim(1, len(df) * 12)
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Fraud is 1 in 600 transactions")
    ax.grid(axis="x", visible=False)
    save(fig, "01_class_balance")


def plot_amount_by_class(df: pd.DataFrame) -> None:
    """2. Fraudulent amounts skew small, not large — the opposite of most expectations.

    An ECDF (cumulative share of transactions at or below each amount) compares the two
    distributions' location far more clearly than overlaid histograms: the curve that sits
    higher-and-left is the smaller-valued group.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for label, colour in [("Legitimate", LEGIT), ("Fraud", FRAUD)]:
        amounts = np.sort(df.loc[df[TARGET] == (label == "Fraud"), "Amount"])
        share = np.arange(1, len(amounts) + 1) / len(amounts) * 100
        ax.step(amounts, share, where="post", linewidth=2, color=colour, label=label)

    ax.set_xscale("symlog")  # symlog keeps the many zero-amount transactions visible
    ax.set_xlim(0, df["Amount"].max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Transaction amount (EUR, log scale)")
    ax.set_ylabel("Cumulative share of transactions (%)")
    ax.set_title("Fraudulent transactions are smaller, not larger")
    ax.legend(loc="lower right")

    medians = df.groupby(TARGET)["Amount"].median()
    ax.text(0.02, 0.97, f"median legit  EUR {medians[0]:.2f}\nmedian fraud  EUR {medians[1]:.2f}",
            transform=ax.transAxes, ha="left", va="top", color=MUTED, fontsize=9)
    save(fig, "02_amount_by_class")


def plot_hourly_pattern(df: pd.DataFrame) -> None:
    """3. Volume and fraud rate by hour, stacked as two plots.

    Deliberately not a dual-axis chart: two y-scales on one plot would invent a
    relationship the data doesn't contain.
    """
    hour = (df["Time"] // 3600) % 24
    volume = df.groupby(hour).size()
    rate = df.groupby(hour)[TARGET].mean() * 100

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    top.bar(volume.index, volume, color=LEGIT, width=0.7)
    top.set_ylabel("Transactions")
    top.set_title("Fraud rate peaks when legitimate volume collapses")

    bottom.bar(rate.index, rate, color=FRAUD, width=0.7)
    bottom.set_ylabel("Fraud rate (%)")
    bottom.set_xlabel("Hour of day")
    bottom.set_xticks(range(0, 24, 2))

    peak = int(rate.idxmax())
    bottom.text(peak, rate.max(), f" {rate.max():.2f}% at {peak:02d}:00",
                va="center", color=INK_SECONDARY, fontsize=9)

    for ax in (top, bottom):
        ax.grid(axis="x", visible=False)
    save(fig, "03_hourly_pattern")


def plot_feature_signal(df: pd.DataFrame) -> pd.Series:
    """4. Rank every feature by |correlation| with Class. Returns the ranking."""
    signal = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).abs().sort_values()

    fig, ax = plt.subplots(figsize=(6.5, 8))
    # One series across nominal categories, so one colour for every bar: a value-ramp
    # here would double-encode bar length as hue.
    ax.barh(signal.index, signal, color=LEGIT, height=0.7)
    ax.set_xlabel("|correlation| with fraud label")
    ax.set_title("V17, V14, V12 and V10 carry the most signal")
    ax.grid(axis="y", visible=False)

    for name in signal.tail(4).index:
        ax.text(signal[name] + 0.005, name, f"{signal[name]:.2f}",
                va="center", color=INK_SECONDARY, fontsize=9)
    save(fig, "04_feature_signal")
    return signal


def plot_top_feature_distributions(df: pd.DataFrame, signal: pd.Series) -> None:
    """5. The separation the model will exploit, made visible rather than asserted."""
    top4 = signal.tail(4).index[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for ax, feature in zip(axes.ravel(), top4):
        bins = np.linspace(df[feature].quantile(0.001), df[feature].quantile(0.999), 60)
        for label, colour in [("Legitimate", LEGIT), ("Fraud", FRAUD)]:
            values = df.loc[df[TARGET] == (label == "Fraud"), feature]
            ax.hist(values, bins=bins, density=True, histtype="step",
                    linewidth=2, color=colour, label=label)
        ax.set_title(feature, fontsize=11)
        ax.set_yticks([])
        ax.grid(visible=False)

    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Fraud sits in a different part of the distribution",
                 fontsize=13, fontweight="semibold")
    fig.tight_layout()
    save(fig, "05_top_feature_distributions")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """6. V1-V28 are mutually uncorrelated, confirming they are PCA outputs."""
    corr = df[FEATURES + [TARGET]].corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    ax.set_title("PCA components are uncorrelated by construction")
    ax.grid(visible=False)

    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label("Pearson correlation")
    bar.outline.set_visible(False)
    save(fig, "06_correlation_heatmap")


def plot_fraud_cluster(df: pd.DataFrame) -> None:
    """7. One picture where the fraud cluster is visibly separate."""
    legit = df[df[TARGET] == 0].sample(15_000, random_state=42)  # subsample for legibility
    fraud = df[df[TARGET] == 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(legit["V17"], legit["V14"], s=6, alpha=0.25, color=LEGIT,
               linewidths=0, label="Legitimate (15,000 sample)")
    ax.scatter(fraud["V17"], fraud["V14"], s=18, alpha=0.8, color=FRAUD,
               edgecolors="white", linewidths=0.5, label=f"Fraud (all {len(fraud)})")

    ax.set_xlabel("V17")
    ax.set_ylabel("V14")
    ax.set_title("Fraud separates on the two strongest features")
    ax.legend(loc="lower left")
    save(fig, "07_fraud_cluster")


def print_summary(df: pd.DataFrame) -> None:
    frauds = int(df[TARGET].sum())
    print("\nDataset summary")
    print(f"  rows            {len(df):,} (after dropping duplicates)")
    print(f"  frauds          {frauds:,} ({frauds / len(df):.3%})")
    print(f"  features        {len(FEATURES)} used for modelling (Time excluded)")
    print(f"  missing values  {df.isna().sum().sum()}")
    print(f"  median amount   legit EUR {df[df[TARGET] == 0]['Amount'].median():.2f}, "
          f"fraud EUR {df[df[TARGET] == 1]['Amount'].median():.2f}")


def main() -> None:
    print("Loading data...")
    df = load_data()
    print_summary(df)

    print("\nGenerating figures...")
    plot_class_balance(df)
    plot_amount_by_class(df)
    plot_hourly_pattern(df)
    signal = plot_feature_signal(df)
    plot_top_feature_distributions(df, signal)
    plot_correlation_heatmap(df)
    plot_fraud_cluster(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
