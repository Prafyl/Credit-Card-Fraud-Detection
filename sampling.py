"""
Credit Card Fraud Detection — Dataset Variant Comparison
=========================================================

APPROACH:
  Step 1 — Generate 4 dataset variants from the original, save as CSVs
  Step 2 — Train all 8 models on each variant
  Step 3 — Compare everything in one master chart

Dataset variants (training data only — test set is NEVER touched):
  variant_imbalanced.csv      — raw data as-is
  variant_undersample.csv     — random undersample majority to 1:1
  variant_oversample.csv      — random oversample minority to 1:1
  variant_smote.csv           — SMOTE synthetic oversampling to 1:1
  variant_smote_tomek.csv     — SMOTE then Tomek link cleanup

Why 5 not 4? Random oversample is added as a baseline between
undersampling and SMOTE — it duplicates real fraud rows rather
than synthesising new ones, so it's a useful middle ground.

IMPORTANT: Test set stays at real-world 0.17% fraud throughout.
Primary metric: PR-AUC. Accuracy is intentionally excluded.

Estimated time on RTX 4060 Mobile:
  Dataset generation : ~5–8 min  (SMOTE+Tomek is the slow one)
  Model training     : ~60–80 min total (5 variants × 8 models)

Outputs:
  variant_*.csv                     — the 5 balanced datasets
  results_<variant>.png             — per-variant chart
  results_master_comparison.png     — all variants side by side
  results_accuracy_trap.png         — why accuracy is misleading
  results_all.csv                   — full combined results table
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
)
from imblearn.over_sampling   import SMOTE, RandomOverSampler
from imblearn.combine         import SMOTETomek
from imblearn.under_sampling  import RandomUnderSampler
from xgboost   import XGBClassifier
from lightgbm  import LGBMClassifier


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
ORIGINAL_CSV = "creditcardfraud/creditcard.csv"
OUT_DIR      = "."
RANDOM_STATE = 42

VARIANTS = {
    "imbalanced":    None,
    "undersample":   RandomUnderSampler(sampling_strategy=1.0, random_state=RANDOM_STATE),
    "oversample":    RandomOverSampler(sampling_strategy=1.0,  random_state=RANDOM_STATE),
    "smote":         SMOTE(sampling_strategy=1.0,              random_state=RANDOM_STATE),
    "smote_tomek":   SMOTETomek(sampling_strategy=1.0,         random_state=RANDOM_STATE),
}

VARIANT_LABELS = {
    "imbalanced":   "Imbalanced (baseline)",
    "undersample":  "Random Undersample",
    "oversample":   "Random Oversample",
    "smote":        "SMOTE",
    "smote_tomek":  "SMOTE + Tomek",
}

VARIANT_COLORS = {
    "imbalanced":   "#5C6BC0",
    "undersample":  "#E53935",
    "oversample":   "#43A047",
    "smote":        "#FB8C00",
    "smote_tomek":  "#8E24AA",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 0. GPU DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def detect_gpu():
    try:
        probe = XGBClassifier(n_estimators=1, device='cuda', verbosity=0)
        probe.fit(np.zeros((4, 2)), [0, 0, 1, 1])
        print("✅  GPU detected — XGBoost & LightGBM will use CUDA.\n")
        return 'cuda', 'gpu'
    except Exception:
        print("⚠️  No CUDA GPU — using CPU.\n")
        return 'cpu', 'cpu'


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_preprocess(path):
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows | fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    sc = StandardScaler()
    df['Amount_scaled'] = sc.fit_transform(df[['Amount']])
    df['Time_scaled']   = sc.fit_transform(df[['Time']])
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GENERATE & SAVE DATASET VARIANTS
#    Each variant CSV contains only the TRAINING portion after resampling.
#    The test set is split off first and never touched.
# ═══════════════════════════════════════════════════════════════════════════════
def generate_variants(X_train, y_train, feature_cols):
    """
    Apply each sampler to X_train/y_train and save as a CSV.
    Returns dict of {variant_name: (X_res, y_res)}
    """
    print("\n" + "="*65)
    print("  STEP 1 — GENERATING DATASET VARIANTS")
    print("="*65)

    variant_data = {}

    for name, sampler in VARIANTS.items():
        csv_path = os.path.join(OUT_DIR, f"variant_{name}.csv")
        label    = VARIANT_LABELS[name]

        if os.path.exists(csv_path):
            print(f"\n  ✅ {label:<30} already exists — loading {csv_path}")
            df_v     = pd.read_csv(csv_path)
            X_res    = df_v.drop('Class', axis=1).values
            y_res    = df_v['Class'].values
            variant_data[name] = (X_res, y_res)
            print(f"     {(y_res==0).sum():,} normal | {(y_res==1).sum():,} fraud | total {len(y_res):,}")
            continue

        print(f"\n  ⚙️  {label:<30}", end=" ", flush=True)
        t0 = time.time()

        if sampler is None:
            # No resampling — use training data as-is
            X_res, y_res = X_train.values, y_train.values
        else:
            X_res, y_res = sampler.fit_resample(X_train, y_train)

        elapsed = time.time() - t0
        print(f"done in {elapsed:.0f}s")
        print(f"     {(y_res==0).sum():,} normal | {(y_res==1).sum():,} fraud | total {len(y_res):,}")

        # Save to CSV
        df_out = pd.DataFrame(X_res, columns=feature_cols)
        df_out['Class'] = y_res
        df_out = df_out.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        df_out.to_csv(csv_path, index=False)
        print(f"     💾 Saved → {csv_path}")

        variant_data[name] = (X_res, y_res)

    return variant_data


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
def get_models(xgb_device, lgbm_device, use_class_weight):
    """
    use_class_weight=True  for imbalanced variant only.
    For all resampled variants the data is already balanced —
    adding class_weight='balanced' on top would double-correct.
    """
    cw  = 'balanced' if use_class_weight else None
    spw = 577 if use_class_weight else 1   # for XGBoost

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight=cw,
            solver='lbfgs', random_state=RANDOM_STATE),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=10,
            class_weight=cw, random_state=RANDOM_STATE),

        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20,
            class_weight=cw, n_jobs=-1, random_state=RANDOM_STATE),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=20,
            class_weight=cw, n_jobs=-1, random_state=RANDOM_STATE),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=RANDOM_STATE),

        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5,
            random_state=RANDOM_STATE),

        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric='aucpr', verbosity=0,
            device=xgb_device, random_state=RANDOM_STATE),

        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=63, max_depth=-1,
            class_weight=cw,
            device=lgbm_device,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN & EVALUATE ONE VARIANT
# ═══════════════════════════════════════════════════════════════════════════════
def run_variant(variant_name, X_res, y_res,
                X_test, y_test,
                xgb_device, lgbm_device):

    label = VARIANT_LABELS[variant_name]
    print(f"\n{'═'*65}")
    print(f"  VARIANT: {label}")
    print(f"  Train: {len(y_res):,}  |  Test: {len(y_test):,}")
    print(f"  Fraud in test: {y_test.sum()} ({y_test.mean()*100:.3f}%)")
    print(f"{'═'*65}")

    use_cw = (variant_name == "imbalanced")
    models = get_models(xgb_device, lgbm_device, use_class_weight=use_cw)
    results = {}

    for name, model in models.items():
        print(f"\n  🔧 {name:<25}", end="", flush=True)
        t0 = time.time()

        model.fit(X_res, y_res)
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)[:, 1]
                  if hasattr(model, 'predict_proba')
                  else model.decision_function(X_test))

        results[name] = {
            "model":     model,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "PR-AUC":    average_precision_score(y_test, y_prob),
            "ROC-AUC":   roc_auc_score(y_test, y_prob),
            "F1":        f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred),
            "Accuracy":  float((y_pred == y_test).mean()),
            "Time (s)":  round(elapsed, 1),
        }
        r = results[name]
        print(f"  PR-AUC={r['PR-AUC']:.4f}  F1={r['F1']:.4f}  "
              f"Recall={r['Recall']:.4f}  [{elapsed:.0f}s]")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
def make_summary(results, variant_name):
    rows = []
    for name, r in results.items():
        rows.append({
            "Variant":   VARIANT_LABELS[variant_name],
            "Model":     name,
            "PR-AUC":    round(r["PR-AUC"],    4),
            "ROC-AUC":   round(r["ROC-AUC"],   4),
            "F1":        round(r["F1"],         4),
            "Precision": round(r["Precision"],  4),
            "Recall":    round(r["Recall"],     4),
            "Accuracy":  round(r["Accuracy"],   4),
            "Time (s)":  r["Time (s)"],
        })
    df = (pd.DataFrame(rows)
            .sort_values("PR-AUC", ascending=False)
            .reset_index(drop=True))
    df.index += 1
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CLASSIFICATION REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
def print_reports(results, y_test, label):
    print(f"\n{'═'*65}")
    print(f"  CLASSIFICATION REPORTS — {label}")
    print(f"{'═'*65}")
    for name, r in results.items():
        print(f"\n  {'─'*38}  {name}")
        print(classification_report(y_test, r["y_pred"],
                                    target_names=["Normal", "Fraud"],
                                    digits=4))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PER-VARIANT CHART
# ═══════════════════════════════════════════════════════════════════════════════
def plot_variant(results, summary_df, y_test, variant_name, out_path):
    n      = len(results)
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    label  = VARIANT_LABELS[variant_name]

    fig = plt.figure(figsize=(22, 28))
    fig.suptitle(f"Fraud Detection — {label}", fontsize=17,
                 fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Metrics bar chart
    ax_bar  = fig.add_subplot(gs[0, :])
    metrics = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall"]
    x, w    = np.arange(len(metrics)), 0.8 / n
    for i, (mname, r) in enumerate(results.items()):
        ax_bar.bar(x + i*w, [r[m] for m in metrics], w,
                   label=mname, color=colors[i], alpha=0.85)
    ax_bar.set_xticks(x + w*(n-1)/2)
    ax_bar.set_xticklabels(metrics, fontsize=12)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title(f"All Metrics — {label}  (test set = real-world imbalanced)")
    ax_bar.legend(ncol=4, fontsize=8, loc='upper right')
    ax_bar.grid(axis='y', alpha=0.3)

    # ROC curves
    ax_roc = fig.add_subplot(gs[1, :2])
    for (mname, r), c in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax_roc.plot(fpr, tpr, color=c, lw=1.5,
                    label=f"{mname} ({r['ROC-AUC']:.3f})")
    ax_roc.plot([0,1],[0,1],'k--', lw=0.8)
    ax_roc.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
    ax_roc.legend(fontsize=7); ax_roc.grid(alpha=0.3)

    # PR curves
    ax_pr = fig.add_subplot(gs[1, 2:])
    for (mname, r), c in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        ax_pr.plot(rec, prec, color=c, lw=1.5,
                   label=f"{mname} ({r['PR-AUC']:.3f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision",
              title="Precision-Recall Curves  ← primary")
    ax_pr.legend(fontsize=7); ax_pr.grid(alpha=0.3)

    # Confusion matrices — top 4
    top4 = summary_df.head(4)["Model"].tolist()
    for col, mname in enumerate(top4):
        ax_cm = fig.add_subplot(gs[2, col])
        cm    = confusion_matrix(y_test, results[mname]["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    cbar=False,
                    xticklabels=['Normal','Fraud'],
                    yticklabels=['Normal','Fraud'])
        missed = cm[1][0]
        ax_cm.set_title(f"{mname}\nMissed fraud: {missed}", fontsize=8)
        ax_cm.set_xlabel("Predicted", fontsize=7)
        ax_cm.set_ylabel("Actual",    fontsize=7)

    # Recall vs Precision scatter
    ax_sc = fig.add_subplot(gs[3, :2])
    for (mname, r), c in zip(results.items(), colors):
        ax_sc.scatter(r["Recall"], r["Precision"], color=c, s=120, zorder=3)
        ax_sc.annotate(mname, (r["Recall"], r["Precision"]),
                       textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax_sc.set(xlabel="Recall (fraud caught)",
              ylabel="Precision (false alarm rate)",
              title="Recall vs Precision — top-right corner is best")
    ax_sc.set_xlim(-0.05, 1.05); ax_sc.set_ylim(-0.05, 1.05)
    ax_sc.grid(alpha=0.3)

    # Summary table
    ax_tbl = fig.add_subplot(gs[3, 2:])
    ax_tbl.axis('off')
    disp = summary_df.drop(columns=["Variant"], errors='ignore')
    tbl  = ax_tbl.table(cellText=disp.values, colLabels=disp.columns,
                        cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.6)
    ax_tbl.set_title("Ranked by PR-AUC", fontsize=10, pad=10)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MASTER COMPARISON CHART — all variants × all models
# ═══════════════════════════════════════════════════════════════════════════════
def plot_master_comparison(all_summaries, out_path):
    variant_names  = list(all_summaries.keys())
    variant_labels = [VARIANT_LABELS[v] for v in variant_names]
    colors         = [VARIANT_COLORS[v] for v in variant_names]

    # Model order: sort by mean PR-AUC across all variants (best first)
    all_models = all_summaries[variant_names[0]]["Model"].tolist()
    mean_prauc = {}
    for m in all_models:
        scores = []
        for sdf in all_summaries.values():
            row = sdf[sdf["Model"] == m]
            if not row.empty:
                scores.append(row["PR-AUC"].values[0])
        mean_prauc[m] = np.mean(scores) if scores else 0
    models_sorted = sorted(all_models, key=lambda m: mean_prauc[m], reverse=True)

    metrics = ["PR-AUC", "F1", "Recall", "Precision"]
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.suptitle(
        "Balancing Strategy Comparison — All Variants × All Models\n"
        "(Test set always real-world imbalanced, 0.17% fraud. Accuracy excluded intentionally.)",
        fontsize=14, fontweight='bold')

    for ax, metric in zip(axes.flat, metrics):
        x = np.arange(len(models_sorted))
        w = 0.8 / len(variant_names)

        for i, (vname, vlabel) in enumerate(zip(variant_names, variant_labels)):
            sdf = all_summaries[vname]
            score_map = dict(zip(sdf["Model"], sdf[metric]))
            scores    = [score_map.get(m, 0) for m in models_sorted]
            bars = ax.bar(x + i*w, scores, w,
                          label=vlabel, color=colors[i], alpha=0.82)

            # Bold outline on best bar per model
            best_i = int(np.argmax([
                all_summaries[v].set_index("Model").loc[m, metric]
                if m in all_summaries[v]["Model"].values else 0
                for v, m_iter in [(v, m) for v in variant_names]
                for m in [m_iter]
            ][:len(variant_names)]))
            ax.bar(x[models_sorted.index(models_sorted[0])] + best_i*w,
                   0, w)  # placeholder — winner highlight below

        # Highlight the best variant for each model with a black border
        for j, m in enumerate(models_sorted):
            per_variant = []
            for vname in variant_names:
                sdf = all_summaries[vname]
                if m in sdf["Model"].values:
                    per_variant.append(sdf.set_index("Model").loc[m, metric])
                else:
                    per_variant.append(0)
            best_i = int(np.argmax(per_variant))
            ax.bar(j + best_i*w, per_variant[best_i], w,
                   color=colors[best_i], alpha=1.0,
                   edgecolor='black', linewidth=1.5)

        ax.set_xticks(x + w*(len(variant_names)-1)/2)
        ax.set_xticklabels(models_sorted, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}  {'← PRIMARY METRIC' if metric=='PR-AUC' else ''}")
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ACCURACY TRAP CHART
# ═══════════════════════════════════════════════════════════════════════════════
def plot_accuracy_trap(all_summaries, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Why Accuracy Is Misleading for Fraud Detection",
                 fontsize=14, fontweight='bold')

    # Left: accuracy vs PR-AUC scatter
    ax = axes[0]
    for vname, sdf in all_summaries.items():
        ax.scatter(sdf["Accuracy"], sdf["PR-AUC"],
                   color=VARIANT_COLORS[vname], s=90, zorder=3,
                   label=VARIANT_LABELS[vname])
    ax.axvline(0.9983, color='red', linestyle='--', lw=1.5,
               label='Baseline: predict all Normal\n(99.83% acc, 0% recall)')
    ax.set(xlabel="Accuracy", ylabel="PR-AUC",
           title="Accuracy vs PR-AUC\nHigh accuracy ≠ good fraud detector")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Right: Recall comparison — how much fraud do we actually catch?
    ax2 = axes[1]
    variant_names  = list(all_summaries.keys())
    models         = all_summaries[variant_names[0]]["Model"].tolist()
    x = np.arange(len(models))
    w = 0.8 / len(variant_names)
    for i, vname in enumerate(variant_names):
        sdf       = all_summaries[vname]
        score_map = dict(zip(sdf["Model"], sdf["Recall"]))
        scores    = [score_map.get(m, 0) for m in models]
        ax2.bar(x + i*w, scores, w,
                label=VARIANT_LABELS[vname],
                color=VARIANT_COLORS[vname], alpha=0.85)
    ax2.set_xticks(x + w*(len(variant_names)-1)/2)
    ax2.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set(ylabel="Recall (% of actual fraud caught)",
            title="Recall by Variant & Model\n← The number that matters most operationally")
    ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. WINNER
# ═══════════════════════════════════════════════════════════════════════════════
def announce_winner(combined_df):
    best = combined_df.sort_values("PR-AUC", ascending=False).iloc[0]
    print("\n" + "🏆"*30)
    print(f"\n  BEST COMBINATION")
    print(f"  Variant : {best['Variant']}")
    print(f"  Model   : {best['Model']}")
    print(f"  PR-AUC  : {best['PR-AUC']}")
    print(f"  ROC-AUC : {best['ROC-AUC']}")
    print(f"  F1      : {best['F1']}")
    print(f"  Recall  : {best['Recall']}")
    print(f"  Precision: {best['Precision']}")
    print()
    print("  Key insight for your supervisor:")
    print("  Look at the Recall column — that is the % of actual frauds caught.")
    print("  Balancing typically boosts Recall at a cost to Precision.")
    print("  The right trade-off depends on business cost:")
    print("    Missing fraud  → financial loss, liability")
    print("    False alarm    → blocked legit customer, support cost")
    print("🏆"*30 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    wall_start = time.time()

    print("\n🔍 Probing GPU...")
    xgb_device, lgbm_device = detect_gpu()

    print("📂 Loading and preprocessing original dataset...")
    X, y = load_and_preprocess(ORIGINAL_CSV)
    feature_cols = list(X.columns)

    # ── Split ONCE — test set never touched again ─────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.20, random_state=RANDOM_STATE)

    print(f"\n  Train pool : {X_train.shape[0]:,} rows  "
          f"(fraud: {y_train.sum():,} = {y_train.mean()*100:.3f}%)")
    print(f"  Test  set  : {X_test.shape[0]:,} rows  "
          f"(fraud: {y_test.sum():,} = {y_test.mean()*100:.3f}%)")
    print(f"\n  ⚠️  Resampling applied to training data only.")
    print(f"      Test set locked at real-world distribution.\n")

    # ── Step 1: Generate and save all variant CSVs ────────────────────────────
    variant_data = generate_variants(X_train, y_train, feature_cols)

    # ── Step 2: Train all models on each variant ──────────────────────────────
    print("\n" + "="*65)
    print("  STEP 2 — TRAINING MODELS ON EACH VARIANT")
    print("="*65)

    all_results   = {}
    all_summaries = {}

    for vname in VARIANTS:
        X_res, y_res = variant_data[vname]

        results = run_variant(
            vname, X_res, y_res,
            X_test.values, y_test.values,
            xgb_device, lgbm_device,
        )
        summary = make_summary(results, vname)
        print_reports(results, y_test.values, VARIANT_LABELS[vname])

        slug     = vname
        png_path = os.path.join(OUT_DIR, f"results_{slug}.png")
        plot_variant(results, summary, y_test.values, vname, png_path)

        all_results[vname]   = results
        all_summaries[vname] = summary

    # ── Step 3: Master comparison charts ─────────────────────────────────────
    print("\n📊 Generating master comparison charts...")
    plot_master_comparison(
        all_summaries,
        os.path.join(OUT_DIR, "results_master_comparison.png"))

    plot_accuracy_trap(
        all_summaries,
        os.path.join(OUT_DIR, "results_accuracy_trap.png"))

    # ── Save full CSV ─────────────────────────────────────────────────────────
    combined = pd.concat(list(all_summaries.values()), ignore_index=True)
    csv_path = os.path.join(OUT_DIR, "results_all.csv")
    combined.to_csv(csv_path, index=False)
    print(f"  💾  Full results → {csv_path}")

    # ── Console summary per variant ───────────────────────────────────────────
    for vname, sdf in all_summaries.items():
        print(f"\n{'═'*65}")
        print(f"  {VARIANT_LABELS[vname]} — Ranked by PR-AUC")
        print(f"{'═'*65}")
        print(sdf.to_string())

    announce_winner(combined)

    total = time.time() - wall_start
    print(f"⏱️  Total wall-clock time: {total/60:.1f} min\n")
    """

from __future__ import annotations

import math
import random
import statistics
import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Iterable


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------

@dataclass
class MetricRecord:
    identifier: str
    values: List[float]
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return statistics.mean(self.values)

    @property
    def median(self) -> float:
        return statistics.median(self.values)

    @property
    def variance(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return statistics.variance(self.values)


@dataclass
class ScoreResult:
    score: float
    confidence: float
    category: str


# ---------------------------------------------------------------------
# Random data generation
# ---------------------------------------------------------------------

def create_identifier(seed: int) -> str:
    raw = f"seed::{seed}::{random.random()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_series(length: int) -> List[float]:
    output = []

    for index in range(length):
        base = math.sin(index / 10)
        modifier = random.uniform(-0.5, 0.5)
        output.append(base + modifier)

    return output


def create_record(index: int) -> MetricRecord:
    return MetricRecord(
        identifier=create_identifier(index),
        values=generate_series(100),
        metadata={
            "source": "simulation",
            "batch": str(index),
        },
    )


# ---------------------------------------------------------------------
# Mathematical utilities
# ---------------------------------------------------------------------

def normalize(values: Iterable[float]) -> List[float]:
    values = list(values)

    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return [0.0 for _ in values]

    return [
        (value - minimum) / (maximum - minimum)
        for value in values
    ]


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 0:
        raise ValueError("Window must be positive.")

    result = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        segment = values[start:i + 1]
        result.append(sum(segment) / len(segment))

    return result


def root_mean_square(values: List[float]) -> float:
    if not values:
        return 0.0

    total = sum(v * v for v in values)
    return math.sqrt(total / len(values))


def euclidean_distance(a: List[float], b: List[float]) -> float:
    total = 0.0

    for x, y in zip(a, b):
        total += (x - y) ** 2

    return math.sqrt(total)


# ---------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------

class ScoringEngine:

    def __init__(self) -> None:
        self.history: List[ScoreResult] = []

    def compute_score(self, record: MetricRecord) -> ScoreResult:
        normalized = normalize(record.values)

        average = statistics.mean(normalized)
        spread = root_mean_square(normalized)

        score = (average * 0.7) + (spread * 0.3)

        if score > 0.8:
            category = "high"
        elif score > 0.5:
            category = "medium"
        else:
            category = "low"

        confidence = min(1.0, score + 0.1)

        result = ScoreResult(
            score=score,
            confidence=confidence,
            category=category,
        )

        self.history.append(result)

        return result

    def summarize(self) -> Dict[str, float]:
        if not self.history:
            return {
                "count": 0,
                "average_score": 0.0,
            }

        scores = [entry.score for entry in self.history]

        return {
            "count": len(scores),
            "average_score": statistics.mean(scores),
            "maximum_score": max(scores),
            "minimum_score": min(scores),
        }


# ---------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------

class DatasetManager:

    def __init__(self) -> None:
        self.records: Dict[str, MetricRecord] = {}

    def add(self, record: MetricRecord) -> None:
        self.records[record.identifier] = record

    def remove(self, identifier: str) -> None:
        self.records.pop(identifier, None)

    def all_records(self) -> List[MetricRecord]:
        return list(self.records.values())

    def random_record(self) -> MetricRecord | None:
        if not self.records:
            return None

        keys = list(self.records.keys())
        key = random.choice(keys)

        return self.records[key]


# ---------------------------------------------------------------------
# Reporting functions
# ---------------------------------------------------------------------

def build_report(
    records: List[MetricRecord],
    engine: ScoringEngine,
) -> Dict[str, Dict]:
    report = {}

    for record in records:
        result = engine.compute_score(record)

        report[record.identifier] = {
            "score": result.score,
            "confidence": result.confidence,
            "category": result.category,
        }

    return report


def compare_records(
    left: MetricRecord,
    right: MetricRecord,
) -> float:
    left_values = normalize(left.values)
    right_values = normalize(right.values)

    return euclidean_distance(left_values, right_values)


# ---------------------------------------------------------------------
# Synthetic benchmark suite
# ---------------------------------------------------------------------

class Benchmark:

    def __init__(self) -> None:
        self.manager = DatasetManager()
        self.engine = ScoringEngine()

    def populate(self, count: int = 50) -> None:
        for index in range(count):
            self.manager.add(create_record(index))

    def execute(self) -> Dict:
        records = self.manager.all_records()
        report = build_report(records, self.engine)

        return {
            "summary": self.engine.summarize(),
            "records": report,
        }


# ---------------------------------------------------------------------
# Miscellaneous utilities
# ---------------------------------------------------------------------

def pairwise(values: List[float]):
    iterator = iter(values)
    return itertools.zip_longest(iterator, iterator)


def chunk(values: List[float], size: int):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def checksum(values: List[float]) -> str:
    content = ",".join(f"{v:.4f}" for v in values)
    return hashlib.md5(content.encode()).hexdigest()


def flatten(items):
    output = []

    for item in items:
        if isinstance(item, list):
            output.extend(flatten(item))
        else:
            output.append(item)

    return output


# ---------------------------------------------------------------------
# Main execution block
# ---------------------------------------------------------------------

def main() -> None:
    benchmark = Benchmark()

    benchmark.populate(25)

    result = benchmark.execute()

    print("=" * 60)
    print("Synthetic benchmark completed.")
    print("=" * 60)

    for key, value in result["summary"].items():
        print(f"{key:20s}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    main()
Generate a .gitignore file for this repository.

This script is intentionally simple: it captures common Python and project
artifacts that should not be committed.


from pathlib import Path

GITIGNORE_CONTENT = # Python cache and bytecode
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
build/
dist/
*.egg-info/
.eggs/

# Virtual environments
.env
.venv
venv/
ENV/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# VS Code
.vscode/

# MacOS
.DS_Store

# Logs
*.log


def write_gitignore(path: Path = Path(".gitignore")) -> None:
    path.write_text(GITIGNORE_CONTENT)


if __name__ == "__main__":
    write_gitignore()
    print(".gitignore file generated.")
"""
