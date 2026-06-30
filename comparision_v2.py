"""
Credit Card Fraud Detection — Full Model Comparison
====================================================
• Trains 8 ML algorithms on BOTH the original (~284k) and augmented (1M) datasets
• Auto-detects RTX GPU and enables CUDA for XGBoost & LightGBM
• Uses SMOTE (applied inside the train fold only — no leakage)
• Primary metric: PR-AUC  (correct choice for heavily imbalanced data)
• Saves all charts and a combined CSV report

Estimated wall-clock time on RTX 4060 Mobile:
  Original  (~284k rows, 80% train): ~12–15 min
  Augmented (~1M  rows, 80% train): ~45–60 min
  Total: ~60–75 min  (well within 12 hrs)

Install deps (once):
  pip install scikit-learn imbalanced-learn xgboost lightgbm matplotlib seaborn pandas numpy
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless — safe on any machine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
)
from imblearn.over_sampling    import SMOTE
from xgboost                   import XGBClassifier
from lightgbm                  import LGBMClassifier


# ─────────────────────────────────────────────────────────────────────────────
# 0. GPU DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_gpu():
    """
    Try a 1-sample XGBoost fit with device='cuda'.
    Returns ('cuda', 'gpu') for (xgb_device, lgbm_device) if CUDA works,
    else ('cpu', 'cpu').
    """
    try:
        probe = XGBClassifier(n_estimators=1, device='cuda', verbosity=0)
        probe.fit(np.zeros((4, 2)), [0, 0, 1, 1])
        print("✅  GPU detected — XGBoost & LightGBM will use CUDA.\n")
        return 'cuda', 'gpu'
    except Exception:
        print("⚠️  No CUDA GPU found (or wrong XGBoost build) — using CPU.\n")
        return 'cpu', 'cpu'


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path):
    print(f"  📂 {path}")
    df = pd.read_csv(path)
    print(f"     {len(df):,} rows | fraud rate: {df['Class'].mean()*100:.3f}%")
    return df


def augment_dataset(df, target_rows=1_000_000, random_state=42):
    """
    Noise-based augmentation preserving the original fraud/normal ratio.
    Identical logic to the user's own script — kept in one file for convenience.
    """
    original_count = len(df)
    if original_count >= target_rows:
        return df

    rng          = np.random.default_rng(random_state)
    rows_needed  = target_rows - original_count
    df_normal    = df[df['Class'] == 0]
    df_fraud     = df[df['Class'] == 1]
    frac_normal  = len(df_normal) / original_count
    frac_fraud   = len(df_fraud)  / original_count

    def _nudge(src_df, n):
        feats = src_df.drop('Class', axis=1).values
        label = int(src_df['Class'].iloc[0])
        idx   = rng.integers(0, len(feats), size=n)
        noise = rng.normal(0, 0.01, size=(n, feats.shape[1] - 1))
        rows  = feats[idx].copy()
        rows[:, 1:] += noise               # leave Time column unperturbed
        out         = pd.DataFrame(rows, columns=src_df.drop('Class', axis=1).columns)
        out['Class'] = label
        return out

    new_normal = _nudge(df_normal, int(rows_needed * frac_normal))
    new_fraud  = _nudge(df_fraud,  int(rows_needed * frac_fraud))

    augmented = (pd.concat([df, new_normal, new_fraud], ignore_index=True)
                   .sample(frac=1, random_state=random_state)
                   .reset_index(drop=True))
    return augmented


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    sc = StandardScaler()
    df['Amount_scaled'] = sc.fit_transform(df[['Amount']])
    df['Time_scaled']   = sc.fit_transform(df[['Time']])
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df.drop('Class', axis=1), df['Class']


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
def get_models(xgb_device, lgbm_device, scale_pos_weight):
    """
    8 diverse algorithms.
    scale_pos_weight is recomputed per dataset so the ratio stays correct
    even after augmentation.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight='balanced',
            solver='lbfgs', random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=10,
            class_weight='balanced', random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20,
            class_weight='balanced', n_jobs=-1, random_state=42,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=20,
            class_weight='balanced', n_jobs=-1, random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr', verbosity=0,
            device=xgb_device, random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=63, max_depth=-1,
            class_weight='balanced',
            device=lgbm_device,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN + EVALUATE ONE DATASET
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(label, X_train, X_test, y_train, y_test,
                   xgb_device, lgbm_device):
    print(f"\n{'='*65}")
    print(f"  EXPERIMENT: {label}")
    print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    print(f"  Fraud in test: {y_test.sum()} ({y_test.mean()*100:.3f}%)")
    print(f"{'='*65}")

    spw     = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models  = get_models(xgb_device, lgbm_device, scale_pos_weight=spw)
    smote   = SMOTE(random_state=42)
    results = {}

    for name, model in models.items():
        print(f"\n  🔧 {name:<25}", end="", flush=True)
        t0 = time.time()

        # SMOTE only on training data
        X_res, y_res = smote.fit_resample(X_train, y_train)
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
            "ROC-AUC":   roc_auc_score(y_test, y_prob),
            "PR-AUC":    average_precision_score(y_test, y_prob),
            "F1":        f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred),
            "Time (s)":  round(elapsed, 1),
        }
        r = results[name]
        print(f"  ROC-AUC={r['ROC-AUC']:.4f}  PR-AUC={r['PR-AUC']:.4f}"
              f"  F1={r['F1']:.4f}  [{elapsed:.0f}s]")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def make_summary(results, dataset_label):
    rows = []
    for name, r in results.items():
        rows.append({
            "Dataset":   dataset_label,
            "Model":     name,
            "ROC-AUC":   round(r["ROC-AUC"],   4),
            "PR-AUC":    round(r["PR-AUC"],     4),
            "F1":        round(r["F1"],          4),
            "Precision": round(r["Precision"],   4),
            "Recall":    round(r["Recall"],      4),
            "Time (s)":  r["Time (s)"],
        })
    df = (pd.DataFrame(rows)
            .sort_values("PR-AUC", ascending=False)
            .reset_index(drop=True))
    df.index += 1
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASSIFICATION REPORTS (console)
# ─────────────────────────────────────────────────────────────────────────────
def print_reports(results, y_test, label):
    print(f"\n{'='*65}")
    print(f"  CLASSIFICATION REPORTS — {label}")
    print(f"{'='*65}")
    for name, r in results.items():
        print(f"\n  {'─'*40}")
        print(f"  {name}")
        print(f"  {'─'*40}")
        print(classification_report(y_test, r["y_pred"],
                                    target_names=["Normal", "Fraud"],
                                    digits=4))


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATION — per-dataset page
# ─────────────────────────────────────────────────────────────────────────────
def plot_experiment(results, summary_df, y_test, label, out_path):
    n      = len(results)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    fig = plt.figure(figsize=(22, 28))
    fig.suptitle(f"Fraud Detection — {label}", fontsize=17,
                 fontweight='bold', y=0.99)
    gs  = gridspec.GridSpec(4, 4, figure=fig,
                            hspace=0.45, wspace=0.35)

    # (A) Metrics bar chart — full width top row
    ax_bar = fig.add_subplot(gs[0, :])
    metrics = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall"]
    x, w    = np.arange(len(metrics)), 0.8 / n
    for i, (name, r) in enumerate(results.items()):
        ax_bar.bar(x + i*w, [r[m] for m in metrics], w,
                   label=name, color=colors[i], alpha=0.85)
    ax_bar.set_xticks(x + w*(n-1)/2)
    ax_bar.set_xticklabels(metrics, fontsize=12)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("All Metrics by Model")
    ax_bar.legend(ncol=4, fontsize=8, loc='upper right')
    ax_bar.grid(axis='y', alpha=0.3)

    # (B) ROC curves
    ax_roc = fig.add_subplot(gs[1, :2])
    for (name, r), c in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax_roc.plot(fpr, tpr, color=c, lw=1.5,
                    label=f"{name} ({r['ROC-AUC']:.3f})")
    ax_roc.plot([0,1],[0,1],'k--', lw=0.8)
    ax_roc.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
    ax_roc.legend(fontsize=7)
    ax_roc.grid(alpha=0.3)

    # (C) PR curves
    ax_pr = fig.add_subplot(gs[1, 2:])
    for (name, r), c in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        ax_pr.plot(rec, prec, color=c, lw=1.5,
                   label=f"{name} ({r['PR-AUC']:.3f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision",
              title="Precision-Recall Curves")
    ax_pr.legend(fontsize=7)
    ax_pr.grid(alpha=0.3)

    # (D) Confusion matrices — top 4 by PR-AUC
    top4 = summary_df.head(4)["Model"].tolist()
    for col, mname in enumerate(top4):
        ax_cm = fig.add_subplot(gs[2, col])
        cm    = confusion_matrix(y_test, results[mname]["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    cbar=False,
                    xticklabels=['Normal','Fraud'],
                    yticklabels=['Normal','Fraud'])
        ax_cm.set_title(f"{mname}", fontsize=8)
        ax_cm.set_xlabel("Predicted", fontsize=7)
        ax_cm.set_ylabel("Actual",    fontsize=7)

    # (E) F1 vs Training Time
    ax_sc = fig.add_subplot(gs[3, :2])
    for (name, r), c in zip(results.items(), colors):
        ax_sc.scatter(r["Time (s)"], r["F1"], color=c, s=120, zorder=3)
        ax_sc.annotate(name, (r["Time (s)"], r["F1"]),
                       textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax_sc.set(xlabel="Training Time (s)", ylabel="F1 Score",
              title="F1 vs Training Time")
    ax_sc.grid(alpha=0.3)

    # (F) Summary table
    ax_tbl = fig.add_subplot(gs[3, 2:])
    ax_tbl.axis('off')
    disp_df = summary_df.drop(columns=["Dataset"], errors='ignore')
    tbl = ax_tbl.table(
        cellText=disp_df.values,
        colLabels=disp_df.columns,
        cellLoc='center', loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.6)
    ax_tbl.set_title("Ranked Summary (↓ PR-AUC)", fontsize=10, pad=10)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. HEAD-TO-HEAD COMPARISON CHART (original vs augmented)
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(summary_orig, summary_aug, out_path):
    merged = pd.merge(
        summary_orig[["Model","PR-AUC","ROC-AUC","F1","Recall","Precision"]],
        summary_aug [["Model","PR-AUC","ROC-AUC","F1","Recall","Precision"]],
        on="Model", suffixes=(" Original"," Augmented")
    )

    fig, axes = plt.subplots(1, 4, figsize=(22, 7))
    fig.suptitle("Original vs Augmented Dataset — Head-to-Head",
                 fontsize=15, fontweight='bold')

    for ax, metric in zip(axes, ["PR-AUC","ROC-AUC","F1","Recall"]):
        orig_col = f"{metric} Original"
        aug_col  = f"{metric} Augmented"
        y        = np.arange(len(merged))
        ax.barh(y - 0.2, merged[orig_col], 0.35,
                label="Original",  color='steelblue', alpha=0.85)
        ax.barh(y + 0.2, merged[aug_col],  0.35,
                label="Augmented", color='coral',     alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(merged["Model"], fontsize=8)
        ax.set_xlabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. WINNER ANNOUNCEMENT
# ─────────────────────────────────────────────────────────────────────────────
def announce_winner(combined_df):
    best = combined_df.sort_values("PR-AUC", ascending=False).iloc[0]
    print("\n" + "🏆"*32)
    print(f"\n  OVERALL BEST MODEL: {best['Model']}  ({best['Dataset']})")
    print(f"    PR-AUC    : {best['PR-AUC']}")
    print(f"    ROC-AUC   : {best['ROC-AUC']}")
    print(f"    F1        : {best['F1']}")
    print(f"    Precision : {best['Precision']}")
    print(f"    Recall    : {best['Recall']}")
    print(f"    Train time: {best['Time (s)']}s")
    print()
    print("  💡 PR-AUC is the right primary metric here because the")
    print("     dataset is highly imbalanced (~0.17% fraud). A model")
    print("     that flags everything as normal gets 99.83% accuracy")
    print("     but catches zero fraud — PR-AUC penalises that hard.")
    print("🏆"*32 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    ORIGINAL_CSV  = "creditcardfraud/creditcard.csv"
    AUGMENTED_CSV = "creditcard_1M.csv"          # produced by augment_dataset()
    OUT_DIR       = "."                           # where to save PNGs and CSV

    wall_start = time.time()

    # ── GPU probe ────────────────────────────────────────────────────────────
    print("\n🔍 Probing GPU...")
    xgb_device, lgbm_device = detect_gpu()

    # ── Load data ────────────────────────────────────────────────────────────
    print("📂 Loading datasets...")
    df_orig = load_data(ORIGINAL_CSV)

    # Build or load augmented dataset
    if os.path.exists(AUGMENTED_CSV):
        df_aug = load_data(AUGMENTED_CSV)
    else:
        print(f"\n⚙️  Augmented file not found — generating {AUGMENTED_CSV} ...")
        t_aug = time.time()
        df_aug = augment_dataset(df_orig, target_rows=1_000_000)
        df_aug.to_csv(AUGMENTED_CSV, index=False)
        print(f"   ✅ Augmentation done in {time.time()-t_aug:.0f}s")

    # ── Preprocess ───────────────────────────────────────────────────────────
    X_orig, y_orig = preprocess(df_orig)
    X_aug,  y_aug  = preprocess(df_aug)

    # ── Split ────────────────────────────────────────────────────────────────
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X_orig, y_orig, stratify=y_orig, test_size=0.20, random_state=42)

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_aug, y_aug, stratify=y_aug, test_size=0.20, random_state=42)

    # ── Experiment 1: Original ────────────────────────────────────────────────
    res_orig = run_experiment(
        "Original (~284k)", X_train_o, X_test_o, y_train_o, y_test_o,
        xgb_device, lgbm_device,
    )
    sum_orig = make_summary(res_orig, "Original")
    print_reports(res_orig, y_test_o, "Original")
    plot_experiment(res_orig, sum_orig, y_test_o,
                    "Original Dataset (~284k rows)",
                    os.path.join(OUT_DIR, "results_original.png"))

    # ── Experiment 2: Augmented ───────────────────────────────────────────────
    res_aug = run_experiment(
        "Augmented (1M)", X_train_a, X_test_a, y_train_a, y_test_a,
        xgb_device, lgbm_device,
    )
    sum_aug = make_summary(res_aug, "Augmented")
    print_reports(res_aug, y_test_a, "Augmented (1M)")
    plot_experiment(res_aug, sum_aug, y_test_a,
                    "Augmented Dataset (1M rows)",
                    os.path.join(OUT_DIR, "results_augmented.png"))

    # ── Head-to-head comparison chart ────────────────────────────────────────
    print("\n📊 Generating comparison chart...")
    plot_comparison(sum_orig, sum_aug,
                    os.path.join(OUT_DIR, "results_comparison.png"))

    # ── Combined CSV report ───────────────────────────────────────────────────
    combined = pd.concat([sum_orig, sum_aug], ignore_index=True)
    csv_path = os.path.join(OUT_DIR, "results_all_models.csv")
    combined.to_csv(csv_path, index=False)
    print(f"  💾  Full results saved → {csv_path}")

    # ── Console summary tables ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  ORIGINAL DATASET — Ranked by PR-AUC")
    print(f"{'='*65}")
    print(sum_orig.to_string())

    print(f"\n{'='*65}")
    print("  AUGMENTED DATASET — Ranked by PR-AUC")
    print(f"{'='*65}")
    print(sum_aug.to_string())

    # ── Winner ────────────────────────────────────────────────────────────────
    announce_winner(combined)

    total = time.time() - wall_start
    print(f"⏱️  Total wall-clock time: {total/60:.1f} min\n")