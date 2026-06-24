"""
Credit Card Fraud Detection - Multi-Model Comparison
=====================================================
Trains 8 ML models and compares them side-by-side using metrics
suited for highly imbalanced datasets (fraud is ~0.17% of data).
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')
 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score, precision_score, recall_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.pipeline import Pipeline
 
# ── Models ──────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
 
# ── Imbalanced-learn ─────────────────────────────────────────────────────────
# pip install imbalanced-learn  (if not installed)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
 
 
# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data(csv_path="creditcardfraud/creditcard.csv"):
    print("📂 Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape}")
    print(f"   Fraud rate: {df['Class'].mean()*100:.3f}%\n")
    return df
 
 
# =============================================================================
# 2. PREPROCESS
# =============================================================================
def preprocess(df):
    """Scale Amount & Time; drop raw originals."""
    df = df.copy()
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
 
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y
 
 
# =============================================================================
# 3. MODEL DEFINITIONS
# =============================================================================
def get_models():
    """
    Returns a dict of { model_name: estimator }.
    Every model is wrapped so it works on the SMOTE-resampled training set.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, class_weight='balanced', random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            n_jobs=-1, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, class_weight='balanced',
            n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.1,
            scale_pos_weight=577,          # ~ratio of negatives/positives
            use_label_encoder=False,
            eval_metric='aucpr',
            random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, learning_rate=0.1,
            class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1
        ),
    }
    return models
 
 
# =============================================================================
# 4. TRAIN & EVALUATE
# =============================================================================
def train_and_evaluate(X_train, X_test, y_train, y_test, models):
    results = {}
    smote = SMOTE(random_state=42)
 
    print("="*60)
    print("  TRAINING MODELS")
    print("="*60)
 
    for name, model in models.items():
        print(f"\n🔧 {name} ...", end=" ", flush=True)
        t0 = time.time()
 
        # Apply SMOTE only on training set
        X_res, y_res = smote.fit_resample(X_train, y_train)
 
        model.fit(X_res, y_res)
        elapsed = time.time() - t0
 
        # Predictions
        y_pred  = model.predict(X_test)
        y_prob  = (model.predict_proba(X_test)[:, 1]
                   if hasattr(model, 'predict_proba')
                   else model.decision_function(X_test))
 
        # Metrics
        roc_auc  = roc_auc_score(y_test, y_prob)
        pr_auc   = average_precision_score(y_test, y_prob)
        f1       = f1_score(y_test, y_pred)
        precision= precision_score(y_test, y_pred)
        recall   = recall_score(y_test, y_pred)
 
        results[name] = {
            "model":     model,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "ROC-AUC":   roc_auc,
            "PR-AUC":    pr_auc,
            "F1":        f1,
            "Precision": precision,
            "Recall":    recall,
            "Time (s)":  round(elapsed, 2),
        }
        print(f"✅  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  "
              f"F1={f1:.4f}  [{elapsed:.1f}s]")
 
    return results
 
 
# =============================================================================
# 5. SUMMARY TABLE
# =============================================================================
def build_summary(results):
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":      name,
            "ROC-AUC":    round(r["ROC-AUC"],   4),
            "PR-AUC":     round(r["PR-AUC"],     4),
            "F1":         round(r["F1"],          4),
            "Precision":  round(r["Precision"],   4),
            "Recall":     round(r["Recall"],      4),
            "Time (s)":   r["Time (s)"],
        })
    df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    df.index += 1   # rank starts at 1
    return df
 
 
# =============================================================================
# 6. VISUALISATIONS
# =============================================================================
def plot_results(results, summary_df, X_test, y_test):
    n_models = len(results)
    names    = list(results.keys())
    colors   = plt.cm.tab10(np.linspace(0, 1, n_models))
 
    fig = plt.figure(figsize=(22, 26))
    fig.suptitle("Credit Card Fraud Detection — Model Comparison", fontsize=18, fontweight='bold', y=0.98)
 
    # ── (A) Metrics bar chart ────────────────────────────────────────────────
    ax1 = fig.add_subplot(4, 2, (1, 2))
    metrics = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall"]
    x = np.arange(len(metrics))
    width = 0.8 / n_models
    for i, (name, r) in enumerate(results.items()):
        vals = [r[m] for m in metrics]
        ax1.bar(x + i*width, vals, width, label=name, color=colors[i], alpha=0.85)
    ax1.set_xticks(x + width*(n_models-1)/2)
    ax1.set_xticklabels(metrics, fontsize=12)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Score")
    ax1.set_title("All Metrics by Model", fontsize=13)
    ax1.legend(loc='upper right', ncol=2, fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
 
    # ── (B) ROC curves ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(4, 2, 3)
    for (name, r), c in zip(results.items(), colors):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax2.plot(fpr, tpr, color=c, lw=1.5,
                 label=f"{name} ({r['ROC-AUC']:.3f})")
    ax2.plot([0,1],[0,1],'k--', lw=0.8)
    ax2.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
 
    # ── (C) Precision-Recall curves ──────────────────────────────────────────
    ax3 = fig.add_subplot(4, 2, 4)
    for (name, r), c in zip(results.items(), colors):
        from sklearn.metrics import precision_recall_curve
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        ax3.plot(rec, prec, color=c, lw=1.5,
                 label=f"{name} ({r['PR-AUC']:.3f})")
    ax3.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)
 
    # ── (D) Confusion matrices for top-4 models (by PR-AUC) ─────────────────
    top4 = summary_df.head(4)["Model"].tolist()
    for idx, model_name in enumerate(top4):
        ax = fig.add_subplot(4, 4, 9 + idx)
        cm = confusion_matrix(y_test, results[model_name]["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    cbar=False, xticklabels=['Normal','Fraud'],
                    yticklabels=['Normal','Fraud'])
        ax.set_title(f"{model_name}\n(Confusion Matrix)", fontsize=8)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("Actual", fontsize=7)
 
    # ── (E) F1 vs Training Time scatter ──────────────────────────────────────
    ax5 = fig.add_subplot(4, 2, 7)
    for (name, r), c in zip(results.items(), colors):
        ax5.scatter(r["Time (s)"], r["F1"], color=c, s=120, zorder=3, label=name)
        ax5.annotate(name, (r["Time (s)"], r["F1"]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax5.set(xlabel="Training Time (s)", ylabel="F1 Score",
            title="F1 vs Training Time")
    ax5.grid(alpha=0.3)
 
    # ── (F) Summary table ────────────────────────────────────────────────────
    ax6 = fig.add_subplot(4, 2, 8)
    ax6.axis('off')
    tbl = ax6.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center', loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.6)
    ax6.set_title("Ranked Summary (↓ PR-AUC)", fontsize=11, pad=12)
 
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("fraud_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n📊 Chart saved → fraud_model_comparison.png")
 
 
# =============================================================================
# 7. DETAILED REPORT
# =============================================================================
def print_classification_reports(results, y_test):
    print("\n" + "="*60)
    print("  DETAILED CLASSIFICATION REPORTS")
    print("="*60)
    for name, r in results.items():
        print(f"\n{'─'*40}")
        print(f"  {name}")
        print('─'*40)
        print(classification_report(y_test, r["y_pred"],
                                    target_names=["Normal", "Fraud"]))
 
 
# =============================================================================
# 8. WINNER ANNOUNCEMENT
# =============================================================================
def announce_winner(summary_df):
    print("\n" + "🏆"*30)
    best = summary_df.iloc[0]
    print(f"\n  BEST MODEL (by PR-AUC): {best['Model']}")
    print(f"    ROC-AUC   : {best['ROC-AUC']}")
    print(f"    PR-AUC    : {best['PR-AUC']}")
    print(f"    F1 Score  : {best['F1']}")
    print(f"    Precision : {best['Precision']}")
    print(f"    Recall    : {best['Recall']}")
    print(f"    Train time: {best['Time (s)']}s")
    print("\n  💡 NOTE: For fraud detection, PR-AUC is preferred over ROC-AUC")
    print("     because the dataset is highly imbalanced (~0.17% fraud).")
    print("🏆"*30)
 
 
# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = load_data("creditcardfraud/creditcard.csv")
 
    # ── 2. Preprocess ────────────────────────────────────────────────────────
    X, y = preprocess(df)
 
    # ── 3. Split  (stratified so both splits keep the fraud ratio) ───────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")
    print(f"Fraud in test set: {y_test.sum()} ({y_test.mean()*100:.3f}%)\n")
 
    # ── 4. Models ────────────────────────────────────────────────────────────
    models = get_models()
 
    # ── 5. Train & evaluate ──────────────────────────────────────────────────
    results = train_and_evaluate(X_train, X_test, y_train, y_test, models)
 
    # ── 6. Summary ───────────────────────────────────────────────────────────
    summary_df = build_summary(results)
    print("\n\n" + "="*60)
    print("  RESULTS SUMMARY  (ranked by PR-AUC)")
    print("="*60)
    print(summary_df.to_string())
 
    # ── 7. Reports ───────────────────────────────────────────────────────────
    print_classification_reports(results, y_test)
 
    # ── 8. Visualise ─────────────────────────────────────────────────────────
    plot_results(results, summary_df, X_test, y_test)
 
    # ── 9. Winner ────────────────────────────────────────────────────────────
    announce_winner(summary_df)
