"""The model comparison grid: 4 algorithms x 2 imbalance strategies = 8 pipelines.

Each entry is a full pipeline (scaler -> optional SMOTE -> classifier), so the saved
winner is a single self-contained object the API can call with a raw DataFrame.

The imbalance strategy is the interesting axis:
  * "weighted" tells the algorithm frauds count more, via class_weight / scale_pos_weight.
  * "smote" instead synthesises new minority examples. SMOTE lives INSIDE the pipeline so
    it only ever resamples the training fold during fit — resampling before the split is
    the classic mistake that leaks synthetic copies of test frauds into training.
"""

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data import RANDOM_STATE

# LightGBM settings found by src/tune.py (randomised search, scored on the validation set).
# The pattern the search discovered: many *shallow* trees (num_leaves=15) at a moderate
# learning rate generalise better than fewer deep ones on this data. Re-run tune.py to
# reproduce these numbers.
TUNED_LIGHTGBM = {
    "n_estimators": 1500,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 30,
    "colsample_bytree": 0.7,
    "reg_lambda": 0.0,
}


def _classifiers(scale_pos_weight: float) -> dict:
    """The four base algorithms, each configured for the imbalance via class weights."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, n_jobs=-1,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=RANDOM_STATE
        ),
        "LightGBM": LGBMClassifier(
            **TUNED_LIGHTGBM, class_weight="balanced", n_jobs=-1, verbose=-1,
            random_state=RANDOM_STATE
        ),
    }


def _plain_classifiers() -> dict:
    """The same four algorithms with NO class weighting — for the SMOTE variants,
    where the resampling already balances the classes, so double-correcting would
    over-weight the minority."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, n_jobs=-1,
            eval_metric="aucpr", random_state=RANDOM_STATE
        ),
        "LightGBM": LGBMClassifier(
            **TUNED_LIGHTGBM, n_jobs=-1, verbose=-1, random_state=RANDOM_STATE
        ),
    }


def build_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    """Return all 8 named pipelines, ready to fit.

    Keys read like "XGBoost + weighted" / "XGBoost + SMOTE" so the comparison table is
    self-explaining.
    """
    models: dict[str, Pipeline] = {}

    for name, clf in _classifiers(scale_pos_weight).items():
        models[f"{name} + weighted"] = Pipeline([
            ("scale", StandardScaler()),
            ("clf", clf),
        ])

    for name, clf in _plain_classifiers().items():
        models[f"{name} + SMOTE"] = Pipeline([
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", clf),
        ])

    # Keep DataFrames (with column names) flowing through every step, so the classifiers
    # see the real V1..Amount names at both fit and predict — otherwise the scaler emits a
    # nameless array and the tree models warn about missing feature names when scored later.
    for pipeline in models.values():
        pipeline.set_output(transform="pandas")

    return models
