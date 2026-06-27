"""FastAPI backend for the fraud detection demo.

Loads the trained pipeline once at startup and exposes it over a small JSON API the
frontend talks to. The model is a self-contained sklearn/imblearn pipeline, so every
endpoint just hands it a DataFrame with the right columns and reads predict_proba.

Run:  uvicorn api.main:app --reload --port 8000   (from the project root)
"""

import io
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REGISTRY = MODELS_DIR / "registry"

# --- Load artifacts once at import time ---------------------------------------------
META = json.loads((MODELS_DIR / "metadata.json").read_text())
FEATURES: list[str] = META["features"]
SAMPLES = pd.read_csv(MODELS_DIR / "holdout_samples.csv")

# Every trained pipeline, keyed by display name ("LightGBM + weighted", ...). Falls back
# to the single best_model.joblib if a registry wasn't produced (older training run).
MODEL_META: dict[str, dict] = META.get("models") or {
    META["model_name"]: {
        "model_name": META["model_name"], "slug": "best",
        "threshold": META["threshold"], "test_metrics": META["test_metrics"],
        "threshold_curve": META["threshold_curve"],
    }
}
DEFAULT_MODEL: str = META.get("default_model", META["model_name"])


def _load(entry: dict):
    path = REGISTRY / f"{entry['slug']}.joblib"
    return joblib.load(path if path.exists() else MODELS_DIR / "best_model.joblib")


MODELS = {name: _load(entry) for name, entry in MODEL_META.items()}


def _resolve(model: str | None) -> str:
    """Validate a requested model name and fall back to the default when none is given."""
    if model is None:
        return DEFAULT_MODEL
    if model not in MODELS:
        raise HTTPException(404, f"Unknown model '{model}'. "
                                 f"Available: {', '.join(MODELS)}")
    return model

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0")

# The Next.js dev server runs on a different port, so the browser needs CORS permission.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only; lock this down for a real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/response models --------------------------------------------------------
class Transaction(BaseModel):
    """One transaction as a mapping of feature name -> value (V1..V28, Amount)."""
    features: dict[str, float]


class BatchRequest(BaseModel):
    transactions: list[dict[str, float]]
    model: str | None = Field(default=None, description="Which model to score with")
    threshold: float | None = Field(default=None, description="Override the default cut-off")


def _resolve_threshold(model_name: str, threshold: float | None) -> float:
    """Fall back to the selected model's own tuned threshold when none is supplied."""
    return MODEL_META[model_name]["threshold"] if threshold is None else threshold


def _score(df: pd.DataFrame, model_name: str, threshold: float) -> list[dict]:
    """Score a DataFrame with the chosen model, labelling each row against the threshold."""
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing features: {missing[:5]}"
                                 f"{'...' if len(missing) > 5 else ''}")
    proba = MODELS[model_name].predict_proba(df[FEATURES])[:, 1]
    return [
        {"fraud_probability": round(float(p), 6),
         "is_fraud": bool(p >= threshold),
         "threshold": threshold,
         "model": model_name}
        for p in proba
    ]


# --- Endpoints ----------------------------------------------------------------------
@app.get("/models")
def list_models() -> dict:
    """List every available model with headline metrics, so the UI can build a picker.

    Sorted by PR-AUC (best first); the default model is flagged.
    """
    items = [
        {
            "model_name": name,
            "default_threshold": entry["threshold"],
            "is_default": name == DEFAULT_MODEL,
            "test_metrics": entry["test_metrics"],
        }
        for name, entry in MODEL_META.items()
    ]
    items.sort(key=lambda m: m["test_metrics"]["PR_AUC"], reverse=True)
    return {"default_model": DEFAULT_MODEL, "models": items}


@app.get("/model-info")
def model_info(model: str | None = None) -> dict:
    """Everything the frontend needs to describe one model: name, metrics, feature
    order, default threshold, and the threshold-sweep curve behind the slider.

    Pass ?model=<name> to describe a specific model; omit it for the default.
    """
    name = _resolve(model)
    entry = MODEL_META[name]
    return {
        "model_name": name,
        "features": FEATURES,
        "default_threshold": entry["threshold"],
        "test_metrics": entry["test_metrics"],
        "threshold_curve": entry["threshold_curve"],
        "n_train": META["n_train"], "n_val": META["n_val"], "n_test": META["n_test"],
    }


@app.get("/samples")
def samples(kind: str = "any", n: int = 1) -> list[dict]:
    """Return random held-out transactions to load into the tester.

    kind = 'fraud' | 'legit' | 'any'. Each sample includes its true label so the UI can
    show whether the model got it right.
    """
    pool = SAMPLES
    if kind == "fraud":
        pool = SAMPLES[SAMPLES["Class"] == 1]
    elif kind == "legit":
        pool = SAMPLES[SAMPLES["Class"] == 0]
    if pool.empty:
        raise HTTPException(404, f"No samples of kind '{kind}'")

    picked = pool.sample(min(n, len(pool)))
    return [
        {"features": {f: float(row[f]) for f in FEATURES},
         "true_label": int(row["Class"]),
         "model_probability": float(row["fraud_probability"])}
        for _, row in picked.iterrows()
    ]


@app.post("/predict")
def predict(txn: Transaction, model: str | None = None, threshold: float | None = None) -> dict:
    """Score a single transaction with the chosen model (default when unspecified)."""
    name = _resolve(model)
    t = _resolve_threshold(name, threshold)
    return _score(pd.DataFrame([txn.features]), name, t)[0]


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> dict:
    """Score a list of transactions and summarise how many were flagged."""
    name = _resolve(req.model)
    t = _resolve_threshold(name, req.threshold)
    results = _score(pd.DataFrame(req.transactions), name, t)
    flagged = sum(r["is_fraud"] for r in results)
    return {"count": len(results), "flagged": flagged, "threshold": t,
            "model": name, "results": results}


@app.post("/predict/csv")
async def predict_csv(
    file: UploadFile = File(...), model: str | None = None, threshold: float | None = None
) -> dict:
    """Score an uploaded CSV. Returns a summary plus per-row predictions.

    Accepts the raw Kaggle format (with or without Time/Class columns) — only the model
    features are used, and a Class column, if present, is echoed back for comparison.
    """
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    name = _resolve(model)
    t = _resolve_threshold(name, threshold)
    results = _score(df, name, t)

    rows = []
    for i, r in enumerate(results):
        row = {"index": i, **r, "Amount": float(df.iloc[i].get("Amount", float("nan")))}
        if "Class" in df.columns:
            row["true_label"] = int(df.iloc[i]["Class"])
        rows.append(row)

    flagged = sum(r["is_fraud"] for r in results)
    summary = {"filename": file.filename, "count": len(rows),
               "flagged": flagged, "threshold": t, "model": name}
    if "Class" in df.columns:  # if labels were provided, report accuracy of the flags
        tp = sum(r["is_fraud"] and rows[i]["true_label"] == 1 for i, r in enumerate(results))
        actual = int(df["Class"].sum())
        summary.update({"actual_fraud": actual, "caught": tp})
    return {"summary": summary, "rows": rows}


@app.get("/")
def root() -> dict:
    return {"status": "ok", "default_model": DEFAULT_MODEL,
            "available_models": list(MODELS),
            "docs": "/docs", "endpoints": ["/models", "/model-info", "/samples", "/predict",
                                           "/predict/batch", "/predict/csv"]}
