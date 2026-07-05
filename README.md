# Sentinel — Credit Card Fraud Detection

An end-to-end machine-learning project on the [ULB credit-card fraud
dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): exploratory analysis, a
model comparison that selects the best classifier, and a professional web app that serves it.

## What's here

```
creditcard.csv          the dataset (284,807 transactions, 0.17% fraud)
requirements.txt        Python dependencies
src/                    the machine-learning side (kept minimal + modular)
  data.py                 load, de-duplicate, split  (the only file that touches the CSV)
  plotting.py             shared, colour-blind-safe chart styling
  eda.py                  7 exploratory figures  -> reports/figures/
  models.py               the 8 candidate pipelines (4 algorithms x 2 imbalance strategies)
  tune.py                 reproduces the LightGBM hyperparameter search
  train.py                trains, compares, tunes each threshold, saves ALL models
reports/                model_comparison.csv + all figures
models/                 best_model.joblib + metadata.json + holdout_samples.csv
  registry/               every trained pipeline (one .joblib each), for live switching
api/                    FastAPI backend that serves the model
web/                    Vite + React + Tailwind frontend
```

## The model

The default winner is **LightGBM with class-weighting**, chosen by **PR-AUC** (average
precision) — the right metric when only 0.17% of transactions are fraud, since accuracy is
meaningless (always predicting "legit" scores 99.83%). On the held-out test set it reaches
**PR-AUC 0.828**. Each model's decision threshold is tuned to minimise total mistakes
(missed frauds + false alarms).

All eight candidate pipelines are saved, not just the winner: the web app has a **model
picker** and a **threshold slider** in its top bar, so you can score any transaction with any
model at any operating point and watch the verdict change live. Every scoring endpoint
accepts `?model=` and `?threshold=` (see `/docs`).

Key choices, all in the code:
- Duplicates dropped **before** splitting, so no row appears in both train and test.
- `Time` excluded from the model (it can't generalise beyond this 2-day sample).
- SMOTE lives **inside** the pipeline, so it only resamples the training fold — never leaks.

## Running it

### 1. Python side (already run once; re-run to regenerate everything)

```powershell
pip install -r requirements.txt
python src/eda.py      # writes the 7 EDA figures
python src/train.py    # trains all models, saves the best + metadata + samples
python src/tune.py      # (optional) reproduce the hyperparameter search
```

### 2. Backend API

```powershell
python -m uvicorn api.main:app --reload --port 8000
```
Interactive API docs at http://localhost:8000/docs

### 3. Frontend

```powershell
cd web
npm install
npm run dev
```
Open http://localhost:5173. The dev server proxies `/api` to the backend on port 8000, so
start the API first.

## The web app

A top control bar lets you choose **which model** scores and set the **decision threshold**;
both apply to all four screens below:

1. **Test one** — load a real held-out transaction (a known fraud or a known legit one),
   tweak any value, and score it. A paste-JSON toggle is there for power users. (Because
   V1–V28 are anonymised PCA components, loading real samples is the only honest way to
   demo them.)
2. **Batch** — build a set of transactions and score them in one request.
3. **Upload CSV** — score a whole file; if it has a `Class` column, the app reports how many
   frauds were caught.
4. **Bank dashboard** — metrics for the selected model, a live threshold slider, a confusion
   matrix that updates as you drag, a precision/recall-vs-threshold chart, and a ranked
   **all-models comparison** table. This is where the precision/recall trade-off becomes
   tangible.
