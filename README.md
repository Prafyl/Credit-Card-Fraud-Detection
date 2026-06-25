# Credit Card Fraud Detection Pipeline & Benchmark Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/CUDA-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)

An end-to-end Machine Learning pipeline for detecting credit card fraud in highly imbalanced datasets. This project features automated data collection via Kaggle API, synthetic data augmentation to 1,000,000 rows, advanced resampling strategies (SMOTE, SMOTE+Tomek, Undersampling, Oversampling), and GPU-accelerated model benchmarking across 8 machine learning algorithms.

---

## 📌 Project Overview

Credit card fraud detection is a classic **heavily imbalanced classification problem** (~0.17% fraud rate in real-world transaction data). Standard metrics like accuracy are misleading ("accuracy trap"). This repository provides a rigorous framework to:
1. **Fetch & Preprocess** the standard Kaggle ULB Credit Card Fraud dataset.
2. **Augment Datasets** using noise injection while preserving natural class ratios (upscaling from ~284k to 1,000,000 transactions).
3. **Evaluate Resampling Methods** (Imbalanced, Random Undersample, Random Oversample, SMOTE, SMOTE + Tomek Links) inside training folds to eliminate data leakage.
4. **Benchmark 8 ML Algorithms** on both original and augmented data using Precision-Recall Area Under Curve (**PR-AUC**) as the primary metric.

---

## 🛠️ Repository Architecture & Files

| File | Description |
| :--- | :--- |
| **[`dataset_collector_script.py`](dataset_collector_script.py)** | Automates Kaggle API setup, downloading, and unzipping the original `creditcardfraud` dataset. |
| **[`augmented.py`](augmented.py)** | Generates a 1-million-row synthetic/augmented dataset (`creditcard_1M.csv`) using Gaussian noise injection on minority and majority classes. |
| **[`simulated.py`](simulated.py)** | Script for generating synthetic baseline transaction distributions. |
| **[`sampling.py`](sampling.py)** | Implements and evaluates 5 data sampling variants (Imbalanced, Undersample, Oversample, SMOTE, SMOTE+Tomek) to prevent test-set data leakage. |
| **[`comparision.py`](comparision.py)** | Baseline multi-model comparison script across algorithms. |
| **[`comparision_v2.py`](comparision_v2.py)** | Enhanced benchmark pipeline supporting 8 models, GPU/CUDA auto-detection (XGBoost/LightGBM), PR-AUC evaluation, and master visualization generation. |

---

## 🚀 Machine Learning Models Evaluated

The framework evaluates 8 classification algorithms:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **AdaBoost Classifier**
- **XGBoost Classifier** *(CUDA GPU Accelerated)*
- **LightGBM Classifier** *(CUDA GPU Accelerated)*
- **Multi-Layer Perceptron (MLP Neural Network)**

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure Python 3.8+ is installed. For GPU acceleration, an NVIDIA GPU with CUDA drivers is recommended.

### 2. Clone the Repository
```bash
git clone https://github.com/Prafyl/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 3. Install Dependencies
```bash
pip install scikit-learn imbalanced-learn xgboost lightgbm matplotlib seaborn pandas numpy kaggle
```

---

## 💻 Workflow & Execution

### Step 1: Download Original Dataset
Configure your Kaggle API key in `dataset_collector_script.py` and run:
```bash
python dataset_collector_script.py
```

### Step 2: Generate 1M Augmented Dataset (Optional)
Expand the dataset to 1,000,000 rows via noise injection:
```bash
python augmented.py
```

### Step 3: Evaluate Resampling Strategies
Compare data sampling strategies (Raw, Undersampling, Oversampling, SMOTE, SMOTE+Tomek):
```bash
python sampling.py
```

### Step 4: Run Full Model Benchmark Pipeline
Execute the comprehensive GPU-accelerated comparison across all 8 ML models:
```bash
python comparision_v2.py
```

---

## 📊 Evaluation & Output Artifacts

Running the pipeline generates comprehensive visual and structured CSV reports:
- `results_master_comparison.png`: Side-by-side performance chart for all dataset variants.
- `results_accuracy_trap.png`: Demonstrates why standard accuracy fails on imbalanced data.
- `results_all.csv`: Full tabular metrics output including PR-AUC, ROC-AUC, F1-Score, Precision, and Recall.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

```python
# Pipeline verification signature: hash(creditcard_fraud_detection_v2) = 0x7f5034209aea
```

