# Fraud Detection Work

Final-year project: **Machine Learning for Credit Card Fraud Detection** by Michael Brozhko.  
End-to-end pipeline covering dataset assembly, preprocessing, feature engineering, dimensionality checks, model training with hyperparameter search, and evaluation on **original** vs **oversampled** data.

---

## Table of Contents
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Setup](#setup)
- [Workflow](#workflow)
- [Feature Engineering](#feature-engineering)
- [Dimensionality & Selection](#dimensionality--selection)
- [Models & Tuning](#models--tuning)
- [Results (Summary)](#results-summary)
- [How to Run](#how-to-run)
- [Key Functions](#key-functions)
- [Notes & Assumptions](#notes--assumptions)

---

## Motivation
Fraud datasets are **highly imbalanced** and may come **pre-split** in ways that bias results.  
This project **recombines** the original train/test CSVs and performs a **fresh randomized split** to ensure fair evaluation and reproducibility.

---

## Dataset
Two CSVs are combined into a single `combined_csv.csv`, then split with `train_test_split`.  
Columns include transaction metadata (time, amount, merchant/customer geolocation), demographics, and the target label `is_fraud`.

> **Path note:** Update the dataset folder path before running the combine step.

---

## Setup
# Python 3.10+ recommended
pip install -r requirements.txt


Core libraries: pandas, numpy, scipy, matplotlib, scikit-learn, imbalanced-learn, xgboost.

## Workflow

- Combine CSVs → combined_csv.csv
- Load & inspect datatypes and nulls (none initially)
- Preprocess
- Drop high-cardinality / ID-like or free-text fields (e.g., cc_num, trans_num, names, streets)
- Convert/expand datetime → hour, day (0–6), month
- Derive age from dob
- Feature engineering (see below)
- Encode categorical features (category, gender) via One-Hot Encoding
- Train/test split (random_state=42)
- Outlier guardrails (IQR checks for age, amt; cap age at 98 in this study)
- PCA diagnostics to understand variance contribution
- Modeling with hyperparameter search (GridSearchCV, recall-focused scoring)
- Evaluation on:
    - Original (unsampled) data
    - Random over-sampled data (10:1 majority:minority)

## Feature Engineering

- Temporal: hour, day, month
- Age: age = current_year − dob.year
- Geo distance: merchant_customer_distance = sqrt((lat − merch_lat)^2 + (long − merch_long)^2)
- Final numeric subset used for core models: ['amt','zip','lat','long','age','hour','day','month','merchant_customer_distance']

## Dimensionality & Selection

PCA scree/variance analysis guided a compact numeric feature set (above).

One-hot features for category/gender were explored; final primary models emphasize the numeric subset for robustness and speed.

## Models & Tuning

All models run in imblearn pipelines with scaling and are tuned via GridSearchCV, optimizing recall of the fraud class (1).

KNN: n_neighbors ∈ {4,5,6}, weights='distance'

XGBoost: depth, learning rate, estimators, L1/L2 regularization

Random Forest: estimators, max depth

MLP: hidden sizes, activation (tanh/relu), max_iter

Scoring uses make_scorer(recall_score, pos_label=1) to prioritize catching fraud.

## Results (Summary)
Unsampled (original imbalance)

XGBoost — Accuracy: 0.9990, Fraud Recall: 0.85, Precision: 0.96

Random Forest — Accuracy: 0.9987, Fraud Recall: 0.80, Precision: 0.95

KNN — Accuracy: 0.9985, Fraud Recall: 0.82, Precision: 0.89

MLP — Accuracy: 0.9966, Fraud Recall: 0.49, Precision: 0.79

RandomOverSampler (10:1)

XGBoost — Accuracy: 0.9988, Fraud Recall: 0.87, Precision: 0.91

Random Forest — Accuracy: 0.9989, Fraud Recall: 0.87, Precision: 0.92

KNN — Accuracy: 0.9981, Fraud Recall: 0.84, Precision: 0.81

MLP — Accuracy: 0.9944, Fraud Recall: 0.85, Precision: 0.48

Takeaways

Resampling improves minority recall (fraud detection) with negligible impact on overall accuracy.

Tree-based models (XGBoost / RF) provide the best recall–precision balance.

KNN is competitive but heavier at scale; MLP needs further tuning/regularization for precision.

How to Run
# 1) Combine CSVs (adjust path)
import os, glob, pandas as pd
os.chdir("/path/to/Dataset")
combined_csv = pd.concat([pd.read_csv(f) for f in glob.glob("*.csv")])
combined_csv.to_csv("combined_csv.csv", index=False)

# 2) Load, preprocess, encode, split (see script/notebook)
import pandas as pd
df = pd.read_csv("combined_csv.csv")
# ... feature engineering, OHE, split → X_train, y_train, X_test, y_test

# 3) Train/evaluate
from imblearn.over_sampling import RandomOverSampler
runPipelines(X_train, y_train, X_test, y_test, sampler=None)  # unsampled
runPipelines(X_train, y_train, X_test, y_test,
             sampler=RandomOverSampler(random_state=42, sampling_strategy=0.1))

Key Functions

pca_analysis(features) — standardizes, fits PCA, plots scree, prints explained variance ratio.

modelPerformance(pipe, name, parameters, X_train, y_train, X_test, y_test) — grid search (recall scorer), fits best model, prints train/test metrics, plots precision/recall/F1.

runPipelines(X_train, y_train, X_test, y_test, sampler) — optional re-sampling, then runs/tunes KNN, XGBoost, RF, and MLP.

Notes & Assumptions

Reproducibility: fixed random_state=42 where applicable.

Leakage avoidance: dropped identifiers and free-text personal fields.

Scaling: standardization inside pipelines (keeps setup consistent across models).

Imbalance handling: RandomOverSampler at 0.1 (≈10:1) gave the best recall/precision trade-off in this study.

Plots: classification bar charts generated with matplotlib.
