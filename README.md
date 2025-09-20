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
```bash
# Python 3.10+ recommended
pip install -r requirements.txt
