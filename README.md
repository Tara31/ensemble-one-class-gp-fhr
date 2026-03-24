# Ensemble One-Class Gaussian Processes for Fetal Heart Rate Anomaly Detection
--

## Overview

Fetal heart rate (FHR) monitoring is critical for identifying fetal distress, but anomaly detection remains challenging due to limited labeled pathological data.

We propose a **one-class learning framework** based on:
- Per-feature **Gaussian Process regression models**
- **Anomaly scoring via standardized residuals**
- An **ensemble strategy** combining multiple feature-wise detectors

The approach is trained only on **healthy (CAT-1)** cases and detects deviations corresponding to **pathological (CAT-3)** conditions.

---

## Method Summary

1. **Preprocessing**
   - Winsorization (robust to outliers)
   - Feature engineering (LF/HF shares)
   - Log / logit transformations
   - Yeo–Johnson transformation (train-only)
   - Robust scaling

2. **Per-feature modeling**
   - Each feature is predicted using a Gaussian Process conditioned on all other features
   - Anomaly score = standardized residual (|z-score|)

3. **Ensemble methods**
   - Fraction of features exceeding threshold
   - Fisher’s method for p-value combination

4. **Evaluation**
   - AUROC
   - Average Precision (AUPRC)
   - Precision / Recall / F1

---

## Dataset

We use a publicly available fetal monitoring dataset:

🔗 https://preana-fo.ece.stonybrook.edu/database.html

Expected input format:
- Tabular feature file (e.g., Excel)
- Column: `cat`
  - `CAT-1` → healthy
  - `CAT-3` → pathological

---

## Project Structure

    .
    ├── scripts/
    │   └── run_ocgp_pipeline.py   # Main pipeline
    ├── data/
    │   └── README.md              # Data instructions
    ├── requirements.txt
    └── README.md
---

## Installation

Install required Python packages:

```bash
pip install -r requirements.txt
