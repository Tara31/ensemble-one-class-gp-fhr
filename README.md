# Ensemble One-Class Gaussian Processes for Fetal Heart Rate Anomaly Detection

## Overview

This repository contains the code for our one-class Gaussian Process framework for anomaly detection in fetal heart rate (FHR) data.

FHR monitoring is important for assessing fetal well-being, but detecting pathological patterns remains difficult because labeled abnormal cases are limited. In this project, the models are trained only on healthy **CAT-1** cases and used to detect deviations associated with **CAT-3** cases.

## Method Summary

The pipeline includes:

1. **Preprocessing**
   - Winsorization
   - Feature engineering (LF/HF shares)
   - Log and logit transformations
   - Yeo–Johnson transformation
   - Robust scaling

2. **Per-feature modeling**
   - Each feature is modeled using a Gaussian Process conditioned on the remaining features
   - Anomaly score is computed as a standardized residual

3. **Ensemble methods**
   - Fraction of features exceeding a threshold
   - Fisher’s method for p-value combination

4. **Evaluation**
   - AUROC
   - Average Precision (AUPRC)
   - Precision / Recall / F1-score

## Dataset

We use a publicly available fetal monitoring dataset:

https://preana-fo.ece.stonybrook.edu/database.html

### Expected Input Format
- Tabular feature file (for example, Excel)
- Required column: `cat`
  - `CAT-1` = healthy
  - `CAT-3` = pathological

## Project Structure

```text
.
├── scripts/
│   └── run_ocgp_pipeline.py
├── data/
│   └── README.md
├── results/
├── src/
├── requirements.txt
└── README.md
```

## Installation

Install the required Python packages:

pip install -r requirements.txt

---

## How to Run

To run the full pipeline:

python scripts/run_ocgp_pipeline.py

### Notes
- Update the input file path inside the script
- Input should be a tabular file with a `cat` column
- Each row should correspond to one FHR segment
- Results are saved in the `results/` folder

---

## Author

Taraneh Ghanbari Azarnir  
PhD Candidate, Electrical Engineering, Stony Brook University  

Feel free to reach out for questions or collaboration.
