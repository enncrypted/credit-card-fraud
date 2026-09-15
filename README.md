# Credit Card Fraud Detection: Logistic Regression vs. XGBoost

## Overview

This project implements a leakage-free, reproducible machine learning pipeline for detecting credit card fraud using the Kaggle **Credit Card Fraud Detection dataset**. 

The primary objective is to evaluate and compare two distinct modeling paradigms on an identical held-out test set:
1. **Logistic Regression (Baseline)**: A linear classifier with feature preprocessing (median imputation, IQR outlier clipping, and Yeo-Johnson PowerTransformation on non-PCA features) combined with `class_weight="balanced"`.
2. **XGBoost (Main Focus)**: A gradient-boosted decision tree algorithm trained directly on raw cleaned features using `scale_pos_weight` to handle severe class imbalance without artificial oversampling.

---

## Key Features & Highlights

- **Data Cleaning & Deduplication**: Identification and removal of 1,081 duplicate transactions to prevent data leakage.
- **Leakage-Free Pipeline**: Strict train/test split (80/20 stratified split) prior to any feature transformation or hyperparameter tuning.
- **Resampling-Free Imbalance Handling**: Utilizes cost-sensitive class weighting (`class_weight="balanced"` and `scale_pos_weight = 599.48`) instead of synthetic oversampling techniques like SMOTE.
- **Imbalance-Conscious Evaluation**: Comprehensive performance evaluation using **PR-AUC (Precision-Recall Area Under Curve)**, **F1-Score**, **ROC-AUC**, **Precision**, **Recall**, and full **Confusion Matrices**.

---

## Dataset Information

The experiments use the Credit Card Fraud Detection dataset available on Kaggle.

- **Total Transactions**: 284,807 (283,726 unique transactions following deduplication)
- **Class Breakdown**:
  - **Legitimate (Class 0)**: 283,253 (99.83%)
  - **Fraudulent (Class 1)**: 473 (0.17%)
- **Features**: 30 numerical input features
  - `V1`–`V28`: PCA-transformed numerical features
  - `Time`: Seconds elapsed between each transaction and the first transaction
  - `Amount`: Transaction amount
  - `Class`: Binary target variable (0 = Non-fraud, 1 = Fraud)

---

## Project Structure

```
.
├── data/
│   └── creditcard.csv                 # Kaggle Credit Card Fraud dataset
├── fraud_detection_comparison.ipynb   # Main Jupyter Notebook with complete analysis & pipeline
├── requirements.txt                   # Project dependencies
└── README.md                          # Documentation
```

---

## Experimental Pipeline

1. **Exploratory Analysis & Integrity Check**: Inspected data shape, distributions, memory usage, dtypes, and range bounds.
2. **Data Cleaning**: Removed 1,081 duplicate records and verified zero missing or invalid values.
3. **Stratified Train/Test Split**: 80/20 stratified split (Train: 226,980 rows with 0.1665% fraud; Test: 56,746 rows with 0.1674% fraud).
4. **Logistic Regression Baseline**:
   - Pipeline: `SimpleImputer` (median), `IQRClipper` (outliers), and `PowerTransformer` (Yeo-Johnson) applied to `Time` and `Amount`; `V1`–`V28` passed through unchanged.
   - Class Imbalance: `class_weight="balanced"`.
   - Hyperparameter Tuning: 5-fold Stratified K-Fold `GridSearchCV` optimizing PR-AUC (`average_precision`).
5. **XGBoost Classifier (Main Focus)**:
   - Tree-based model trained on raw cleaned features (scaling and power transformations omitted as tree splits are scale-invariant).
   - Class Imbalance: `scale_pos_weight = 599.48` computed strictly from the training set ($N_{\text{negative}} / N_{\text{positive}}$).
   - Hyperparameters: `n_estimators=500`, `max_depth=5`, `learning_rate=0.05`, `eval_metric="aucpr"`.
6. **Evaluation**: Both models evaluated on the identical 56,746 held-out test set instances.

---

## Experimental Results

All evaluations were conducted on the exact same 20% test split (56,746 total transactions: 56,651 Non-fraud, 95 Fraud).

### Performance Metrics Summary

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** (Baseline) | 5.51% (0.0551) | **87.37%** (0.8737) | 10.38% (0.1038) | 96.20% (0.9620) | 67.05% (0.6705) |
| **XGBoost** (Main Focus) | **92.59%** (0.9259) | 78.95% (0.7895) | **85.23%** (0.8523) | **98.25%** (0.9825) | **82.05%** (0.8205) |

### Confusion Matrix Detailed Breakdown

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 55,229 | 1,422 | **12** | **83** |
| **XGBoost** | **56,645** | **6** | 20 | 75 |

---

## Key Findings & Insights

1. **XGBoost Delivers Superior Precision and Overall Performance**:
   - **XGBoost** achieved an **F1-Score of 85.23%** compared to Logistic Regression's **10.38%**, alongside a significantly higher **PR-AUC (82.05% vs. 67.05%)**.
   - Logistic Regression detected 8 additional fraud cases (Recall: 87.37% vs 78.95%), but incurred **1,422 false positives** (Precision of only 5.51%), which is impractical in real-world fraud detection environments due to customer friction and operational investigation costs.
   - **XGBoost** drastically minimized false positives to just **6 instances**, yielding an exceptional **Precision of 92.59%**.

2. **Cost-Sensitive Weighting Eliminates Resampling Artifacts**:
   - Setting `scale_pos_weight = 599.48` in XGBoost proved highly effective at handling the ~1:600 class imbalance without requiring synthetic oversampling (e.g., SMOTE), preventing model overfitting on artificial samples.

3. **Model-Agnostic Feature Engineering Strategy**:
   - Non-linear transformations (Yeo-Johnson power transformation) and outlier clipping were necessary for Logistic Regression due to heavy skewness in non-PCA features (`Time` and `Amount`).
   - XGBoost required no feature scaling or power transformation, streamlining the production inference pipeline.

---

## Getting Started

### Prerequisites

Ensure Python 3.10+ is installed.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/enncrypted/credit-card-fraud.git
   cd credit-card-fraud
   ```

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup**:
   Place the Kaggle `creditcard.csv` dataset in the `data/` folder:
   ```
   ./data/creditcard.csv
   ```

4. **Run Notebook**:
   ```bash
   jupyter notebook fraud_detection_comparison.ipynb
   ```

---

## License

This project is provided for research and educational purposes.