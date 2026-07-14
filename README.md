# Credit Card Fraud Detection using Genetic Algorithm (GA) Feature Selection

## Overview

This project implements a machine learning pipeline for detecting fraudulent credit card transactions using the European Credit Card Fraud Detection dataset.

The project evaluates the impact of:

- Baseline machine learning models
- SMOTE oversampling
- Genetic Algorithm (GA) based feature selection
- Multiple classifiers

The goal was to reproduce and evaluate claims from published work suggesting that Genetic Algorithm feature selection significantly improves fraud detection performance.

---

## Dataset

The experiments use the Credit Card Fraud Detection dataset available on Kaggle.

Dataset characteristics:

- 284,807 transactions
- 492 fraudulent transactions
- 30 numerical features
  - V1–V28 (PCA transformed)
  - Time
  - Amount
- Binary target
  - 0 = Legitimate
  - 1 = Fraud

---

## Project Structure

```
.
├── fraud_detection.ipynb
├── baseline.csv
├── oversampling.csv
├── feature_selection_oversampling_1.csv
├── feature_selection_oversampling_2.csv
├── feature_selection_oversampling_3.csv
├── feature_selection_oversampling_4.csv
├── feature_selection_oversampling_5.csv
├── requirements.txt
└── README.md
```

---

## Workflow

The experimental pipeline consists of the following steps:

1. Load dataset
2. Train/Test split
3. Feature scaling
4. Optional Genetic Algorithm feature selection
5. SMOTE oversampling (training set only)
6. Model training
7. Model evaluation

The classifiers evaluated are:

- Random Forest
- Artificial Neural Network (ANN)
- Decision Tree
- Logistic Regression
- Naive Bayes

---

## Evaluation Metrics

The following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Experimental Results

### Baseline

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|---------|----------|
| Random Forest | **99.95%** | **95.73%** | 75.68% | **84.53%** |
| ANN | 99.95% | 93.97% | 73.65% | 82.58% |
| Decision Tree | 99.92% | 77.86% | 73.65% | 75.69% |
| Logistic Regression | 99.91% | 85.05% | 61.49% | 71.37% |
| Naive Bayes | 97.80% | 6.04% | **80.41%** | 11.24% |

---

### Oversampling (SMOTE)

Random Forest remained the best-performing classifier.

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|---------|----------|
| Random Forest | **99.95%** | **89.23%** | **78.38%** | **83.45%** |

---

### GA Feature Selection + Oversampling

Five GA-generated feature subsets were evaluated.

Across all experiments:

- Random Forest accuracy dropped to approximately **93–94%**
- Precision dropped significantly
- F1-score decreased substantially
- No feature subset outperformed the baseline model

---

## Discussion

The experiments indicate that:

- SMOTE improves fraud detection performance by addressing class imbalance.
- The baseline Random Forest already performs exceptionally well on the Credit Card Fraud dataset.
- Genetic Algorithm feature selection did **not** improve classification performance.
- In this dataset, the PCA-transformed features already contain highly informative representations, leaving little room for feature selection to provide additional gains.

These findings differ from some published studies that report significant improvements after GA feature selection. The results suggest that the effectiveness of GA feature selection is highly dependent on dataset characteristics and experimental setup.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
fraud_detection.ipynb
```

Run all cells sequentially.

---

## Key Findings

- Baseline Random Forest achieved the best overall performance.
- SMOTE effectively mitigated class imbalance.
- Genetic Algorithm feature selection did not improve model performance.
- The computational cost of GA was not justified by the observed predictive performance.

---

## Future Work

Potential improvements include:

- Multi-objective GA optimization
- Fitness based on F1-score or PR-AUC instead of Accuracy
- Cross-validation inside the GA fitness function
- Hyperparameter optimization
- Evaluation on additional fraud detection datasets

---

## License

This project is intended for educational and research purposes.