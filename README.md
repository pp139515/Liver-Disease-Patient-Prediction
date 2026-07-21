# Liver Disease Prediction using Machine Learning

A machine learning project that classifies patients as liver-diseased or non-liver-diseased using clinical and biochemical attributes, built with the CRISP-DM methodology.

## Overview

Liver disease is a major global health issue that, if caught early, can lead to significantly better patient outcomes. This project builds and compares multiple classification models to predict liver disease from demographic and biochemical test data, with the goal of supporting early, automated diagnosis.

Full write-up: see [`ML_ReportFinal.docx`](./ML_ReportFinal.docx) for the complete report (methodology, visualizations, and per-model results).

## Dataset

- **Source:** Indian Liver Patient Dataset (ILPD), Kaggle
- **Records:** ~30,000 patient records
- **Features:** Age, Gender, and biochemical markers — Total/Direct Bilirubin, Alkaline Phosphotase, SGPT, SGOT, Total Proteins, Albumin, A/G Ratio
- **Target:** `Result` — 1 = liver disease, 0 = no liver disease

## Approach (CRISP-DM)

1. **Data Understanding** – inspected structure, missing values, and duplicates
2. **Data Preparation** – cleaned column names, converted fields to numeric, imputed missing values with the median, dropped duplicates and invalid rows, encoded `Gender`
3. **Exploratory Data Analysis** – distribution plots, boxplots for outlier detection, and a correlation heatmap
4. **Class Imbalance Handling** – applied **SMOTE** to the training set (dataset was ~70/30 imbalanced toward liver-disease cases)
5. **Modeling** – trained and evaluated 8 classifiers, each with hyperparameter tuning (Randomized/Grid Search):
   - Logistic Regression
   - Decision Tree
   - Gaussian Naive Bayes
   - Support Vector Machine (SVM)
   - Random Forest
   - XGBoost
   - Gradient Boosting
   - K-Nearest Neighbors (KNN)
6. **Evaluation** – compared models using accuracy, precision, recall, F1-score, and ROC-AUC, with confusion matrices for each

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| XGBoost | 0.9987 | 0.9992 | 0.9988 | 0.9990 | 0.9998 |
| Gradient Boosting | 0.9956 | 0.9974 | 0.9966 | 0.9969 | 0.9998 |
| Random Forest | 0.9973 | 0.9811 | 0.9981 | 0.9812 | 0.9967 |
| KNN | 0.9622 | 0.9761 | 0.9709 | 0.9735 | 0.9557 |
| Decision Tree | 0.8805 | 0.9100 | 0.8800 | 0.8900 | 0.9350 |
| SVM | 0.7000 | 0.8300 | 0.7000 | 0.7100 | 0.8335 |
| Logistic Regression | 0.6391 | 0.7800 | 0.6400 | 0.6500 | 0.7666 |
| Naive Bayes | 0.5600 | 0.8000 | 0.5600 | 0.5600 | 0.7407 |

**XGBoost** was the top-performing model, offering the best balance of accuracy, robustness, and clinical reliability, with Random Forest close behind. Tree-based ensembles substantially outperformed linear and distance-based baselines on this dataset.

## Tech Stack

- **Python**, **Jupyter Notebook**
- **pandas**, **NumPy** – data handling
- **matplotlib**, **seaborn** – visualization
- **scikit-learn** – modeling, scaling, train/test split, hyperparameter search
- **imbalanced-learn (SMOTE)** – class imbalance correction
- **XGBoost**

## Repository Contents

```
├── liver_disease_model.ipynb   # Full notebook: EDA, preprocessing, modeling, evaluation
├── ML_ReportFinal.docx         # Detailed written report
└── README.md
```

## Getting Started

```bash
git clone <repo-url>
cd liver-disease-prediction
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
jupyter notebook liver_disease_model.ipynb
```

> Note: place the source CSV (`Liver Patient Dataset (LPD)_train.csv`) in the project directory before running the notebook.

## Future Scope

- Incorporate more diverse, recent patient data to improve generalizability
- Validate externally on datasets from different clinical settings
- Add model explainability (e.g., SHAP) for clinical interpretability
- Package the best model behind a simple API or UI for a clinical decision-support prototype

## References

- Kaggle. (2023). Liver Disease Prediction Dataset.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
