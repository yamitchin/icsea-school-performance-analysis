# ICSEA School Performance Analysis — NSW 2024

## Overview
Predicting the socio-educational advantage level (ICSEA) of 2,187 NSW public schools using machine learning classification models. The goal: identify which schools fall in Low, Medium, or High ICSEA brackets — and uncover the root causes of educational inequity across New South Wales.

**Best model:** Linear SVM with GridSearchCV tuning — **AUC 85.6% / Accuracy 85.6%**

---

## Key Findings
- Indigenous student percentage shows the strongest correlation with ICSEA: **r = -0.71**
- Sydney schools cluster in Medium-High ICSEA (800–1200); regional schools show a second peak below 800
- Lasso feature selection reduced 1,723 features to 48 while maintaining 83.7% accuracy
- PCA reduced accuracy to 45.8%, confirming that feature reduction via Lasso outperforms PCA for this dataset

---

## Model Comparison
| Model | Accuracy |
|---|---|
| Linear SVM — all variables | 84.5% |
| SVM + GridSearchCV tuning | **85.6%** |
| Lasso + SVM | 83.7% |
| KNN (k=8) | 82.8% |
| Neural Network (MLP) | 85.5% |
| PCA + SVM | 45.8% |

---

## Tech Stack
- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, TensorFlow/Keras
- **Models:** Linear SVM, RBF SVM, KNN, Lasso Regression, Decision Tree, Neural Network, MLP, PCA
- **Techniques:** EDA, missing value imputation, OneHot encoding, MinMaxScaler, 5-fold cross-validation, GridSearchCV
- **Visualisation:** Tableau Public

---

## Dataset
- **Source:** NSW Department of Education — Master Public School Dataset
- **Size:** 2,211 schools × 45 variables → cleaned to 2,187 observations × 26 variables
- **Target:** ICSEA class — Low (≤799) · Medium (800–1000) · High (≥1001)

---

## How to Run
