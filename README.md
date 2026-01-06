# Feature-selection-secom
feature selection and machine learning for semiconductor yeild prediction using sensor data

# Semiconductor Yield Prediction

This project focuses on predicting semiconductor wafer yield using
sensor data collected from manufacturing processes. End‑to‑end machine learning project to classify **Pass (0)** vs **Fail (1)** units in a semiconductor manufacturing line using high‑dimensional sensor data, and to identify which signals are actually necessary for good prediction performance.


## Dataset
- 1567 samples
- 592 sensor features
- Target: Pass / Fail

---

## 1. Problem & Objective

- Each production entity (wafer / lot) is measured by hundreds of sensors during in‑house testing.  
- Not all measurements are useful: some are noisy, redundant, or rarely recorded.  
- **Goal:**  
  - Build a supervised classifier that predicts Pass/Fail from sensor readings.  
  - Use feature importance to find a **much smaller set of critical signals** that keeps model performance almost unchanged.

---

## 2. Data Overview

- **Samples:** 1,567 entities  
- **Features:** 591 sensor variables (after basic raw cleaning)  
- **Target:**  
  - `0` → Pass  
  - `1` → Fail  
- **Key characteristics:**  
  - Wide, high‑dimensional table.  
  - Contains missing values and outliers.  
  - Strong class imbalance (Pass ≫ Fail).  

---

## Techniques Used
- Data preprocessing
- Missing value imputation
- Feature selection (Variance Threshold, ANOVA)
- Machine Learning models

## Tools
- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

---

## 4. Workflow Summary

### 4.1 Exploratory Data Analysis

- Target distribution (Pass vs Fail) to expose class imbalance.  
- Histograms & boxplots for selected sensors to inspect spread, skewness, and outliers.  
- Correlation plots for top features to spot redundant or strongly related sensors.

### 4.2 Data Cleaning

- Convert all sensor columns to numeric and mark invalid values as `NaN`.  
- Drop features with very high missing rate and those with near‑zero variance.  
- Median imputation for remaining missing values.  
- Map original labels `-1` → `0`, `1` → `1` for easier modeling.

### 4.3 Pre‑processing

- Split into **features `X`** and **target `y`**.  
- Stratified train–test split to keep Pass/Fail ratio consistent.  
- Balance only the **training** set using **SMOTE** (synthetic Fail examples).  
- Standardise features (zero mean, unit variance) for algorithms that depend on scale (SVM, KNN, Logistic).

---

## 5. Models Trained

At least three different families were tried on the same processed data:

- **Random Forest Classifier**  
  - Good at handling many noisy features.  
  - Provides feature importances for interpretability.

- **Support Vector Machine (SVM)**  
  - Trained on scaled data with linear and RBF kernels.  
  - Checks if a margin‑based model can separate Pass and Fail.

- **Gaussian Naive Bayes**  
  - Very fast baseline with strong assumptions about feature independence.  

Optionally, additional models like GradientBoosting or KNN were added for comparison.

### Training, Tuning & Evaluation

- Cross‑validation on the training set to compare models fairly.  
- Small grid or random search over key hyperparameters (e.g., number of trees, tree depth, `C` and kernel for SVM).  
- Metrics used on validation and test sets:
  - Accuracy  
  - ROC‑AUC  
  - Precision, Recall, F1 (Fail class is emphasised)  
  - Confusion matrix  
  - Precision–Recall curve

---

## 6. Model Comparison & Final Choice

- A summary table was built with CV scores and test metrics for each model.  
- The **Random Forest** consistently gave the best balance of:
  - High Fail‑class F1 and ROC‑AUC.  
  - Stable performance across folds.  
  - Clear feature importance scores.

**Final selected model:** RandomForestClassifier (tuned).  

Random Forest was chosen because it captured non‑linear relationships between sensors and yield, handled the imbalanced, noisy data well, and directly highlighted which sensors were most influential.

---

## 7. Feature Importance & Dimensionality Reduction

- Feature importances extracted from the final Random Forest.  
- Sensors ranked and the top‑k (e.g., 20 or 50) used to retrain a compact model.  
- Performance with the reduced feature set stayed very close to the full model, showing that many original signals are redundant for prediction.

---

## 8. Saving & Using the Model

The final pipeline is saved so it can be reused on new lots:

```python
import joblib

joblib.dump(final_model, "rf_yield_model.joblib")
joblib.dump(imputer, "imputer.joblib")
joblib.dump(scaler, "scaler.joblib")
