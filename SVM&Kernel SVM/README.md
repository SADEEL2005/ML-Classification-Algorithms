# Predictive Maintenance using Support Vector Machines (SVM)

This project implements a predictive maintenance system using the **AI4I 2020 Predictive Maintenance Dataset**. It utilizes Support Vector Machine (SVM) classifiers (both Linear and RBF kernels) to predict machine failures while addressing real-world challenges like data leakage and class imbalance.

## Project Overview

In predictive maintenance, predicting a machine failure before it happens is crucial. However, using columns that specify the *type* of failure as features during training leads to **Data Leakage** (since these details are unknown before a failure occurs). 

This pipeline resolves this issue by dropping post-failure indicators and focuses on predicting machine failures purely based on real-time operational measurements.

## Dataset

The dataset used is `ai4i2020.csv`, which consists of 10,000 data points and 14 features. 
To build a realistic model, we drop:
* **Identifiers:** `UDI`, `Product ID`
* **Target Variable:** `Machine failure` (used as label `y`)
* **Failure Modes (Data Leakage):** `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (Type of failures)

### Remaining Features used for Training:
1. **Type:** Type of product (Low, Medium, High quality represented as L, M, H) - *One-Hot Encoded*
2. **Air temperature [K]**
3. **Process temperature [K]**
4. **Rotational speed [rpm]**
5. **Torque [Nm]**
6. **Tool wear [min]**

---

## Key Features of this Implementation

1. **Anti-Data Leakage Design:** Excludes failure mode columns from the feature matrix to ensure the model learns from pre-failure conditions only.
2. **Class Imbalance Handling:** 
   * Since only ~3.39% of the dataset represents machine failures, we use `stratify=y` during the train-test split.
   * We apply `class_weight='balanced'` in the SVM classifiers to penalize misclassifications of the minority class (failures) more heavily.
3. **Feature Scaling:** Standardizes features using `StandardScaler` to optimize SVM performance.
4. **Comprehensive Evaluation:** Uses `classification_report` (Precision, Recall, F1-Score) alongside Confusion Matrices instead of relying solely on accuracy, which can be misleading for imbalanced data.

---

## Installation & Requirements

Ensure you have Python installed, then install the required libraries:

```bash
pip install pandas numpy scikit-learn