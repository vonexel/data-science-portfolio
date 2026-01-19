# Adult Income Classification

Binary classification project on the **UCI Adult / Census Income** dataset: predict whether a person’s annual income is **`<=50K`** or **`>50K`** using **Logistic Regression** and **Support Vector Machine (SVM)**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Workflow](#workflow)
- [EDA Highlights](#eda-highlights)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Results](#results)

---

## Project Overview

This project solves a **binary classification** task: predict a person’s income bracket (**`<=50K`** vs **`>50K`**) based on demographic and employment-related features (age, education, occupation, hours-per-week, etc.).

Models trained and compared:
- **Logistic Regression** (baseline, interpretable linear model)
- **SVM (SVC)** (non-linear decision boundary via kernels, strong general-purpose classifier)

Evaluation:
- **Accuracy** via `.score()`
- **ROC-AUC** for ranking quality of predicted probabilities

---

## Motivation

This is a classic supervised learning pipeline that practices:
- Real-world missing value handling in **categorical** features
- Exploratory analysis and sanity-checking distributions
- Encoding of mixed-type data
- Comparing two widely-used classification approaches under the same preprocessing

---

## Dataset

[Adult dataset](https://www.cs.toronto.edu/~delve/data/adult/desc.html) 
**Rows:** 48,842
**Columns:** 15 (14 features + target)

Target:
- `income` ∈ {`<=50K`, `>50K`}

Features include:
- `age`, `workclass`, `fnlwgt`, `education`, `educational-num`,
- `marital-status`, `occupation`, `relationship`, `race`, `gender`,
- `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`

Missing values:
- Encoded as `'?'` in the raw data (primarily in: `workclass`, `occupation`, `native-country`)

---

## Tech Stack

Core:
- `pandas`, `numpy`

Visualization / EDA:
- `matplotlib`, `seaborn`, `missingno`

Modeling:
- `scikit-learn` (LogisticRegression, SVC, train_test_split, metrics)
- `catboost` (used for **missing value imputation** in categorical columns)
- `optuna` (hyperparameter search for CatBoost imputation models)

---

## Workflow

1. Load data (`adult.csv`)
2. Quick dataset profiling (`shape`, `describe`, `info`)
3. Replace `'?'` → `NA`
4. Visualize missingness patterns (matrix / heatmap / dendrogram)
5. EDA plots (distribution and relationship to target)
6. Handle missing values (baseline deletion + proposed alternative imputation)
7. Encode categorical variables
8. Train/test split (80/20)
9. Train **LogReg** and **SVM**
10. Evaluate (Accuracy + ROC-AUC)
11. Compare models + conclusions

---

## EDA Highlights

<div align="center">
  <img src="./Income%20Level%20Classification/income.png">
</div>

<div align="center">
  <img src="./Income%20Level%20Classification/age.png">
</div>

<div align="center">
  <img src="./Income%20Level%20Classification/hours_per_week.png">
</div>

Key observations from exploratory analysis:

- **Income is imbalanced**: `<=50K` is the majority class.
- **Age** distribution peaks around mid-life; higher-income group tends to be older.
- **Hours-per-week** shows two common modes (part-time vs full-time); higher-income group tends to work more hours.
- Missing values are concentrated in:
  - `workclass` (~5–6%)
  - `occupation` (~5–6%)
  - `native-country` (~2%)
  - total missingness ~8%

---

## Data Preprocessing

### Missing values

<div align="center">
  <img src="./Income%20Level%20Classification/missing_values_plot.png">
</div>

Baseline approach (simple, but can discard signal):
- Drop rows with missing values.

Chosen approach (keeps data, reduces information loss):
- **Impute categorical missing values using CatBoostClassifier** trained to predict each missing column.
- Imputation order (from fewer missing → more missing):
  1. `native-country`
  2. `workclass`
  3. `occupation`

Hyperparameters for each imputation model were tuned with **Optuna**.

<div align="center">
  <img src="./Income%20Level%20Classification/imputation_quality.png">
</div>

Imputation quality (train F1, weighted):
- `native-country`: ~0.95
- `workclass`: ~0.86
- `occupation`: ~0.69

> Note: Imputation is a modeling choice. It can improve downstream performance, but it may also introduce bias/noise if the imputation model is weak (especially for `occupation`).

### Encoding

- All categorical features were encoded with `LabelEncoder`.

---

## Modeling

### Train/Test split
- `train_test_split(test_size=0.2, random_state=42)`

### Logistic Regression
- `LogisticRegression(max_iter=1000)`
- Observed `ConvergenceWarning` (lbfgs reached max iterations)

### SVM
- `SVC(probability=True)`
- Trained on the same feature set and split

---

## Results

| Model                | Accuracy | ROC-AUC |
|---------------------|---------:|--------:|
| Logistic Regression |   ~0.81  |  ~0.82  |
| SVM (SVC)           |   ~0.80  |  ~0.61  |

Interpretation:
- **Accuracy is similar**, but **Logistic Regression is notably better by ROC-AUC** in this setup.
- Given the ROC-AUC gap, **Logistic Regression is preferred** for ranking/separation quality.



