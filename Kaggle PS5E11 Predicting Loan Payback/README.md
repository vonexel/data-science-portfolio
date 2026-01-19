# PS5E11 — Predicting Loan Payback  
**EDA 📊 | Feature Engineering ⚙️ | CatBoost 🚀 | Optuna 🔬 | Stacking 🧱🧠**

This repository contains an end-to-end solution for the Kaggle Playground Series **Season 5, Episode 11**: *Predicting Loan Payback* — a **binary classification** problem focused on estimating the probability that a borrower will fully repay a loan.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Metric](#metric)
- [Data](#data)
- [Tooling](#tooling)
- [Workflow](#workflow)
  - [EDA Highlights](#eda-highlights)
  - [Feature Engineering](#feature-engineering)
  - [Modeling](#modeling)
  - [Explainability](#explainability)
- [Results](#results)

---

## Project Overview

Credit risk is a canonical ML application: the objective is to **rank** borrowers from safest to riskiest using socioeconomic + financial signals (income, DTI, credit score, loan characteristics, and institutional grades). This project emphasizes:

- **Business-consistent EDA** (risk factors, non-linear thresholds, segment effects)
- **Feature engineering** designed for credit underwriting logic
- **GPU-accelerated gradient boosting** (CatBoost / XGBoost / LightGBM)
- **Bayesian hyperparameter optimization** via Optuna
- **Stacking** with a logistic meta-model for robust rank performance

---

## Metric

The competition is evaluated using **ROC AUC** (Area Under the ROC Curve).
AUC is threshold-agnostic and measures how well predicted probabilities **rank** positives above negatives. In practice, it’s preferable to accuracy for imbalanced problems and score-based risk ranking.

---

## Data

The [dataset](https://www.kaggle.com/competitions/playground-series-s5e11/data) consists of `train.csv` and `test.csv`. Core columns include:

- `id` — unique identifier 
- `annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate` — numeric risk signals 
- `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade` — categorical segmentation 
- `loan_paid_back` — **target** (train only): `1` = paid in full, `0` = default 

---

## Tooling

**Core stack**
- pandas, numpy
- scikit-learn (metrics, preprocessing, CV, stacking meta-model)
- CatBoost (GPU)
- Optuna (Bayesian HPO)
- XGBoost (GPU), LightGBM (GPU)
- statsmodels / scipy (stat tests, residual analysis)
- shap (model explainability)

**Why CatBoost**
- Strong out-of-the-box handling of categorical features (ordered boosting / target statistics)
- Stable performance on tabular + mixed-type data

---

## Workflow

### EDA Highlights

Key EDA Insights:

- **Class imbalance** (~4:1 paid vs default) → prefer AUC, stratified CV, and cost-aware decisions.
- **Two orthogonal risk dimensions**:
  - *Financial capacity*: DTI and its non-linear thresholds (e.g., sharp risk increase after DTI ≈ 0.15)
  - *Creditworthiness*: credit score + ordinal grade/subgrade hierarchy
- **Correlation structure**:
  - credit_score ↘ interest_rate (risk-based pricing)
  - DTI ↘ repayment probability
- **Segment effects**:
  - employment_status is often one of the strongest categorical drivers (e.g., Unemployed/Student vs Retired)
  - loan_purpose differentiates productive vs consumptive borrowing

### Feature Engineering

The FE pipeline is engineered to convert “raw attributes” into underwriting-like ratios and non-linear signals:

**1) Skewness handling**
- Yeo–Johnson transform for heavy right tails (income / loan amount / DTI where appropriate)
- Winsorization for ultra-extreme outliers (stabilizes tree splits)

**2) Ratio & burden features**
- `loan_to_income_ratio`
- `payment_to_income`
- `interest_burden`
- `interest_credit_ratio`
- `income_to_dti` (inverse scale, monotonic “safer as larger”)

**3) Non-linear risk boundaries**
- binary flags: `high_dti`, `very_high_dti`
- binned risk tiers: `dti_binned` (low → high risk)

**4) Ordinal structure for credit grades**
- map `grade_subgrade` (A1…F5) → ordinal 1…30
- optionally add polynomial terms to capture “risk jumps” at boundaries

**5) Interaction features**
- polynomial interactions among {DTI, credit_score, interest_rate}
- explicit squared terms (`dti_sq`, `credit_sq`) to help models discover curvature faster

**6) Encoding & scaling**
- keep selected features as `category` for CatBoost
- scale numeric features for stacking / LR meta-model consistency

### Modeling

**Validation**
- Stratified K-Fold CV (e.g., 7–9 folds) to produce OOF predictions and reduce single-split variance.

**Base models**
- CatBoost (primary)
- XGBoost (GPU)
- LightGBM (GPU)

**Hyperparameter optimization**
- Optuna objective optimizes **AUC** on validation split, with early stopping.

**Ensembling**
- Stack OOF predictions from base learners
- Train logistic regression as a meta-model on stacked probabilities

### Explainability

- **SHAP** summary plots for CatBoost identify top drivers and validate business plausibility:
  - DTI-related features, credit score, grade ordinal, employment indicators, and burden ratios typically dominate.

---

## Results

- The pipeline is optimized for **ranking performance (AUC)**, not raw accuracy.
- Public Kaggle baselines for this competition show that GPU-boosting approaches can reach **~0.92 AUC on leaderboard**.



