# 📉 Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer will leave (churn) the service, using classification algorithms and SMOTE for handling class imbalance.

---

## 📌 Project Overview

| Field | Details |
|-------|---------|
| **Type** | Supervised Learning — Binary Classification |
| **Domain** | Telecom / Customer Retention |
| **Best Model** | Random Forest |
| **Best AUC-ROC** | 0.8026 (80.26%) |
| **Dataset Size** | 2,000 customers |

---

## 🎯 Objective

To predict whether a telecom customer will **cancel their subscription (churn)** based on their behavior and account details — so the business can take action before losing them.

This is similar to what **Jio, Airtel, or any subscription service** uses to retain customers by offering them deals before they leave.

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `tenure` | How many months the customer has been with the company |
| `monthly_charges` | Amount charged per month |
| `total_charges` | Total amount charged so far |
| `contract` | Contract type (Month-to-month, One year, Two year) |
| `internet_service` | Type of internet (DSL, Fiber optic, No) |
| `payment_method` | How they pay (Credit card, Bank transfer, etc.) |
| `num_products` | Number of services subscribed |
| `tech_support` | Has tech support? (Yes/No) |
| `online_security` | Has online security? (Yes/No) |
| `paperless_billing` | Uses paperless billing? (1/0) |
| `senior_citizen` | Is a senior citizen? (1/0) |
| `satisfaction_score` | Customer satisfaction rating (1–5) |
| `num_support_calls` | Number of complaints/support calls made |
| `churn` | **Target variable** — Did they leave? (1=Yes, 0=No) |

---

## ⚙️ Project Pipeline

```
Data Generation → Preprocessing → EDA → SMOTE → Model Training → Evaluation
```

### Step 1 — Data Collection
- Generated realistic dataset of **2,000 telecom customers**
- Churn rate: **54.6%** (1,092 churned / 908 retained)
- Added 2% missing values to simulate real-world messiness

### Step 2 — Preprocessing & Feature Engineering
- Filled missing values using **median and mode imputation**
- Applied **Label Encoding** on categorical columns
- Created 4 new features:
  - `charges_per_month` = total charges ÷ tenure
  - `is_new_customer` = 1 if tenure ≤ 6 months
  - `high_value` = 1 if monthly charges above median
  - `dissatisfied` = 1 if satisfaction score ≤ 2

### Step 3 — Exploratory Data Analysis
Generated 6 visualizations:
- Churn distribution pie chart
- Tenure by churn (histogram)
- Monthly charges vs churn (boxplot)
- Churn rate by contract type
- Churn rate by satisfaction score
- Support calls vs churn rate

### Step 4 — Handling Class Imbalance with SMOTE
```
Before SMOTE: {Churned: 874, Retained: 726}  ← unbalanced
After SMOTE:  {Churned: 874, Retained: 874}  ← balanced ✅
```
SMOTE (Synthetic Minority Oversampling Technique) creates synthetic examples of the minority class so the model doesn't get biased toward always predicting "not churned."

### Step 5 — Model Training
Trained and compared 4 classification models:

| Model | Accuracy | F1 Score | AUC-ROC | Recall |
|-------|----------|----------|---------|--------|
| Logistic Regression | 0.7425 | 0.7576 | 0.7986 | 0.7385 |
| Decision Tree | 0.7025 | 0.7187 | 0.7104 | 0.6972 |
| **Random Forest** | **0.7550** | **0.7793** | **0.8026** ✅ | **0.7936** |
| Gradient Boosting | 0.7400 | 0.7668 | 0.7974 | 0.7844 |

---

## 📈 Output Charts

| File | Description |
|------|-------------|
| `outputs/03_churn_eda.png` | 6-panel Exploratory Data Analysis |
| `outputs/04_churn_results.png` | ROC curves, confusion matrix, feature importance |

---

## 💡 Key Insights

- Customers on **month-to-month contracts** churn the most
- **Low satisfaction score (1–2)** is the strongest churn predictor
- **New customers (tenure < 6 months)** are at highest risk
- More **support calls** = higher chance of churning
- **Two-year contract** customers almost never churn

---

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** — data manipulation
- **numpy** — numerical operations
- **scikit-learn** — ML models, preprocessing, metrics
- **imbalanced-learn** — SMOTE oversampling
- **matplotlib** — plotting
- **seaborn** — statistical visualizations

---

## ▶️ How to Run

**Step 1 — Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

**Step 2 — Run the script:**
```bash
python churn_prediction.py
```

**Step 3 — View outputs:**
Charts are saved in the `outputs/` folder automatically.

---

## 📁 Project Structure

```
churn/
├── churn_prediction.py      ← Main Python script
├── README.md                ← This file
└── outputs/
    ├── 03_churn_eda.png     ← EDA visualizations
    └── 04_churn_results.png ← Model result charts
```

---

## 💡 Key Learnings

- Binary classification for real business problems
- SMOTE for handling imbalanced datasets
- AUC-ROC is better than accuracy for imbalanced classification
- Random Forest outperforms single decision trees
- Feature engineering creates more meaningful signals for the model

