
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, accuracy_score)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

import os
os.makedirs("outputs", exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = ["#06D6A0", "#118AB2", "#EF476F", "#FFD166", "#073B4C"]

# ─────────────────────────────────────────
# STEP 1: GENERATE REALISTIC TELECOM DATASET
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1: DATA COLLECTION & GENERATION")
print("="*60)

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    "tenure"             : np.random.randint(0, 72, n),
    "monthly_charges"    : np.round(np.random.uniform(20, 120, n), 2),
    "total_charges"      : np.round(np.random.uniform(100, 8000, n), 2),
    "contract"           : np.random.choice(["Month-to-month","One year","Two year"], n, p=[0.55,0.25,0.20]),
    "internet_service"   : np.random.choice(["DSL","Fiber optic","No"], n, p=[0.35,0.45,0.20]),
    "payment_method"     : np.random.choice(["Electronic check","Mailed check","Bank transfer","Credit card"], n),
    "num_products"       : np.random.randint(1, 6, n),
    "tech_support"       : np.random.choice(["Yes","No"], n, p=[0.4, 0.6]),
    "online_security"    : np.random.choice(["Yes","No"], n, p=[0.35, 0.65]),
    "paperless_billing"  : np.random.choice([1, 0], n, p=[0.6, 0.4]),
    "senior_citizen"     : np.random.choice([0, 1], n, p=[0.84, 0.16]),
    "partner"            : np.random.choice([1, 0], n, p=[0.5, 0.5]),
    "dependents"         : np.random.choice([1, 0], n, p=[0.3, 0.7]),
    "num_support_calls"  : np.random.poisson(1.5, n),
    "satisfaction_score" : np.random.randint(1, 6, n),   # 1-5
})

# Realistic churn probability
churn_prob = (
    0.3
    + (df["tenure"] < 12) * 0.2
    + (df["contract"] == "Month-to-month") * 0.25
    + (df["monthly_charges"] > 80) * 0.15
    + (df["tech_support"] == "No") * 0.1
    + (df["satisfaction_score"] <= 2) * 0.25
    + (df["num_support_calls"] >= 4) * 0.15
    - (df["tenure"] > 48) * 0.2
    - (df["num_products"] >= 3) * 0.1
    - (df["contract"] == "Two year") * 0.25
).clip(0.02, 0.92)

df["churn"] = (np.random.rand(n) < churn_prob).astype(int)

# Add some noise / missing values
df.loc[df.sample(frac=0.015).index, "total_charges"] = np.nan
df.loc[df.sample(frac=0.01).index, "satisfaction_score"] = np.nan

print(f"  Dataset shape  : {df.shape}")
print(f"  Churn rate     : {df['churn'].mean()*100:.1f}%  ({df['churn'].sum()} churned / {n - df['churn'].sum()} retained)")
print(f"  Missing values : {df.isnull().sum().sum()} cells")

# ─────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2: PREPROCESSING & FEATURE ENGINEERING")
print("="*60)

df["total_charges"].fillna(df["total_charges"].median(), inplace=True)
df["satisfaction_score"].fillna(df["satisfaction_score"].mode()[0], inplace=True)

# Encode
le = LabelEncoder()
for col in ["contract","internet_service","payment_method","tech_support","online_security"]:
    df[col+"_enc"] = le.fit_transform(df[col])

# Feature engineering
df["charges_per_month"] = (df["total_charges"] / df["tenure"].replace(0, 1)).round(2)
df["is_new_customer"]   = (df["tenure"] <= 6).astype(int)
df["high_value"]        = (df["monthly_charges"] > df["monthly_charges"].median()).astype(int)
df["dissatisfied"]      = (df["satisfaction_score"] <= 2).astype(int)

print("  ✓ Missing values imputed")
print("  ✓ Categorical encoding done")
print("  ✓ New features: charges_per_month, is_new_customer, high_value, dissatisfied")

# ─────────────────────────────────────────
# STEP 3: EDA
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3: EXPLORATORY DATA ANALYSIS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Customer Churn Prediction — EDA", fontsize=17, fontweight="bold")

# Churn distribution
ax = axes[0, 0]
counts = df["churn"].value_counts()
wedges, texts, autotexts = ax.pie(counts, labels=["Retained","Churned"], autopct="%1.1f%%",
       colors=[PALETTE[0], PALETTE[2]], startangle=90,
       explode=[0, 0.05], shadow=True)
ax.set_title("Churn Distribution", fontweight="bold")

# Tenure by churn
ax = axes[0, 1]
df[df["churn"]==0]["tenure"].hist(ax=ax, bins=30, alpha=0.6, color=PALETTE[0], label="Retained")
df[df["churn"]==1]["tenure"].hist(ax=ax, bins=30, alpha=0.6, color=PALETTE[2], label="Churned")
ax.set_title("Tenure by Churn", fontweight="bold")
ax.set_xlabel("Tenure (months)")
ax.legend()

# Monthly charges by churn
ax = axes[0, 2]
df.boxplot(column="monthly_charges", by="churn", ax=ax,
           boxprops=dict(color=PALETTE[1]), patch_artist=True)
ax.set_title("Monthly Charges vs Churn", fontweight="bold")
ax.set_xlabel("Churn (0=No, 1=Yes)")
ax.set_ylabel("Monthly Charges ($)")
plt.sca(ax); plt.title("Monthly Charges vs Churn", fontweight="bold")

# Contract type churn rate
ax = axes[1, 0]
ct_churn = df.groupby("contract")["churn"].mean().sort_values(ascending=False)
ct_churn.plot(kind="bar", ax=ax, color=PALETTE[3], edgecolor="white", rot=20)
ax.set_title("Churn Rate by Contract Type", fontweight="bold")
ax.set_ylabel("Churn Rate")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# Satisfaction score vs churn
ax = axes[1, 1]
sat_churn = df.groupby("satisfaction_score")["churn"].mean()
bars = ax.bar(sat_churn.index, sat_churn.values, color=[PALETTE[2] if v > 0.4 else PALETTE[0] for v in sat_churn.values])
ax.set_title("Churn Rate by Satisfaction Score", fontweight="bold")
ax.set_xlabel("Satisfaction Score")
ax.set_ylabel("Churn Rate")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# Support calls vs churn
ax = axes[1, 2]
call_churn = df.groupby("num_support_calls")["churn"].mean().head(8)
ax.plot(call_churn.index, call_churn.values, marker="o", color=PALETTE[4], linewidth=2.5)
ax.fill_between(call_churn.index, call_churn.values, alpha=0.2, color=PALETTE[4])
ax.set_title("Support Calls vs Churn Rate", fontweight="bold")
ax.set_xlabel("Number of Support Calls")
ax.set_ylabel("Churn Rate")

plt.tight_layout()
plt.savefig("outputs/03_churn_eda.png", dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
plt.close()
print("  ✓ EDA saved → outputs/03_churn_eda.png")

# ─────────────────────────────────────────
# STEP 4: CLASS IMBALANCE → SMOTE
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4: HANDLING CLASS IMBALANCE WITH SMOTE")
print("="*60)

feature_cols = ["tenure","monthly_charges","total_charges","num_products",
                "tech_support_enc","online_security_enc","paperless_billing",
                "senior_citizen","partner","dependents","num_support_calls",
                "satisfaction_score","contract_enc","internet_service_enc",
                "payment_method_enc","charges_per_month","is_new_customer",
                "high_value","dissatisfied"]

X = df[feature_cols].copy()
y = df["churn"]

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      stratify=y, random_state=42)

print(f"  Before SMOTE: {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE : {pd.Series(y_train_sm).value_counts().to_dict()}")

scaler = StandardScaler()
X_train_s  = scaler.fit_transform(X_train_sm)
X_test_s   = scaler.transform(X_test)

# ─────────────────────────────────────────
# STEP 5: MODEL TRAINING
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5: MODEL TRAINING & EVALUATION")
print("="*60)

models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
}

results = {}
print(f"\n  {'Model':<22} {'Accuracy':>9} {'F1 Score':>9} {'AUC-ROC':>9} {'Recall':>9}")
print("  " + "-"*62)

for name, model in models.items():
    X_tr = X_train_s
    model.fit(X_tr, y_train_sm)
    preds = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:, 1]

    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    auc  = roc_auc_score(y_test, proba)
    rec  = classification_report(y_test, preds, output_dict=True)["1"]["recall"]

    results[name] = {"acc":acc, "f1":f1, "auc":auc, "recall":rec,
                     "preds":preds, "proba":proba, "model":model}
    print(f"  {name:<22} {acc:>9.4f} {f1:>9.4f} {auc:>9.4f} {rec:>9.4f}")

best = max(results, key=lambda k: results[k]["auc"])
print(f"\n  ✓ Best model: {best}  (AUC = {results[best]['auc']:.4f})")

# ─────────────────────────────────────────
# STEP 6: VISUALIZE RESULTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Customer Churn Prediction — Model Results", fontsize=16, fontweight="bold")

# ROC curves
ax = axes[0, 0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
ax.plot([0,1],[0,1],"k--", linewidth=1)
ax.set_title("ROC Curves — All Models", fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9)

# Confusion matrix (best model)
ax = axes[0, 1]
cm = confusion_matrix(y_test, results[best]["preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Retained","Churned"], yticklabels=["Retained","Churned"])
ax.set_title(f"Confusion Matrix ({best})", fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

# Metric comparison
ax = axes[1, 0]
metrics_df = pd.DataFrame({n: [r["acc"], r["f1"], r["auc"], r["recall"]]
                           for n, r in results.items()},
                          index=["Accuracy","F1","AUC","Recall"])
metrics_df.T.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
ax.set_title("Model Metric Comparison", fontweight="bold")
ax.set_ylabel("Score")
ax.legend(loc="lower right", fontsize=9)
ax.tick_params(axis="x", rotation=20)
ax.set_ylim(0, 1)

# Feature importance (RF)
ax = axes[1, 1]
rf = results["Random Forest"]["model"]
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(10)
fi.plot(kind="barh", ax=ax, color=PALETTE[1])
ax.set_title("Top 10 Churn Predictors (RF)", fontweight="bold")
ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig("outputs/04_churn_results.png", dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
plt.close()

