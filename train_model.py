"""
Run once to train the model and save all artifacts to model/
Usage:  python train_model.py
"""
import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import duckdb
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
DATA_URL  = ("https://raw.githubusercontent.com/IBM/employee-attrition-aif360"
             "/master/data/emp_attrition.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
if os.path.exists(DATA_PATH):
    df_raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"Loaded from local file — {df_raw.shape[0]:,} rows")
else:
    df_raw = pd.read_csv(DATA_URL)
    print(f"Loaded from URL — {df_raw.shape[0]:,} rows")

# ── Feature engineering via DuckDB ────────────────────────────────────────────
conn = duckdb.connect()
conn.register("hr_raw", df_raw)

df_features = conn.execute("""
    SELECT
        CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END               AS attrition_flag,
        Age, DailyRate, DistanceFromHome, Education,
        EnvironmentSatisfaction, HourlyRate, JobInvolvement, JobLevel,
        JobSatisfaction, MonthlyIncome, NumCompaniesWorked,
        PercentSalaryHike, PerformanceRating, RelationshipSatisfaction,
        StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear,
        WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
        YearsSinceLastPromotion, YearsWithCurrManager,
        CASE WHEN YearsSinceLastPromotion >= 3 THEN 1 ELSE 0 END     AS stalled_promotion_flag,
        ROUND(
            (JobSatisfaction + EnvironmentSatisfaction
             + RelationshipSatisfaction + WorkLifeBalance) / 4.0, 2
        )                                                             AS satisfaction_composite,
        YearsAtCompany - YearsWithCurrManager                        AS manager_tenure_gap,
        BusinessTravel, Department, EducationField,
        Gender, JobRole, MaritalStatus, OverTime
    FROM hr_raw
""").df()

# ── Encode categoricals ───────────────────────────────────────────────────────
CATEGORICAL_COLS = ["BusinessTravel", "Department", "EducationField",
                    "Gender", "JobRole", "MaritalStatus", "OverTime"]
DERIVED_COLS     = ["stalled_promotion_flag", "satisfaction_composite", "manager_tenure_gap"]

encoders = {}
df_model = df_features.copy()
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le

FEATURE_COLS = [c for c in df_model.columns if c != "attrition_flag"]
X = df_model[FEATURE_COLS]
y = df_model["attrition_flag"]

print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
print(f"Attrition rate: {y.mean()*100:.1f}%")

# ── Train on full dataset (AUC already validated in notebook) ─────────────────
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X, y)

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
print(f"CV AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# ── Defaults: median (numeric) / mode (categorical, as original label) ────────
defaults = {}
for col in FEATURE_COLS:
    if col in CATEGORICAL_COLS:
        defaults[col] = str(df_features[col].mode()[0])
    elif col in DERIVED_COLS:
        defaults[col] = None  # always computed from user inputs
    else:
        defaults[col] = float(df_features[col].median())

# ── Save artifacts ────────────────────────────────────────────────────────────
joblib.dump(rf,           os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(encoders,     os.path.join(MODEL_DIR, "encoders.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_names.pkl"))
joblib.dump(defaults,     os.path.join(MODEL_DIR, "defaults.pkl"))

print("\nSaved to model/:")
print("  rf_model.pkl  encoders.pkl  feature_names.pkl  defaults.pkl")
print("\nDone. Run:  streamlit run app.py")
