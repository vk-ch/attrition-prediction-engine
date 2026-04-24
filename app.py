import os, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

CATEGORICAL_COLS = ["BusinessTravel", "Department", "EducationField",
                    "Gender", "JobRole", "MaritalStatus", "OverTime"]
DERIVED_COLS     = ["stalled_promotion_flag", "satisfaction_composite", "manager_tenure_gap"]

FEATURE_DISPLAY = {
    "Age": "Age", "DailyRate": "Daily Rate",
    "DistanceFromHome": "Distance from Home", "Education": "Education Level",
    "EnvironmentSatisfaction": "Environment Satisfaction", "HourlyRate": "Hourly Rate",
    "JobInvolvement": "Job Involvement", "JobLevel": "Job Level",
    "JobSatisfaction": "Job Satisfaction", "MonthlyIncome": "Monthly Income ($)",
    "NumCompaniesWorked": "# Prior Companies", "PercentSalaryHike": "Salary Hike %",
    "PerformanceRating": "Performance Rating",
    "RelationshipSatisfaction": "Relationship Satisfaction",
    "StockOptionLevel": "Stock Option Level", "TotalWorkingYears": "Total Working Years",
    "TrainingTimesLastYear": "Trainings Last Year", "WorkLifeBalance": "Work-Life Balance",
    "YearsAtCompany": "Years at Company", "YearsInCurrentRole": "Years in Current Role",
    "YearsSinceLastPromotion": "Years Since Promotion",
    "YearsWithCurrManager": "Years w/ Manager",
    "stalled_promotion_flag": "Stalled Promotion (3+ yrs)",
    "satisfaction_composite": "Satisfaction Score (avg)",
    "manager_tenure_gap": "Manager Tenure Gap",
    "BusinessTravel": "Business Travel", "Department": "Department",
    "EducationField": "Education Field", "Gender": "Gender",
    "JobRole": "Job Role", "MaritalStatus": "Marital Status",
    "OverTime": "Works Overtime",
}

# ── Model loading — trains from scratch if pkl files aren't present ───────────
@st.cache_resource(show_spinner="Setting up model — takes ~30s on first run...")
def load_model():
    model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    if os.path.exists(model_path):
        rf            = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
        encoders      = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
        feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        defaults      = joblib.load(os.path.join(MODEL_DIR, "defaults.pkl"))
    else:
        rf, encoders, feature_names, defaults = _train_and_save()

    explainer = shap.TreeExplainer(rf)
    return rf, encoders, feature_names, defaults, explainer


def _train_and_save():
    import duckdb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    os.makedirs(MODEL_DIR, exist_ok=True)

    data_path = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    DATA_URL  = ("https://raw.githubusercontent.com/IBM/employee-attrition-aif360"
                 "/master/data/emp_attrition.csv")
    df_raw = (pd.read_csv(data_path, encoding="utf-8-sig")
              if os.path.exists(data_path) else pd.read_csv(DATA_URL))

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
            ROUND((JobSatisfaction + EnvironmentSatisfaction
                   + RelationshipSatisfaction + WorkLifeBalance) / 4.0, 2) AS satisfaction_composite,
            YearsAtCompany - YearsWithCurrManager                        AS manager_tenure_gap,
            BusinessTravel, Department, EducationField,
            Gender, JobRole, MaritalStatus, OverTime
        FROM hr_raw
    """).df()

    encoders = {}
    df_model = df_features.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    FEATURE_COLS = [c for c in df_model.columns if c != "attrition_flag"]
    X = df_model[FEATURE_COLS]
    y = df_model["attrition_flag"]

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)

    defaults = {}
    for col in FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            defaults[col] = str(df_features[col].mode()[0])
        elif col in DERIVED_COLS:
            defaults[col] = None
        else:
            defaults[col] = float(df_features[col].median())

    joblib.dump(rf,           os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(encoders,     os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(defaults,     os.path.join(MODEL_DIR, "defaults.pkl"))
    return rf, encoders, FEATURE_COLS, defaults


def get_shap_vals(explainer, X_input):
    raw = explainer.shap_values(X_input)
    ev  = explainer.expected_value
    if isinstance(raw, list):
        return raw[1][0], float(ev[1] if hasattr(ev, "__len__") else ev)
    if hasattr(raw, "ndim") and raw.ndim == 3:
        return raw[0, :, 1], float(ev[1] if hasattr(ev, "__len__") else ev)
    base = float(np.asarray(ev).flat[-1])
    return np.asarray(raw).reshape(X_input.shape[1]), base


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .pred-card   { border-radius:12px; padding:22px 28px; text-align:center; margin-bottom:14px; }
  .high-risk   { background:#fde8e8; border:2px solid #E74C3C; }
  .medium-risk { background:#fef3e2; border:2px solid #E67E22; }
  .low-risk    { background:#e8f8ef; border:2px solid #27AE60; }
  .risk-label  { font-size:26px; font-weight:800; margin-bottom:2px; }
  .risk-pct    { font-size:42px; font-weight:900; margin:6px 0 2px; }
  .risk-sub    { font-size:13px; color:#666; }
  .driver-row  { background:#f8f9fa; border-left:3px solid #2980B9;
                 padding:6px 12px; margin:4px 0; border-radius:4px; font-size:13.5px; }
  .stat-box    { background:#f0f4f8; border-radius:8px; padding:10px 14px;
                 text-align:center; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
rf, encoders, FEATURE_COLS, defaults, explainer = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔮 Employee Attrition Predictor")
st.markdown(
    "Fill in the employee profile on the left — the model predicts their attrition risk "
    "and explains exactly which factors are driving it.  \n"
    "*IBM HR Analytics · 1,470 employees · 35 features · Random Forest · AUC 0.87*"
)
st.divider()

# ── Two-column layout ─────────────────────────────────────────────────────────
col_in, col_out = st.columns([4, 5], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# INPUT PANEL
# ══════════════════════════════════════════════════════════════════════════════
with col_in:
    st.markdown("### 👤 Employee Profile")

    with st.expander("💼  Work Factors", expanded=True):
        overtime        = st.selectbox("Overtime",         ["No", "Yes"],             index=0)
        job_level       = st.slider("Job Level",           1, 5, 2,
                                    help="1 = Entry · 3 = Mid · 5 = C-Suite")
        department      = st.selectbox("Department",       ["Research & Development",
                                                             "Sales", "Human Resources"])
        job_role        = st.selectbox("Job Role",         [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources",
        ])
        business_travel = st.selectbox("Business Travel",  ["Travel_Rarely",
                                                             "Non-Travel", "Travel_Frequently"])

    with st.expander("😊  Satisfaction  (1 = Low  →  4 = Very High)", expanded=True):
        job_sat = st.slider("Job Satisfaction",          1, 4, 3)
        env_sat = st.slider("Environment Satisfaction",  1, 4, 3)
        wlb     = st.slider("Work-Life Balance",         1, 4, 3)
        rel_sat = st.slider("Relationship Satisfaction", 1, 4, 3)

    with st.expander("📈  Career & Compensation", expanded=True):
        age           = st.slider("Age",                          18, 60, 32)
        monthly_inc   = st.number_input("Monthly Income ($)",   1009, 19999, 5000, step=500)
        total_yrs     = st.slider("Total Working Years",          0,  40,  8)
        yrs_company   = st.slider("Years at Company",             0,  40,  3)
        yrs_role      = st.slider("Years in Current Role",        0,  18,  2)
        yrs_promo     = st.slider("Years Since Last Promotion",   0,  15,  1)
        yrs_manager   = st.slider("Years with Current Manager",   0,  17,  2)
        num_companies = st.slider("# Prior Companies Worked",     0,   9,  1)

    with st.expander("🧑  Personal Details", expanded=False):
        marital    = st.selectbox("Marital Status",    ["Single", "Married", "Divorced"])
        gender     = st.selectbox("Gender",            ["Male", "Female"])
        distance   = st.slider("Distance from Home (miles)", 1, 29, 7)
        education  = st.slider("Education Level",     1, 5, 3,
                                help="1=Below College · 3=Bachelor · 5=Doctor")
        edu_field  = st.selectbox("Education Field",  ["Life Sciences", "Medical",
                                                        "Marketing", "Technical Degree",
                                                        "Human Resources", "Other"])
        stock_opt  = st.slider("Stock Option Level",  0, 3, 1)
        job_inv    = st.slider("Job Involvement",     1, 4, 3,
                                help="1=Low · 4=Very High")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD FEATURE ROW
# ══════════════════════════════════════════════════════════════════════════════
row = {}
for col in FEATURE_COLS:
    v = defaults.get(col)
    if v is not None:
        row[col] = v

# Encode categoricals
cat_map = {
    "OverTime": overtime, "Department": department, "JobRole": job_role,
    "BusinessTravel": business_travel, "MaritalStatus": marital,
    "Gender": gender, "EducationField": edu_field,
}
for col, val in cat_map.items():
    row[col] = int(encoders[col].transform([val])[0])

# Numeric overrides
row.update({
    "Age": age, "MonthlyIncome": monthly_inc, "TotalWorkingYears": total_yrs,
    "YearsAtCompany": yrs_company, "YearsInCurrentRole": yrs_role,
    "YearsSinceLastPromotion": yrs_promo, "YearsWithCurrManager": yrs_manager,
    "NumCompaniesWorked": num_companies, "DistanceFromHome": distance,
    "JobSatisfaction": job_sat, "EnvironmentSatisfaction": env_sat,
    "WorkLifeBalance": wlb, "RelationshipSatisfaction": rel_sat,
    "JobLevel": job_level, "Education": education,
    "StockOptionLevel": stock_opt, "JobInvolvement": job_inv,
})

# Derived features
row["stalled_promotion_flag"] = 1 if yrs_promo >= 3 else 0
row["satisfaction_composite"] = round((job_sat + env_sat + rel_sat + wlb) / 4.0, 2)
row["manager_tenure_gap"]     = yrs_company - yrs_manager

X_input = pd.DataFrame([row])[FEATURE_COLS]

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT + EXPLAIN
# ══════════════════════════════════════════════════════════════════════════════
proba    = float(rf.predict_proba(X_input)[0][1])
pct      = round(proba * 100, 1)
verdict  = "likely to leave" if proba >= 0.50 else "likely to stay"

if proba >= 0.55:
    tier, css, emoji, color = "HIGH RISK",   "high-risk",   "🔴", "#E74C3C"
elif proba >= 0.30:
    tier, css, emoji, color = "MEDIUM RISK", "medium-risk", "🟡", "#E67E22"
else:
    tier, css, emoji, color = "LOW RISK",    "low-risk",    "🟢", "#27AE60"

shap_vals, base_val = get_shap_vals(explainer, X_input)

display_names = [FEATURE_DISPLAY.get(f, f) for f in FEATURE_COLS]

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT PANEL
# ══════════════════════════════════════════════════════════════════════════════
with col_out:
    st.markdown("### 🎯 Risk Assessment")

    # ── Prediction card ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="pred-card {css}">
      <div class="risk-label" style="color:{color};">{emoji}&nbsp; {tier}</div>
      <div class="risk-pct"   style="color:{color};">{pct}%</div>
      <div class="risk-sub">probability of leaving  ·  this employee is <strong>{verdict}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(proba, 1.0))
    st.caption("Threshold: 0 – 30% = Low  ·  30 – 55% = Medium  ·  55 – 100% = High")

    # ── Quick stats row ───────────────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    s1.markdown(f'<div class="stat-box"><b>{row["satisfaction_composite"]}</b><br>Satisfaction avg (1–4)</div>',
                unsafe_allow_html=True)
    s2.markdown(f'<div class="stat-box"><b>{"Yes" if overtime == "Yes" else "No"}</b><br>Working overtime</div>',
                unsafe_allow_html=True)
    s3.markdown(f'<div class="stat-box"><b>{yrs_promo} yr{"s" if yrs_promo != 1 else ""}</b><br>Since last promotion</div>',
                unsafe_allow_html=True)

    st.divider()

    # ── SHAP waterfall chart ──────────────────────────────────────────────────
    st.markdown("### 📊 What's Driving This Prediction?")
    st.caption(
        "🔴 Red = pushes risk **up**  ·  🔵 Blue = pushes risk **down**  ·  "
        "Starting point (E[f(x)]) = average attrition rate across all employees"
    )

    explanation = shap.Explanation(
        values        = shap_vals,
        base_values   = base_val,
        data          = X_input.iloc[0].values,
        feature_names = display_names,
    )

    shap.plots.waterfall(explanation, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close("all")

    st.divider()

    # ── Top drivers in plain English ──────────────────────────────────────────
    st.markdown("### 💡 Top Factors")
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:6]
    for i in top_idx:
        sv    = shap_vals[i]
        fname = display_names[i]
        fval  = X_input.iloc[0, i]
        if sv > 0:
            direction = f'<span style="color:#E74C3C; font-weight:700;">↑ Increases risk</span>'
        else:
            direction = f'<span style="color:#27AE60; font-weight:700;">↓ Reduces risk</span>'
        st.markdown(
            f'<div class="driver-row">{direction} &nbsp;·&nbsp; '
            f'<strong>{fname}</strong> = <code>{fval}</code></div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built by **Venkat Kowshik** · IBM HR Analytics Dataset (1,470 employees, 35 features) · "
    "Random Forest AUC 0.87 · SHAP TreeExplainer · "
    "[GitHub](https://github.com/vk-ch/attrition-prediction-engine)  \n"
    "Predictions are for demonstration purposes. Not a substitute for professional HR judgment."
)
