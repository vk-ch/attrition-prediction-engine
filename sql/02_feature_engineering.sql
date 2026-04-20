-- ============================================================
-- 02_feature_engineering.sql
-- Creates the enriched hr_features table used for ML modeling
-- Run via DuckDB in the notebook
-- ============================================================

CREATE OR REPLACE TABLE hr_features AS
SELECT
    -- ── Identity (kept for reference, not used in model) ──
    EmployeeNumber,

    -- ── Target ───────────────────────────────────────────
    Attrition,
    CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END           AS attrition_flag,

    -- ── Raw numeric features ─────────────────────────────
    Age,
    DailyRate,
    DistanceFromHome,
    Education,
    EnvironmentSatisfaction,
    HourlyRate,
    JobInvolvement,
    JobLevel,
    JobSatisfaction,
    MonthlyIncome,
    MonthlyRate,
    NumCompaniesWorked,
    PercentSalaryHike,
    PerformanceRating,
    RelationshipSatisfaction,
    StockOptionLevel,
    TotalWorkingYears,
    TrainingTimesLastYear,
    WorkLifeBalance,
    YearsAtCompany,
    YearsInCurrentRole,
    YearsSinceLastPromotion,
    YearsWithCurrManager,

    -- ── Engineered: Tenure buckets ───────────────────────
    CASE
        WHEN YearsAtCompany = 0              THEN '< 1 year'
        WHEN YearsAtCompany BETWEEN 1 AND 2  THEN '1-2 years'
        WHEN YearsAtCompany BETWEEN 3 AND 5  THEN '3-5 years'
        WHEN YearsAtCompany BETWEEN 6 AND 10 THEN '6-10 years'
        ELSE '10+ years'
    END                                                      AS tenure_bucket,

    -- ── Engineered: Income band ───────────────────────────
    CASE
        WHEN MonthlyIncome < 3000            THEN 'Low (< $3K)'
        WHEN MonthlyIncome BETWEEN 3000 AND 7000 THEN 'Mid ($3K-$7K)'
        ELSE 'High (> $7K)'
    END                                                      AS income_band,

    -- ── Engineered: Stalled career flag ──────────────────
    -- Employees who haven't been promoted in 3+ years
    CASE WHEN YearsSinceLastPromotion >= 3   THEN 1 ELSE 0  END AS stalled_promotion_flag,

    -- ── Engineered: Composite satisfaction score ─────────
    -- Average of 4 satisfaction dimensions (all on 1-4 scale)
    ROUND(
        (JobSatisfaction + EnvironmentSatisfaction
         + RelationshipSatisfaction + WorkLifeBalance) / 4.0,
    2)                                                       AS satisfaction_composite,

    -- ── Engineered: Manager tenure gap ───────────────────
    -- Long gap between years at company and years with current manager
    -- suggests frequent manager turnover — a known flight risk signal
    YearsAtCompany - YearsWithCurrManager                   AS manager_tenure_gap,

    -- ── Categorical fields (encoded in Python for ML) ─────
    BusinessTravel,
    Department,
    EducationField,
    Gender,
    JobRole,
    MaritalStatus,
    OverTime

FROM hr_raw;


-- ── Validation: confirm shape ─────────────────────────────
SELECT
    COUNT(*)                                                 AS total_rows,
    COUNT(DISTINCT EmployeeNumber)                           AS unique_employees,
    SUM(attrition_flag)                                      AS total_attrition,
    ROUND(100.0 * SUM(attrition_flag) / COUNT(*), 1)        AS attrition_rate_pct
FROM hr_features;
