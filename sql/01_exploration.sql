-- ============================================================
-- 01_exploration.sql
-- Reference SQL queries for HR Attrition EDA
-- These are the same queries run inside the notebook via DuckDB
-- ============================================================


-- ── 1. Workforce snapshot ─────────────────────────────────
SELECT
    COUNT(*)                                                          AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)              AS attrited,
    SUM(CASE WHEN Attrition = 'No'  THEN 1 ELSE 0 END)              AS retained,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                                 AS attrition_rate_pct,
    ROUND(AVG(Age), 1)                                               AS avg_age,
    ROUND(AVG(MonthlyIncome), 0)                                     AS avg_monthly_income,
    ROUND(AVG(YearsAtCompany), 1)                                    AS avg_tenure_years
FROM hr_raw;


-- ── 2. Attrition rate by department ───────────────────────
SELECT
    Department,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)              AS attrited,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                                 AS attrition_rate_pct
FROM hr_raw
GROUP BY Department
ORDER BY attrition_rate_pct DESC;


-- ── 3. Attrition by age group ─────────────────────────────
SELECT
    CASE
        WHEN Age < 25               THEN 'Under 25'
        WHEN Age BETWEEN 25 AND 34  THEN '25-34'
        WHEN Age BETWEEN 35 AND 44  THEN '35-44'
        WHEN Age BETWEEN 45 AND 54  THEN '45-54'
        ELSE '55+'
    END                                                               AS age_group,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)              AS attrited,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                                 AS attrition_rate_pct
FROM hr_raw
GROUP BY 1
ORDER BY
    CASE age_group
        WHEN 'Under 25' THEN 1
        WHEN '25-34'    THEN 2
        WHEN '35-44'    THEN 3
        WHEN '45-54'    THEN 4
        ELSE 5
    END;


-- ── 4. Attrition by tenure bucket ────────────────────────
-- KEY FINDING: early-tenure employees (0-2 years) are highest risk
SELECT
    CASE
        WHEN YearsAtCompany = 0              THEN '< 1 year'
        WHEN YearsAtCompany BETWEEN 1 AND 2  THEN '1-2 years'
        WHEN YearsAtCompany BETWEEN 3 AND 5  THEN '3-5 years'
        WHEN YearsAtCompany BETWEEN 6 AND 10 THEN '6-10 years'
        ELSE '10+ years'
    END                                                               AS tenure_bucket,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)              AS attrited,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                                 AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0)                                     AS avg_monthly_income
FROM hr_raw
GROUP BY 1
ORDER BY
    CASE tenure_bucket
        WHEN '< 1 year'   THEN 1
        WHEN '1-2 years'  THEN 2
        WHEN '3-5 years'  THEN 3
        WHEN '6-10 years' THEN 4
        ELSE 5
    END;


-- ── 5. Satisfaction x Overtime cross-tab (heatmap source) ─
SELECT
    JobSatisfaction,
    OverTime,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)              AS attrited,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                                 AS attrition_rate_pct
FROM hr_raw
GROUP BY JobSatisfaction, OverTime
ORDER BY JobSatisfaction, OverTime;


-- ── 6. Attrition by job role ──────────────────────────────
SELECT
    JobRole,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)              AS attrited,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                                 AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0)                                     AS avg_monthly_income,
    ROUND(AVG(JobSatisfaction), 2)                                   AS avg_satisfaction
FROM hr_raw
GROUP BY JobRole
ORDER BY attrition_rate_pct DESC;


-- ── 7. Monthly income distribution: leavers vs stayers ───
SELECT
    Attrition,
    ROUND(MIN(MonthlyIncome), 0)    AS min_income,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY MonthlyIncome), 0) AS q1_income,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY MonthlyIncome), 0) AS median_income,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY MonthlyIncome), 0) AS q3_income,
    ROUND(MAX(MonthlyIncome), 0)    AS max_income,
    ROUND(AVG(MonthlyIncome), 0)    AS avg_income
FROM hr_raw
GROUP BY Attrition;
