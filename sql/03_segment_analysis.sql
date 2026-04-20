-- ============================================================
-- 03_segment_analysis.sql
-- Risk segment queries — the "business insight" layer
-- These drive the HRBP dashboard section of the notebook
-- ============================================================


-- ── 1. High-risk role + department combinations ───────────
-- Minimum 15 employees in segment for statistical reliability
SELECT
    Department,
    JobRole,
    COUNT(*)                                                          AS headcount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)              AS attrited,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                                 AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0)                                     AS avg_monthly_income,
    ROUND(AVG(JobSatisfaction), 2)                                   AS avg_job_satisfaction,
    SUM(CASE WHEN OverTime = 'Yes' THEN 1 ELSE 0 END)               AS overtime_headcount,
    ROUND(
        100.0 * SUM(CASE WHEN OverTime = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                                 AS overtime_pct
FROM hr_raw
GROUP BY Department, JobRole
HAVING COUNT(*) >= 15
ORDER BY attrition_rate_pct DESC;


-- ── 2. Overtime x Marital Status — a hidden risk signal ──
-- Single employees on overtime have the highest attrition
SELECT
    MaritalStatus,
    OverTime,
    COUNT(*)                                                          AS total,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                                 AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0)                                     AS avg_income
FROM hr_raw
GROUP BY MaritalStatus, OverTime
ORDER BY attrition_rate_pct DESC;


-- ── 3. Retention risk by satisfaction + income band ──────
SELECT
    CASE
        WHEN MonthlyIncome < 3000            THEN 'Low (< $3K)'
        WHEN MonthlyIncome BETWEEN 3000 AND 7000 THEN 'Mid ($3K-$7K)'
        ELSE 'High (> $7K)'
    END                                                               AS income_band,
    CASE
        WHEN JobSatisfaction IN (1, 2) THEN 'Low Satisfaction'
        ELSE 'High Satisfaction'
    END                                                               AS satisfaction_tier,
    COUNT(*)                                                          AS total,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                                 AS attrition_rate_pct
FROM hr_raw
GROUP BY 1, 2
ORDER BY attrition_rate_pct DESC;


-- ── 4. Promotion stagnation effect ────────────────────────
SELECT
    CASE
        WHEN YearsSinceLastPromotion = 0     THEN 'Recently Promoted'
        WHEN YearsSinceLastPromotion BETWEEN 1 AND 2 THEN '1-2 years ago'
        WHEN YearsSinceLastPromotion BETWEEN 3 AND 5 THEN '3-5 years ago'
        ELSE '5+ years ago'
    END                                                               AS last_promotion,
    COUNT(*)                                                          AS total,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                                 AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0)                                     AS avg_income
FROM hr_raw
GROUP BY 1
ORDER BY
    CASE last_promotion
        WHEN 'Recently Promoted' THEN 1
        WHEN '1-2 years ago'     THEN 2
        WHEN '3-5 years ago'     THEN 3
        ELSE 4
    END;


-- ── 5. The triple risk flag: Overtime + Low Satisfaction + Early Tenure ──
-- Employees who tick all three boxes — most actionable retention target
SELECT
    Department,
    COUNT(*)                                                          AS high_risk_employees,
    ROUND(AVG(MonthlyIncome), 0)                                     AS avg_income,
    ROUND(AVG(Age), 1)                                               AS avg_age
FROM hr_raw
WHERE
    OverTime = 'Yes'
    AND JobSatisfaction <= 2
    AND YearsAtCompany <= 2
GROUP BY Department
ORDER BY high_risk_employees DESC;
