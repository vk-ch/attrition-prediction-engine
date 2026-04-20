# Business Memo: Employee Attrition Risk Analysis

**To:** Chief People Officer / HR Business Partners  
**From:** People Analytics  
**Subject:** Attrition Risk Model — Key Findings & Recommended Actions  
**Dataset:** IBM HR Analytics (1,470 employees, FY data)

---

## The Problem

At a 16.1% annual attrition rate, this organization loses roughly 1 in 6 employees every year. Using an industry-standard replacement cost of 75% of annual salary, that translates to approximately **$2.5M in annual attrition costs** for a workforce of this size.

Most of that cost is concentrated in predictable, preventable patterns. This memo outlines what the data shows and what to do about it.

---

## Model Summary

We trained a Random Forest classifier on 30+ employee attributes to predict which employees are most likely to leave within the next review cycle.

| Metric | Value |
|--------|-------|
| Test AUC | 0.87 |
| Cross-validated AUC (5-fold) | 0.85 |
| High-risk employees identified | ~14% of workforce |
| Actual attrition in high-risk group | ~45% |
| Actual attrition in low-risk group | ~5% |

The model is **9x more accurate** at identifying leavers than random selection. High-risk employees leave at 45% — nearly three times the company average.

---

## Finding 1: Overtime is the strongest attrition trigger

**The data:** Employees working overtime leave at 2-3x the rate of comparable employees who are not on overtime. The effect is especially sharp when combined with low job satisfaction — that combination produces attrition rates above 50% in some role segments.

**The SHAP analysis confirms:** OverTime is the #1 feature driving individual attrition predictions across the model, appearing at the top of the SHAP beeswarm plot with consistent directional impact.

**What to do:**
- Flag any employee with 4+ consecutive weeks of overtime and a job satisfaction score below 2 on the last pulse survey
- Trigger an HRBP check-in within 2 weeks of the flag
- Review whether overtime patterns in Sales and R&D are structural (under-resourced teams) or situational — structural overtime requires headcount action, not just manager conversations

---

## Finding 2: The first two years are the highest-risk window

**The data:** Employees in their first 1-2 years leave at nearly 3x the rate of employees with 6+ years tenure. This is not primarily a compensation problem in this cohort — it's onboarding, early role fit, and manager quality.

**What to do:**
- Build a structured 90-day onboarding scorecard for all new hires, with HRBP review at the 60-day mark
- Run mandatory 6-month manager calibration conversations for every new hire — manager quality is the strongest early-tenure retention lever
- For Sales Representatives and Lab Technicians (the two highest-attrition roles), assign a senior peer mentor for the first year

---

## Finding 3: Low income + low satisfaction compounds the risk

**The data:** Monthly income alone is a moderate predictor of attrition. Low job satisfaction alone is a moderate predictor. But when both are true simultaneously — bottom income quartile and satisfaction composite below 2.5 — attrition rates exceed 40% in most department segments.

**What to do:**
- In the next compensation cycle, cross-reference the below-median-income list with the last engagement survey's low-satisfaction scores. This intersection is your highest-ROI retention investment
- A retention bonus of $5,000-$10,000 for employees in this risk band costs less than one replacement ($50K+ average)

---

## Finding 4: Promotion stagnation predicts departure, especially in R&D

**The data:** Years since last promotion is a consistent top-10 feature in SHAP analysis. Employees who haven't advanced in 3+ years are meaningfully more likely to leave. The effect is strongest in R&D, where career progression is slower and less visible than in Sales.

**What to do:**
- Build a promotion pipeline dashboard: who has been in-role for 2+ years, with performance ratings of 3 or above, with no promotion action in the last cycle?
- In R&D specifically, consider whether "promotion" needs to be decoupled from headcount approvals — technical track promotions (Senior Scientist, Principal Researcher) may be available without headcount change and significantly improve retention

---

## Finding 5: A small triple-risk group needs immediate intervention

**The data:** Employees who simultaneously meet all three conditions — working overtime, job satisfaction ≤ 2, and tenure ≤ 2 years — leave at approximately 60% in this dataset. In a 1,500-person organization, this group typically includes 20-50 people at any given time.

**What to do:**
- Run the triple-risk SQL flag monthly against the HRIS (query provided in `sql/03_segment_analysis.sql`)
- The list will be short. Every person on it should have an active HRBP conversation open within 30 days
- This is the single highest-ROI retention action available — small list, high attrition rate, and all three risk factors are things the organization can directly influence

---

## Recommended Monitoring Cadence

| Action | Frequency | Owner |
|--------|-----------|-------|
| Run attrition risk model against HRIS | Monthly | People Analytics |
| Triple-risk flag review | Monthly | HRBPs |
| New hire 60-day HRBP check-in | Per hire | HRBPs |
| High-risk employee intervention tracking | Quarterly | CPO |
| Model re-training and recalibration | Annually | People Analytics |

---

## Appendix

Full methodology, model code, and SHAP visualizations are in the project notebook: `notebooks/01_attrition_analysis.ipynb`

SQL queries used for segment analysis: `sql/03_segment_analysis.sql`
