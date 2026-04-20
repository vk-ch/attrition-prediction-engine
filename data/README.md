# Data

## Source

**IBM HR Analytics Employee Attrition & Performance Dataset**  
Originally published by IBM data scientists as a fictional dataset for exploring people analytics problems.

- **Kaggle:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset  
- **Rows:** 1,470 employees  
- **Columns:** 35 features  
- **Target:** `Attrition` (Yes / No) — 16.1% positive rate

## How to Get the Data

The notebook loads the dataset automatically from a public URL. If that fails:

1. Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from the Kaggle link above
2. Place it in this `data/` folder
3. The notebook will fall back to reading from this path automatically

## Key Columns

| Column | Type | Description |
|--------|------|-------------|
| Attrition | Target | Whether the employee left (Yes/No) |
| Age | Numeric | Employee age |
| MonthlyIncome | Numeric | Monthly salary in USD |
| YearsAtCompany | Numeric | Tenure in years |
| JobSatisfaction | Ordinal | 1 (Low) to 4 (Very High) |
| EnvironmentSatisfaction | Ordinal | 1 (Low) to 4 (Very High) |
| OverTime | Binary | Whether employee works overtime (Yes/No) |
| Department | Categorical | HR, R&D, Sales |
| JobRole | Categorical | 9 distinct roles |
| WorkLifeBalance | Ordinal | 1 (Bad) to 4 (Best) |

## Notes on Data Quality

- No missing values in this dataset
- Three columns are constant (dropped in analysis): `EmployeeCount`, `Over18`, `StandardHours`
- `EmployeeNumber` is an ID column — not used as a feature
