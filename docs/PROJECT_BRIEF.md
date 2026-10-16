# Startup Intelligence AI — Original Project Brief

> **Status: superseded.** This is the original planning document, kept for
> reference only. It contains no measured results and describes the project as
> it was scoped *before* implementation.
>
> **For current documentation and measured results, see [`../README.md`](../README.md).**
>
> Where this brief and the README disagree on a number (dataset size, model
> metrics, test count), the README is authoritative — its figures are generated
> by `python run_full.py`.

## Indian Startup Funding Analytics, Machine Learning & Explainable AI

**Project type:** Data Analytics + Machine Learning + AI  
**Dataset:** Indian startup funding events, January 2015–January 2020  
**Dataset size discussed:** 3,044 funding events (verify against the actual source file)  
**Status (at time of writing):** Project plan / implementation blueprint

---

## 1. Project Overview

Startup Intelligence AI upgrades a traditional Indian startup funding analysis project into an end-to-end analytics and machine-learning application. It combines data cleaning, SQL analysis, exploratory data analysis (EDA), predictive modeling, explainable AI, and interactive dashboards.

The goal is to help users explore historical startup funding patterns and understand what a model learns from available historical data. The project is an analytical prototype, **not investment advice** and not a guarantee of future startup success.

## 2. Business Problem

Startup funding data can be difficult to interpret because records may contain inconsistent startup names, missing funding amounts, mixed date formats, different sector labels, and incomplete investor information.

This project aims to answer questions such as:

- Which sectors attracted the most recorded funding?
- How did funding activity change over time?
- Which cities had the highest recorded funding activity?
- Which investors appeared most frequently in the dataset?
- What patterns distinguish different funding events?
- How accurately can a model estimate a funding amount or classify a funding range using information available at prediction time?
- Which features influence a model's predictions?

## 3. Project Objectives

1. Audit, clean, and document the source dataset.
2. Perform SQL analysis and exploratory data analysis.
3. Build reproducible data-preprocessing and feature-engineering pipelines.
4. Train and evaluate a baseline funding prediction model.
5. Optionally classify funding events into documented funding ranges.
6. Explore startup/funding-pattern clusters where the available features support meaningful grouping.
7. Explain model outputs with feature importance and SHAP.
8. Present analytical findings and model limitations in a dashboard.
9. Document setup, methodology, evaluation, and reproducibility in GitHub.

## 4. Dataset

The working dataset is described as containing **3,044 Indian startup funding events from January 2015 to January 2020**. Confirm the count, date coverage, license, and schema from the actual file before publishing results.

Potential fields to inspect (do not assume all are present):

- Startup/company name
- Funding date
- Funding amount
- Industry/sector
- City/location
- Investors
- Funding round or stage

### Data quality checks

- Missing values and missingness patterns
- Duplicate rows versus legitimate multiple funding rounds
- Inconsistent company, sector, and city names
- Date parsing and out-of-range dates
- Currency and number-format consistency
- Implausible values and outliers
- Records with unknown or ambiguous funding amounts
- Target leakage and features unavailable at prediction time

Do not automatically replace missing funding amounts with zero. Keep raw data unchanged and write cleaned data to a separate location.

## 5. Analytics Questions

### SQL and EDA

- Total recorded funding and number of funding events
- Funding activity by year, sector, city, and stage (where available)
- Median and distribution of funding amounts
- Most active recorded investors (if investor fields are reliable)
- Sector-level and city-level trends over time
- Missingness and data-quality summary

Use medians and distributions alongside totals because funding amounts can be highly skewed by a small number of large rounds.

## 6. Machine Learning Plan

### Model A — Funding Amount Estimation

**Goal:** Estimate the amount for a clearly defined funding event using only features available at the prediction point.

Candidate algorithms:
- DummyRegressor as a baseline
- Linear Regression or a regularized regression model
- Random Forest Regressor
- XGBoost Regressor (optional)

Evaluation:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²), reported with context

Funding amounts are often highly skewed. Investigate transformations such as `log1p` only when appropriate, and explain how predictions are converted back to the original currency scale.

### Model B — Funding Range Classification (optional)

**Goal:** Classify events into low, medium, or high funding categories.

Requirements:
- Define category thresholds transparently using training data or a documented business rule.
- Avoid creating thresholds from the full dataset before the split.
- Compare with a simple baseline.
- Report precision, recall, F1-score, and a confusion matrix.
- Consider class imbalance.

### Model C — Funding Pattern Clustering (optional)

**Goal:** Explore groups of funding events with similar characteristics.

Candidate algorithm: K-Means, with scaling and suitable encoding where required. DBSCAN may be considered if the data shape supports it.

Evaluate clusters using silhouette score where appropriate and, more importantly, interpretability and stability. Do not label clusters as real-world startup types unless the evidence supports those labels.

## 7. Explainable AI

Use permutation importance and/or SHAP to explain model behavior.

Possible outputs:
- Overall feature importance
- SHAP summary plot
- Explanation for an individual prediction
- Plain-language explanation of which features pushed a prediction higher or lower

**Important:** feature importance describes associations learned by the model; it does not prove that a feature causes funding outcomes.

## 8. Validation and Leakage Prevention

- Prefer a time-based evaluation when the objective is to estimate performance on later funding events.
- Fit imputers, encoders, scalers, feature selection, and category thresholds using training data only.
- Keep the test set untouched until final evaluation.
- Ensure every feature would be available at the time the prediction is intended to be made.
- Consider repeated events from the same startup when choosing a split; explain how this affects generalization.
- Compare model performance with a simple baseline.
- Report actual measured results; never invent accuracy or claim a model is successful before evaluation.

The 2015–2020 dataset is historical. It cannot by itself describe current market conditions or guarantee future funding.

## 9. Dashboard and Application

Suggested application: Streamlit + Plotly. Power BI can be used for the business intelligence dashboard.

Suggested pages:
1. **Overview:** dataset size, date range, recorded funding totals, event counts, and data-quality indicators.
2. **Funding Trends:** year, sector, city, and stage analysis.
3. **Investor Insights:** investor activity where data is reliable.
4. **ML Evaluation:** baseline comparison and validation metrics.
5. **Funding Estimator:** user inputs and model prediction, only for fields supported by the trained model.
6. **Explainability:** feature importance and individual prediction explanation.
7. **Methodology:** data limitations, assumptions, and evaluation design.

Show the historical data period clearly in the interface.

## 10. Technology Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **Database/querying:** SQL; SQLite or PostgreSQL as needed
- **Visualization:** Matplotlib, Seaborn, Plotly, Power BI
- **Machine learning:** Scikit-learn; XGBoost optional
- **Explainability:** SHAP, permutation importance
- **Application:** Streamlit; FastAPI optional
- **Development:** Jupyter Notebook, VS Code, Git, GitHub

Start with free, local tools. An LLM API is not required for the core analytics or ML models.

## 11. Suggested Repository Structure

```text
startup-intelligence-ai/
├── data/
│   ├── raw/                 # Keep original data unchanged; do not commit sensitive/large files
│   └── processed/           # Reproducibly generated cleaned data
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_eda_and_sql.ipynb
│   ├── 03_funding_prediction.ipynb
│   ├── 04_clustering.ipynb
│   └── 05_model_explainability.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

Adjust the structure as the implementation evolves. Do not commit model artifacts, raw datasets, credentials, or private data unless appropriate.

## 12. Implementation Roadmap

- [ ] **Phase 1 — Data audit:** confirm source, schema, license, row count, and date coverage.
- [ ] **Phase 2 — Cleaning:** document cleaning rules and generate a reproducible cleaned dataset.
- [ ] **Phase 3 — SQL + EDA:** create queries, charts, and evidence-backed findings.
- [ ] **Phase 4 — Baseline model:** define one prediction target and establish a baseline.
- [ ] **Phase 5 — Model evaluation:** test appropriate algorithms with leakage-safe validation.
- [ ] **Phase 6 — Explainability:** add feature importance/SHAP if supported.
- [ ] **Phase 7 — Dashboard:** build a Streamlit or Power BI experience.
- [ ] **Phase 8 — Documentation:** add screenshots, setup instructions, measured metrics, limitations, and reproducibility steps.

## 13. Definition of Done

The project is ready to present when:

- [ ] Data source and license are documented.
- [ ] Data cleaning is reproducible.
- [ ] SQL/EDA findings can be traced to queries or notebook cells.
- [ ] The target variable and prediction timing are clearly defined.
- [ ] Evaluation uses an appropriate held-out set and baseline.
- [ ] Metrics are actual results, not placeholders.
- [ ] The dashboard runs from documented instructions.
- [ ] Limitations and historical data dates are visible.
- [ ] README includes screenshots and steps to reproduce results.

## 14. Resume Description (use only after implementation)

**Startup Intelligence AI — Indian Startup Funding Analytics & Prediction**  
*Python, Pandas, SQL, Scikit-learn, SHAP, Power BI, Streamlit*

- Analyzed historical Indian startup funding events to investigate funding trends, sector performance, geographic distribution, and investor activity.
- Developed and evaluated machine-learning models for funding estimation and/or funding-range classification using leakage-aware validation.
- Applied clustering to explore funding patterns and explainable-AI techniques to interpret model predictions.
- Built an interactive dashboard combining historical analytics, model evaluation, and prediction explanations.

Replace or remove any bullet that has not been implemented. Add verified dataset counts and measured model metrics only after validating them.

## 15. Important Limitations

- Historical records from January 2015–January 2020 do not represent the current startup market.
- A small or incomplete dataset may limit model generalization.
- Recorded funding amounts may be skewed and may contain missing or inconsistent values.
- Investor frequency is not the same as investor quality or future performance.
- Model predictions are exploratory estimates, not financial advice or guarantees.
