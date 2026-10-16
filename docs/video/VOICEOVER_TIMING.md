# Voiceover timing sheet

The render is silent. Record against these timings; the burned-in
captions follow the same beats. Figures are read from the live artifacts
by `video_facts.py`, so this sheet regenerates rather than drifting.

| Scene | In | Out | Dur | Voiceover |
|---|---|---|---|---|
| 1. The Dataset | 0.0s | 6.0s | 6.0s | Three thousand forty four Indian startup funding rounds. Thirty eight billion dollars. And 979 of them with no disclosed amount. |
| 2. The Problem | 6.0s | 13.0s | 7.0s | 979 rows have no amount. 4 have unparseable dates. Most projects delete those and move on. |
| 3. Cleaning | 13.0s | 22.0s | 9.0s | 21 cities, 70 industry labels, 37 investment types. All normalized by hand. Bangalore and Bengaluru are the same city. |
| 4. Storage and BI | 22.0s | 31.0s | 9.0s | One SQLite table, four views, thirteen analysis queries. Then the same thing again in Power BI. |
| 5. Regression | 31.0s | 42.0s | 11.0s | Random Forest won, and it won on cross validation inside the training window. RMSLE 1.27 against a 2.43 dummy baseline. R squared is negative. I'm showing you that number too. |
| 6. The Anti-Leakage Call | 42.0s | 50.0s | 8.0s | Random Forest scored higher on the test set. I didn't pick it, because it lost on cross validation. Picking it on test would have been fitting the answer I wanted. |
| 7. Classification and Clustering | 50.0s | 60.0s | 10.0s | Classification splits rounds into low, medium and high. Thresholds from training percentiles only. Clustering lands on three segments. 2,062 events. |
| 8. Explainability | 60.0s | 69.0s | 9.0s | Permutation importance and SHAP. Investment type dominates. Most other features are near zero, or negative. Investor features made it worse, so they ship disabled. |
| 9. The App | 69.0s | 90.0s | 21.0s | Ten dashboard pages. 161 tests. A scheduled runner that skips itself when the CSV hasn't changed. A PDF generated from the live metrics, so the figures can't drift. Analytical prototype. Not investment advice. Which piece would you want the code for? |

Total **90s**, 216 spoken words (~144 wpm).

## Figures shown on screen

- Funding events: 3,044
- Total recorded funding: 38.06B USD
- Amounts missing: 979   invalid dates: 4
- Cities / industries / types normalized: 21 / 70 / 37
- Tests collected by pytest: 161
- RF CV RMSLE 1.2741 -> test RMSLE 1.9138, MAE 35.4M, R2 -0.0274
- Classifier: LogReg shipped (CV F1 0.5317, test F1 0.5386); RF test F1 0.5867 not shipped
- Clustering: k=3, silhouette 0.2182, 2,062 events

## Corrections against the original package

- test count: the package says 137; pytest collects 161
- cities normalized: the package says 20, CITY_NORMALIZATION has 21 entries
- group-feature ablation: the package shows 'RM 2.4029 -> 1.8686', but the shipped Random Forest test RMSLE in model_metrics.json is 1.9138
- permutation importance: the package calls city/year/month '~0'; they are negative and investors_name is the exact 0
- runtime: the package's 61 seconds is the --skip-shap run; a full run including SHAP is 96.1 seconds
