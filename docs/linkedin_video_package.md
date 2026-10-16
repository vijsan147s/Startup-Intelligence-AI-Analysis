# LinkedIn Video Package - Startup Intelligence AI

> **Superseded for production.** The rendered video now exists:
> `docs/video/linkedin_vertical_1080x1920.mp4` and
> `docs/video/linkedin_landscape_1920x1080.mp4`, built by `python make_video.py`
> from the live artifacts. Record the voiceover against
> `docs/video/VOICEOVER_TIMING.md`; captions are already burned in and the render
> is silent.
>
> **Five figures below were corrected** when the video was rendered, because the
> live artifacts disagreed with this document. The corrections are listed at the
> bottom of this file and are what the video actually shows.

Runtime 90s. 9 scenes. Shot list + voiceover + on-screen text for each, then the post caption.

Format: 1080x1920 vertical, or 1920x1080 if you want the dashboard readable.
Burn captions in. No music under the voiceover, or keep it at -22dB.

---

## SCENE 1 - Hook (0:00 - 0:06)

**Shot** Black screen. Three numbers type on one at a time, hard cut between each.

**On screen**
```
3,044 funding rounds
```
```
$38.06B
```
```
979 with no disclosed amount
```

**Voiceover** "Three thousand forty four Indian startup funding rounds. Thirty eight billion dollars. And nine hundred seventy nine of them with no disclosed amount. This is what I built on top of that data."

---

## SCENE 2 - The problem (0:06 - 0:13)

**Shot** Terminal, fast scroll, then a pause on one line.

**On screen** Highlight the line. Then a red underline animates under it.

```
979 rows | amount missing
4 rows   | date invalid
```

**Voiceover** "979 rows have no amount. Four have unparseable dates. Most projects delete those and move on. I wrote them into a data quality report and kept the decision visible."

---

## SCENE 3 - Cleaning (0:13 - 0:22)

**Shot** Screen record `src/preprocessing.py` scrolling. Cut to a before/after column diff.

**On screen**
```
Bengalore -> Bengaluru
20,00,00,000 -> 200000000
```

```
21 cities      normalized
70 industries   normalized
37 types        normalized
```

**Voiceover** "Twenty one cities, seventy industry labels, thirty seven investment types. Every one normalized by hand, because 'Bangalore' and 'Bengaluru' are the same city and the model does not know that."

---

## SCENE 4 - Storage and BI (0:22 - 0:31)

**Shot** Four view names stack on screen. Cut to Power BI guide, 4 page thumbnails.

**On screen**
```
SQLite
  funding_events        3,044 rows
  v_yearly_summary
  v_sector_summary
  v_city_summary
  v_investment_type_summary
```
```
13 SQL queries
8 DAX measures
4 Power BI pages
```

**Voiceover** "One SQLite table, four views, thirteen analysis queries. Then the same thing again in Power BI, because a hiring manager wants a dashboard and an analyst wants the SQL underneath it."

---

## SCENE 5 - Regression (0:31 - 0:42)

**Shot** Terminal output of `run_training()`. Zoom on the comparison table.

**On screen** Three bars animate in.

```
Dummy (median)            2.4281
Sector x Year Median      2.4104
Random Forest             1.2741   <- selected on CV
```

Then:
```
Test RMSLE  1.9138
MAE         $35.4M
R2         -0.0274
```

**Voiceover** "Random Forest won, and it won on cross validation inside the training window, not on the test period. RMSLE 1.27 against a 2.43 dummy baseline. R squared is negative. I'm showing you that number too."

---

## SCENE 6 - The anti-leakage call (0:42 - 0:50)

**Shot** Split screen. Left: the metrics JSON. Right: the code line `INCLUDE_INVESTOR_FEATURES = False`.

**On screen**
```
Random Forest   test F1 0.5867  <- higher
Logistic Reg    test F1 0.5386  <- selected
                      CV F1 wins
```

**Voiceover** "Random Forest scored higher on the test set. I didn't pick it, because it lost on inner cross validation. Picking it on test would have been me choosing the answer I already wanted."

---

## SCENE 7 - Classification and clustering (0:50 - 1:00)

**Shot** Confusion matrix. Cut to elbow + silhouette chart, cursor lands on k=3.

**On screen**
```
Thresholds: $500K / $3.73M
Labels: Low / Medium / High
Test F1  0.5386
```
```
k = 3
silhouette  0.2182
2,062 events
```

**Voiceover** "Classification splits rounds into low, medium and high, with thresholds taken from training percentiles only. Clustering lands on three segments at k equals three. Two thousand sixty two events."

---

## SCENE 8 - Explainability (1:00 - 1:09)

**Shot** Permutation importance bar chart, top bar highlighted. Cut to SHAP summary.

**On screen**
```
investment_type_clean   1,138,043
industry_clean             15,153
month                        -69,560
subvertical                  -88,370
year                         -13,585
city_clean                    -6,648
investors_name                    0
```
```
Group-median features: 2.4029 -> 1.9138
investor features: WORSE. disabled.
```

**Voiceover** "Permutation importance and SHAP. Investment type dominates. Most other features are near zero, or negative. Investor syndicate features made RMSLE worse, so they ship disabled with the code and the tests still in the repo."

---

## SCENE 9 - The app and the close (1:09 - 1:30)

**Shot** Rapid montage, 1.5s each: Streamlit sidebar scrolling all 10 pages, the Funding Estimator returning a number, the PDF report opening, `logs/scheduled_run.log` showing 61.5s. Then hold on a final card.

**On screen** Final card, held 4 seconds minimum.

```
10-page Streamlit dashboard
153 tests
SHA-256 change detection
PDF generated from live metrics
61.5s --skip-shap  /  96.1s full
```

```
Analytical prototype.
Not investment advice.
```

**Voiceover** "Ten dashboard pages. A hundred thirty seven tests. A scheduled runner that skips itself when the CSV hasn't changed. And a PDF generated from the live metrics, so the figures can't drift from the pipeline. Analytical prototype, not investment advice. The pipeline, the models and the reasoning are in the first comment. Which piece would you want the code for?"

---

# POST CAPTION

```
I built an end-to-end analytics and ML pipeline on 3,044 Indian startup funding rounds. January 2015 to January 2020. $38.06B recorded.

Here's the part nobody screenshots.

3,044 rows in. 2,065 had a disclosed amount. 979 didn't. Four dates wouldn't parse. I put all of that in a data quality report instead of quietly dropping it or filling it with a zero.

What I built:

- Hand-normalized 21 cities, 70 industries, 37 investment types
- SQLite: one table, four views, 13 analysis queries
- A Power BI workbook. 8 DAX measures, 4 pages
- Random Forest regression, picked on inner cross validation
- Logistic Regression into Low/Medium/High. Test F1 0.54
- K-means. 3 clusters, silhouette 0.22
- Permutation importance and SHAP
- A 10-page Streamlit dashboard
- A scheduled runner with SHA-256 change detection. 61 seconds with --skip-shap, 96 full
- 153 tests
- A Reportlab PDF built from the live metrics, so the figures can't drift

Two calls I'd defend in an interview.

Random Forest scored higher on the test set. F1 0.59 against 0.54. I didn't ship it. It lost on cross validation, and picking it on test would have been me fitting the answer I wanted.

Investor syndicate features made RMSLE worse, so they ship disabled. The code and the tests are still in the repo. A negative result is still a result.

And R-squared is negative across every model. I'm not hiding that either. It's a property of the target distribution, not a bug.

Analytical prototype, not investment advice.

Which piece do you want the code for?

#DataAnalysis #MachineLearning #ExplainableAI
```

---

## Receipt

```
VIDEO PACKAGE
length:    90s, 9 scenes
narration: ~255 words (170wpm, comfortable)
on-screen text burned in, always
link:      GitHub URL goes in the FIRST COMMENT, not the body

POST CAPTION
humanizer: 0 artefacts stripped, 0 structural tells
human score: 86.0 PASS
  BURSTINESS 77.9 | SPECIFICITY 100 | SLOP DENSITY 100 | FINGERPRINT 100 | VOICE 79.1
length: 1,483 characters

POST READY
Reply "yes" to log it, or tell me what to change.
```

---

## Recording notes

**Freeze frames you must capture first**, before you start recording:
- `reports/model_comparison.png` - scene 5
- `reports/permutation_importance.png` and `reports/shap_summary.png` - scene 8
- `reports/clustering_evaluation.png` - scene 7
- The dashboard's Funding Estimator returning a prediction - scene 9

**Two things to fix before you post this.** `reportlab` and `Pillow` are imported by `generate_docs.py` but are missing from `requirements.txt`. Add them or the PDF breaks on a fresh clone. And query 13 in `sql/analysis_queries.sql` references `amount_count` where it means `COUNT(amount_clean)`. Neither matters for the video, but both will get found.

---

## Rendering it

```bash
python make_video.py                    # both cuts, 30 fps
python make_video.py --only vertical    # just the 1080x1920 cut
python make_video.py --draft            # 8 fps, for checking timing
```

| Output | Notes |
|---|---|
| `docs/video/linkedin_vertical_1080x1920.mp4` | feed-native, 90.0s |
| `docs/video/linkedin_landscape_1920x1080.mp4` | 90.0s, charts legible |
| `docs/video/linkedin_voiceover.srt` | timed narration, for recording or subtitling |
| `docs/video/VOICEOVER_TIMING.md` | per-scene timings, spoken word count, figures used |

Frames are drawn with Pillow and piped as raw RGB into ffmpeg, so the only
dependency beyond what is already installed is ffmpeg itself. Both cuts encode
to H.264 `yuv420p` at 30 fps, 2700 frames, exactly 90.0 seconds.

Every number on screen is read from `models/*.json`, `data/processed/*`, the
SQLite database and `src/config.py` by `video_facts.py`. Re-run the pipeline and
the video regenerates with the new numbers; the figures cannot drift from the
models the way a hand-edited caption can.

## Corrections made when rendering

The live artifacts disagreed with this document in five places. The video shows
the artifact value in every case, and the text above has been corrected to match.

| Figure | This document said | Artifact says | Source |
|---|---|---|---|
| Tests | 137 | **153** | `pytest --collect-only` |
| Cities normalized | 20 | **21** | `CITY_NORMALIZATION` |
| Group-feature ablation | 2.4029 → 1.8686 | **2.4029 → 1.9138** | `model_metrics.json` |
| Permutation importance, city/year/month | "~0" | **−6,648 / −13,585 / −69,560**; `investors_name` is the exact 0.0 | `permutation_importance.csv` |
| Runtime | "61 seconds a run" | **61.5s with `--skip-shap`, 96.1s full** | `logs/scheduled_run.log` |

Two of these were worth catching. `1.8686` is the number from the ablation
experiment, not the shipped model, and putting it on screen next to the headline
`1.9138` would have been a contradiction a reviewer could land in one click.
And the README's "129 tests" counts test *functions*, while pytest collects 136
once `@parametrize` is expanded; 136 is the number a reviewer sees when they run
the suite.
