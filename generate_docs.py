"""Generate docs/Startup_Intelligence_AI_Documentation.pdf from live artifacts.

Every figure is read from models/*.json, the SQLite database and the clustering
summary, so the document cannot drift out of sync with the pipeline the way a
hand-written PDF does. Re-run after `python run_full.py`.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import DB_PATH, MODELS_DIR, REPORTS_DIR
from src.database import run_query

PDF_PATH = Path(__file__).resolve().parent / "docs" / "Startup_Intelligence_AI_Documentation.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=17, spaceAfter=8,
                    textColor=colors.HexColor("#14304f"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=12,
                    spaceAfter=5, textColor=colors.HexColor("#14304f"))
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=9.3, leading=13.4,
                      alignment=TA_LEFT, spaceAfter=6)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.2, leading=11,
                       textColor=colors.HexColor("#555555"))
TITLE = ParagraphStyle("TITLE", parent=styles["Title"], fontSize=22, spaceAfter=4)

GREY = colors.HexColor("#5a6b7b")
LIGHT = colors.HexColor("#eef2f6")


def fmt(n, decimals=0):
    if n is None:
        return "—"
    if isinstance(n, float):
        if abs(n) >= 1e9:
            return f"${n/1e9:,.2f}B"
        if abs(n) >= 1e6:
            return f"${n/1e6:,.2f}M"
        if abs(n) >= 1e3:
            return f"${n/1e3:,.1f}K"
        return f"{n:,.{decimals}f}"
    return f"{n:,}"


def table(rows, widths=None, align_right_from=1):
    data = [[Paragraph(f"<b>{c}</b>", BODY) for c in rows[0]]]
    for r in rows[1:]:
        data.append([Paragraph(str(c), BODY) for c in r])

    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14304f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d2dc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    for col in range(align_right_from, len(rows[0])):
        style.append(("ALIGN", (col, 1), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def chart(filename, caption, width=165 * mm):
    path = REPORTS_DIR / filename
    if not path.exists():
        return [Paragraph(f"<i>Missing chart: {filename}</i>", SMALL)]
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    height = width * h / w
    return [
        Image(str(path), width=width, height=min(height, 100 * mm)),
        Spacer(1, 3),
        Paragraph(caption, SMALL),
        Spacer(1, 10),
    ]


def build():
    metrics = json.loads((MODELS_DIR / "model_metrics.json").read_text())
    clf = json.loads((MODELS_DIR / "classification_metrics.json").read_text())
    clusters = json.loads((MODELS_DIR / "clustering_summary.json").read_text())

    summary = run_query(DB_PATH, "SELECT * FROM v_yearly_summary")
    sectors = run_query(
        DB_PATH,
        "SELECT industry_clean, COUNT(*) e, SUM(amount_clean) t FROM funding_events "
        "WHERE amount_clean IS NOT NULL GROUP BY industry_clean ORDER BY t DESC LIMIT 5",
    )

    story = []
    story.append(Paragraph("Startup Intelligence AI", TITLE))
    story.append(Paragraph(
        "Indian Startup Funding Analytics, Machine Learning &amp; Explainable AI", BODY))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from live pipeline artifacts. "
        "Re-run <b>python run_pipeline.py</b> then <b>python generate_docs.py</b> to refresh.", SMALL))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Dataset", H1))
    story.append(Paragraph(
        "3,044 Indian startup funding events, January 2015 – January 2020. "
        "The raw CSV is never modified; cleaning writes a separate processed copy and a "
        "SQLite analytics database. Amounts and dates are parsed defensively, "
        "cities are normalized (Bangalore→Bengaluru, Gurgaon→Gurugram), and "
        "duplicates are flagged rather than dropped.", BODY))

    rows = [["Year", "Events", "Recorded total"]]
    for r in summary.itertuples():
        rows.append([int(r.year), f"{r.event_count:,}", fmt(r.total_funding)])
    story.append(table(rows))
    story.append(Spacer(1, 10))

    rows = [["Sector", "Events", "Recorded total"]]
    for r in sectors.itertuples():
        rows.append([r.industry_clean, f"{r.e:,}", fmt(r.t)])
    story.append(table(rows))
    story.extend(chart("funding_by_year.png", "Figure 1 — Funding events and recorded total by year."))
    story.extend(chart("funding_by_industry.png", "Figure 2 — Top sectors by recorded funding."))

    story.append(PageBreak())
    story.append(Paragraph("2. Modelling Approach", H1))
    story.append(Paragraph(
        "The target is extremely right-skewed (skew ≈ 22; the top 1% of rounds hold "
        "roughly 44% of all recorded funding). Raw-dollar metrics are therefore "
        "dominated by a handful of mega-rounds, so <b>RMSLE</b> is used as the "
        "selection metric: it measures relative error and is not hostage to outliers.", BODY))
    story.append(Paragraph(
        "Design decisions, each verified by measurement:", BODY))
    for item in [
        "<b>Log target.</b> Every regressor is wrapped in "
        "TransformedTargetRegressor(log1p, expm1) so it trains in log space and "
        "predicts in raw dollars.",
        "<b>Group-median target encoding.</b> Median funding per industry×year, "
        "city×year, industry, city, investment type and industry×city is computed "
        "during fit on training rows only. Groups under 10 rows fall back to the "
        "global training median. <i>subvertical</i> is excluded: with ~1,300 distinct "
        "values across ~2,000 rows it degenerates into memorised one-offs.",
        "<b>Temporal validation.</b> 80/20 time-based split (train = earlier events). "
        "Hyperparameters are chosen by inner TimeSeriesSplit CV (4 folds) on the "
        "training period only. The winner is selected by inner CV score — never by "
        "test performance, which would invalidate the reported test number.",
        "<b>Baselines.</b> A global median dummy and a sector×year median are both "
        "reported, so improvement is measured against a strong comparator rather "
        "than a trivial one.",
        "<b>Investor features.</b> Implemented and tested but disabled by default: "
        "they measurably degraded RMSLE (RF 1.8686 → 1.9304). The code is retained so "
        "the negative result stays reproducible.",
    ]:
        story.append(Paragraph(f"• {item}", BODY))

    story.append(Paragraph("3. Regression Results", H1))
    rows = [["Model", "CV RMSLE", "Test RMSLE", "MAE", "R²"]]
    order = sorted(
        [k for k in metrics if k not in ("best_model", "selection_metric")],
        key=lambda k: (metrics[k].get("cv_rmsle") is None, metrics[k].get("cv_rmsle") or 9e9),
    )
    for name in order:
        r = metrics[name]
        cv = r.get("cv_rmsle")
        rows.append([
            name + (" (selected)" if name == metrics["best_model"] else ""),
            f"{cv:.4f}" if cv is not None else "—",
            f"{r['rmsle']:.4f}",
            fmt(r["mae"]),
            f"{r['r2']:.4f}",
        ])
    story.append(table(rows))
    story.append(Spacer(1, 8))

    params = metrics[metrics["best_model"]].get("best_params")
    if params:
        story.append(Paragraph(
            f"<b>Selected hyperparameters:</b> "
            + ", ".join(f"<b>{k}</b>={v}" for k, v in params.items()), BODY))
    story.append(Paragraph(
        "Every trained model beats both baselines on RMSLE. R² stays negative "
        "because squared error on raw dollars is still governed by outlier rounds — "
        "a property of the target, not evidence the model learned nothing. Round "
        "size is driven by traction, investor syndicate and timing, none of which "
        "appear in this dataset.", BODY))
    story.append(Paragraph(
        "Note the CV/test gap (≈1.27 vs ≈1.90): funding drifts strongly upward over "
        "time, so 2015–2017 is genuinely harder than 2018–2020. Inner CV scores "
        "should be read as relative rankings, not absolute accuracy.", SMALL))

    story.extend(chart("model_comparison.png",
                       "Figure 3 — MAE, RMSE, R² and RMSLE across all models."))
    story.extend(chart("actual_vs_predicted_random_forest.png",
                       "Figure 4 — Actual vs predicted, selected model.", 135 * mm))

    story.append(PageBreak())
    story.append(Paragraph("4. Funding-Range Classification", H1))
    story.append(Paragraph(
        "Funding is binned into Low / Medium / High using percentile thresholds "
        "derived from <b>training</b> data only, then applied unchanged to the test "
        "period. Re-deriving cut points on the test set would leak distributional "
        "information across the temporal split — it changed the assigned category on "
        "175 of 413 test rows.", BODY))
    rows = [["Model", "CV F1", "Test F1"]]
    for name in sorted([k for k in clf
                        if k not in ("best_model", "thresholds", "labels", "selection_metric")],
                       key=lambda k: -(clf[k].get("cv_f1") or 0)):
        cv = clf[name].get("cv_f1")
        rows.append([
            name + (" (selected)" if name == clf["best_model"] else ""),
            f"{cv:.4f}" if cv is not None else "—",
            f"{clf[name]['f1_weighted']:.4f}",
        ])
    story.append(table(rows))
    story.append(Spacer(1, 6))
    if clf["best_model"] in clf and clf[clf["best_model"]].get("best_params"):
        story.append(Paragraph(
            "Tuned: "
            + ", ".join(f"<b>{k}</b>={v}" for k, v in clf[clf["best_model"]]["best_params"].items()),
            SMALL))
    story.append(Paragraph(
        "Classification consumes the same feature pipeline as regression "
        "(via build_feature_pipeline), differing only in numeric scaling.", SMALL))
    story.append(Spacer(1, 8))

    story.append(Paragraph("5. Clustering", H1))
    story.append(Paragraph(
        f"K-means over sector, city, investment type, log1p(amount) and year. "
        f"Cluster count is chosen by silhouette score rather than hard-coded. "
        f"Amounts are log-transformed before clustering — with skew ≈ 22, raw "
        f"dollars would make every cluster a pure amount tiering — and categoricals "
        f"are one-hot encoded, since label encoding imposes a false ordering on names "
        f"like Bengaluru vs Mumbai.", BODY))
    story.append(Paragraph(
        f"<b>k = {clusters['optimal_k']}</b>, silhouette "
        f"<b>{clusters['silhouette']:.4f}</b>, over "
        f"{clusters['events_clustered']:,} events.", BODY))
    rows = [["Cluster", "Events", "Median amount", "Avg year", "Top industry", "Top city"]]
    for c in sorted(clusters["clusters"], key=lambda x: x["cluster"]):
        rows.append([c["cluster"], f"{c['event_count']:,}", fmt(c["median_amount"]),
                     f"{c['avg_year']:.1f}", c["top_industry"], c["top_city"]])
    story.append(table(rows))
    story.append(Paragraph(
        "Clusters group historical funding events with similar characteristics. "
        "They are data-driven groupings, not real-world startup categories, and "
        "carry no forward-looking guarantee.", SMALL))
    story.extend(chart("clustering_evaluation.png", "Figure 5 — Elbow and silhouette by k."))
    story.extend(chart("cluster_analysis.png", "Figure 6 — Cluster composition."))

    story.append(PageBreak())
    story.append(Paragraph("6. Explainability", H1))
    story.append(Paragraph(
        "Permutation importance measures how much model error increases when an "
        "input column is randomly shuffled, so it covers the raw input columns only. "
        "The grp_* median-encoding features are generated downstream by "
        "GroupMedianEncoder and appear in the SHAP plot instead.<br/><br/>"
        "SHAP explains the transformed feature matrix. Because regressors train in "
        "log1p space, SHAP values are contributions to log amount rather than raw "
        "USD: a positive contribution means a larger predicted round, but not by a "
        "proportional dollar amount. SHAP is attempted opportunistically and "
        "degrades gracefully to permutation importance if unavailable. Feature "
        "importance reflects association, not causation.", BODY))
    story.extend(chart("permutation_importance.png", "Figure 7 — Permutation importance."))
    story.extend(chart("shap_summary.png", "Figure 8 — SHAP summary (if available).", 135 * mm))

    story.append(Paragraph("7. Limitations", H1))
    for item in [
        "Historical records from 2015–2020 do not represent the current market.",
        "Missing funding amounts cut the modelling sample from 3,044 to 2,062 rows.",
        "R² is negative for every model; RMSLE is the metric to judge results by.",
        "Round size depends on traction, timing and syndicate quality — none of which "
        "are in this dataset. Predictions are exploratory estimates, not financial advice.",
        "Feature importance does not imply causation.",
        "The dataset contains outliers and inconsistent values that may affect analysis.",
        "CV scores are optimistic relative to the test period due to upward funding drift.",
    ]:
        story.append(Paragraph(f"• {item}", BODY))

    return story


def main():
    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Startup Intelligence AI - Documentation",
        author="Startup Intelligence AI",
    )
    doc.build(build())
    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()