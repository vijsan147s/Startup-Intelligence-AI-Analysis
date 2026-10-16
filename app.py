import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    CLEANED_CSV,
    DB_PATH,
    MAX_PREDICTABLE_YEAR,
    MIN_PREDICTABLE_YEAR,
    MODELS_DIR,
    PROCESSED_DIR,
    QUALITY_REPORT,
    RAW_CSV,
    REPORTS_DIR,
)
from src.clustering import load_cluster_summary
from src.data_loader import load_raw_csv, run_data_audit
from src.database import create_database, get_db_summary, run_query
from src.explainability import compute_permutation_importance, plot_permutation_importance
from src.classification import run_classification
from src.features import create_time_split, prepare_features
from src.predict import FundingPredictor
from src.preprocessing import preprocess
from src.train import run_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Startup Intelligence AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "Overview",
    "Funding Trends",
    "Industry Insights",
    "City Insights",
    "Investor Insights",
    "Machine Learning",
    "Clustering",
    "Funding Estimator",
    "Explainable AI",
    "Methodology",
]


@st.cache_data
def load_data():
    if CLEANED_CSV.exists():
        df = pd.read_csv(CLEANED_CSV)
    else:
        raw = load_raw_csv(RAW_CSV)
        df = preprocess(raw)
    if "date_parsed" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date_parsed"], format="mixed", errors="coerce")
    # CSV round-trip turns empty strings and literals like "NA"/"N/A" into NaN;
    # normalize back to "Unknown" so sorted()/filters never hit float-vs-str errors.
    for col in ["industry_clean", "subvertical", "city_clean", "investment_type_clean"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    return df


@st.cache_data
def load_quality_report():
    if QUALITY_REPORT.exists():
        with open(QUALITY_REPORT) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_db_summary():
    if DB_PATH.exists():
        return get_db_summary(DB_PATH)
    return {}


@st.cache_resource
def get_predictor():
    pipeline_path = MODELS_DIR / "funding_model_pipeline.joblib"
    if pipeline_path.exists():
        return FundingPredictor(pipeline_path)
    return None


@st.cache_data
def load_model_metrics():
    metrics_path = MODELS_DIR / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def sidebar():
    st.sidebar.title("Startup Intelligence AI")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Historical Data:** Jan 2015 – Jan 2020")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "> **Disclaimer:** This dashboard analyzes historical data. "
        "It does not predict future startup success or guarantee funding outcomes."
    )
    return page


def page_overview(df, quality_report, db_summary):
    st.title("Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", f"{len(df):,}")
    col2.metric("Valid Amounts", f"{df['has_valid_amount'].sum():,}")
    col3.metric("Valid Dates", f"{df['has_valid_date'].sum():,}")
    total_funding = df["amount_clean"].sum()
    col4.metric("Total Recorded Funding", f"${total_funding / 1e9:.2f}B")

    st.markdown("---")
    st.subheader("Dataset Coverage")

    if "date_parsed" in df.columns:
        valid_dates = df["date_parsed"].dropna()
        if len(valid_dates) > 0:
            parsed_dates = pd.to_datetime(valid_dates, format="mixed", dayfirst=True, errors="coerce")
            parsed_dates = parsed_dates.dropna()
            if len(parsed_dates) > 0:
                min_date = parsed_dates.min()
                max_date = parsed_dates.max()
                st.info(f"**Date Range:** {min_date.strftime('%B %Y')} – {max_date.strftime('%B %Y')}")

    st.markdown("---")
    st.subheader("Data Quality Indicators")

    if quality_report:
        cols = st.columns(3)
        missing_amounts = quality_report.get("missing_values", {}).get("amount_in_usd", {})
        cols[0].metric("Missing Amounts", f"{missing_amounts.get('percentage', 0)}%")
        missing_dates = quality_report.get("missing_values", {}).get("date_dd_mm_yyyy", {})
        cols[1].metric("Missing Dates", f"{missing_dates.get('percentage', 0)}%")
        cols[2].metric("Duplicate Rows", quality_report.get("duplicated_rows", 0))

    st.markdown("---")
    st.subheader("Key Findings")

    if "industry_clean" in df.columns:
        top_industry = df.groupby("industry_clean")["amount_clean"].sum().idxmax()
        st.write(f"- **Top Industry by Recorded Funding:** {top_industry}")

    if "city_clean" in df.columns:
        top_city = df.groupby("city_clean")["amount_clean"].sum().idxmax()
        st.write(f"- **Top City by Recorded Funding:** {top_city}")

    if "investment_type_clean" in df.columns:
        top_type = df["investment_type_clean"].value_counts().idxmax()
        st.write(f"- **Most Common Funding Type:** {top_type}")

    if "investors_name" in df.columns:
        top_investor = df["investors_name"].value_counts().idxmax()
        st.write(f"- **Most Frequent Investor:** {top_investor}")

    st.markdown("---")
    st.subheader("Funding at a Glance")

    valid_amounts = df["amount_clean"].dropna()
    m1, m2, m3, m4 = st.columns(4)
    if len(valid_amounts) > 0:
        m1.metric("Median Funding", f"${valid_amounts.median():,.0f}")
        m2.metric("Mean Funding", f"${valid_amounts.mean():,.0f}")
    if "startup_name" in df.columns:
        m3.metric("Distinct Startups", f"{df['startup_name'].nunique():,}")
    if "city_clean" in df.columns:
        m4.metric("Distinct Cities", f"{df['city_clean'].nunique():,}")

    st.markdown("---")
    st.subheader("Top Industries & Cities")

    c1, c2 = st.columns(2)
    with c1:
        if "industry_clean" in df.columns:
            top_ind = df.groupby("industry_clean")["amount_clean"].sum().nlargest(5).reset_index()
            fig = px.bar(top_ind, x="amount_clean", y="industry_clean", orientation="h",
                         title="Top 5 Industries by Recorded Funding",
                         labels={"amount_clean": "Total Funding (USD)", "industry_clean": "Industry"})
            fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "city_clean" in df.columns:
            top_city = df["city_clean"].value_counts().nlargest(5).reset_index()
            top_city.columns = ["City", "Events"]
            fig = px.bar(top_city, x="Events", y="City", orientation="h",
                         title="Top 5 Cities by Funding Events")
            fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Activity Over Time & Funding Mix")

    c3, c4 = st.columns(2)
    with c3:
        if "year" in df.columns:
            yearly = df.groupby("year").size().reset_index(name="events")
            fig = px.bar(yearly, x="year", y="events", title="Funding Events per Year",
                         labels={"year": "Year", "events": "Events"})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        if "investment_type_clean" in df.columns:
            top_types = df["investment_type_clean"].value_counts().nlargest(6).reset_index()
            top_types.columns = ["Funding Type", "Events"]
            fig = px.pie(top_types, names="Funding Type", values="Events",
                         title="Top Funding Types by Event Share")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.warning(
        "**Historical Data Notice:** This dataset covers January 2015 to January 2020. "
        "It does not represent current market conditions."
    )


def page_funding_trends(df):
    st.title("Funding Trends")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        years = sorted(df["year"].dropna().unique())
        selected_years = st.multiselect("Year", years, default=years)
    with col2:
        industries = sorted(df["industry_clean"].unique())
        selected_industries = st.multiselect("Industry", industries, default=industries[:5])
    with col3:
        cities = sorted(df["city_clean"].unique())
        selected_cities = st.multiselect("City", cities, default=cities[:5])

    filtered = df.copy()
    if selected_years:
        filtered = filtered[filtered["year"].isin(selected_years)]
    if selected_industries:
        filtered = filtered[filtered["industry_clean"].isin(selected_industries)]
    if selected_cities:
        filtered = filtered[filtered["city_clean"].isin(selected_cities)]

    st.markdown("---")
    st.subheader("Funding Events Over Time")

    if "year" in filtered.columns and len(filtered) > 0:
        yearly = filtered.groupby("year").agg(
            event_count=("sr_no", "count"),
            total_funding=("amount_clean", "sum"),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=yearly["year"], y=yearly["event_count"], name="Event Count"))
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["total_funding"] / 1e9,
            name="Total Funding (B USD)", yaxis="y2", mode="lines+markers",
        ))
        fig.update_layout(
            title="Annual Funding Events and Total Recorded Funding",
            xaxis_title="Year",
            yaxis_title="Event Count",
            yaxis2=dict(title="Total Funding (B USD)", overlaying="y", side="right"),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Monthly Trends")

    if "year" in filtered.columns and "month" in filtered.columns:
        monthly = filtered.groupby(["year", "month"]).agg(
            event_count=("sr_no", "count"),
        ).reset_index()
        monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))

        fig = px.line(monthly, x="date", y="event_count", title="Monthly Funding Event Count")
        fig.update_layout(xaxis_title="Date", yaxis_title="Event Count", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Funding Amount Distribution")

    if "amount_clean" in filtered.columns:
        valid_amounts = filtered["amount_clean"].dropna()
        if len(valid_amounts) > 0:
            fig = px.histogram(valid_amounts, nbins=50, title="Distribution of Funding Amounts (USD)")
            fig.update_layout(xaxis_title="Amount (USD)", yaxis_title="Frequency", height=400)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Median", f"${valid_amounts.median():,.0f}")
            col2.metric("Mean", f"${valid_amounts.mean():,.0f}")
            col3.metric("Total", f"${valid_amounts.sum() / 1e9:.2f}B")


def page_industry_insights(df):
    st.title("Industry Insights")
    st.markdown("---")

    if "industry_clean" not in df.columns:
        st.warning("Industry data not available.")
        return

    industry_stats = df.groupby("industry_clean").agg(
        event_count=("sr_no", "count"),
        total_funding=("amount_clean", "sum"),
        avg_funding=("amount_clean", "mean"),
        median_funding=("amount_clean", "median"),
    ).reset_index().sort_values("total_funding", ascending=False)

    st.subheader("Industry Rankings by Total Recorded Funding")
    st.dataframe(industry_stats, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 10 Industries by Total Funding")

    top10 = industry_stats.head(10)
    fig = px.bar(
        top10, x="industry_clean", y="total_funding",
        title="Top 10 Industries by Total Recorded Funding (USD)",
        labels={"total_funding": "Total Funding (USD)", "industry_clean": "Industry"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Industry Event Counts")

    fig = px.bar(
        industry_stats.sort_values("event_count", ascending=False).head(15),
        x="industry_clean", y="event_count",
        title="Top 15 Industries by Event Count",
        labels={"event_count": "Event Count", "industry_clean": "Industry"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Missing/Unknown Categories")

    unknown_count = (df["industry_clean"] == "Unknown").sum()
    st.write(f"**Unknown Industry:** {unknown_count} events ({unknown_count / len(df) * 100:.1f}%)")


def page_city_insights(df):
    st.title("City Insights")
    st.markdown("---")

    if "city_clean" not in df.columns:
        st.warning("City data not available.")
        return

    city_stats = df.groupby("city_clean").agg(
        event_count=("sr_no", "count"),
        total_funding=("amount_clean", "sum"),
        avg_funding=("amount_clean", "mean"),
        median_funding=("amount_clean", "median"),
    ).reset_index().sort_values("total_funding", ascending=False)

    st.subheader("City Rankings by Total Recorded Funding")
    st.dataframe(city_stats, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 10 Cities by Total Funding")

    top10 = city_stats.head(10)
    fig = px.bar(
        top10, x="city_clean", y="total_funding",
        title="Top 10 Cities by Total Recorded Funding (USD)",
        labels={"total_funding": "Total Funding (USD)", "city_clean": "City"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("City Event Counts")

    fig = px.bar(
        city_stats.sort_values("event_count", ascending=False).head(15),
        x="city_clean", y="event_count",
        title="Top 15 Cities by Event Count",
        labels={"event_count": "Event Count", "city_clean": "City"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("City Normalization")

    st.markdown(
        "The following city name variants have been normalized:\n"
        "- Bangalore → Bengaluru\n"
        "- Gurgaon → Gurugram\n"
        "- New Delhi / Delhi → New Delhi\n"
        "- Cochin → Kochi"
    )


def page_investor_insights(df):
    st.title("Investor Insights")
    st.markdown("---")

    if "investors_name" not in df.columns:
        st.warning("Investor data not available.")
        return

    investor_counts = df["investors_name"].value_counts().head(20).reset_index()
    investor_counts.columns = ["Investor", "Appearance Count"]

    st.subheader("Top 20 Most Frequently Recorded Investors")
    st.dataframe(investor_counts, use_container_width=True)

    st.markdown("---")

    fig = px.bar(
        investor_counts.head(10), x="Investor", y="Appearance Count",
        title="Top 10 Investors by Recorded Appearance Frequency",
        labels={"Appearance Count": "Number of Events"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.info(
        "**Note:** Investor frequency reflects how often an investor appears in the dataset. "
        "It does not indicate investor quality, returns, or future performance."
    )


def page_machine_learning(df):
    st.title("Machine Learning")
    st.markdown("---")

    metrics = load_model_metrics()

    if not metrics:
        st.warning("No trained model found. Please run the training pipeline first.")
        return

    st.subheader("Model Comparison")

    model_names = [k for k in metrics.keys() if k not in ("best_model", "selection_metric")]
    maes = [metrics[n]["mae"] for n in model_names]
    rmses = [metrics[n]["rmse"] for n in model_names]
    r2s = [metrics[n]["r2"] for n in model_names]
    rmsles = [metrics[n].get("rmsle", float("nan")) for n in model_names]

    comparison_df = pd.DataFrame({
        "Model": model_names,
        "CV RMSLE": [metrics[n].get("cv_rmsle") for n in model_names],
        "Test RMSLE": rmsles,
        "MAE (USD)": maes,
        "RMSE (USD)": rmses,
        "R²": r2s,
    })
    st.dataframe(comparison_df, use_container_width=True)
    st.caption(
        "RMSLE (root mean squared log error) is the selection metric. The target is "
        "heavily right-skewed — the top 1% of rounds hold ~44% of all funding — so "
        "raw-dollar metrics are dominated by a few outliers while RMSLE measures "
        "relative error. Lower is better."
    )

    st.markdown("---")
    st.subheader("Evaluation Metrics")

    best = metrics.get("best_model", "N/A")
    st.write(f"**Best Model (by inner CV RMSLE):** {best}")
    st.caption(
        "The winner is chosen by inner TimeSeriesCV score on the training period, "
        "never by test performance — selecting on the test set would invalidate "
        "the reported test number."
    )

    col1, col2, col3, col4 = st.columns(4)
    if best in metrics:
        cv = metrics[best].get("cv_rmsle")
        col1.metric("CV RMSLE", f"{cv:.4f}" if cv is not None else "n/a")
        col2.metric("Test RMSLE", f"{metrics[best]['rmsle']:.4f}")
        col3.metric("MAE", f"${metrics[best]['mae']:,.0f}")
        col4.metric("R²", f"{metrics[best]['r2']:.4f}")

    if metrics[best].get("best_params"):
        st.write("**Tuned hyperparameters:**")
        st.json(metrics[best]["best_params"])

    st.info(
        "Negative R² is expected on this dataset. Round size depends on traction, "
        "investor syndicate and timing, none of which are present here. R² stays "
        "negative even for a perfect relative-error model because a handful of "
        "billion-dollar rounds dominate the squared-error term."
    )

    st.markdown("---")
    st.subheader("Model Performance Visualization")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="MAE", x=model_names, y=maes, marker_color="steelblue"))
    fig.add_trace(go.Bar(name="RMSE", x=model_names, y=rmses, marker_color="coral"))
    fig.update_layout(
        title="Model Comparison: MAE and RMSE",
        xaxis_title="Model",
        yaxis_title="Error (USD)",
        barmode="group",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Test RMSLE", x=model_names, y=rmsles, marker_color="seagreen"))
    fig2.update_layout(
        title="Model Comparison: Test RMSLE (lower is better)",
        xaxis_title="Model",
        yaxis_title="RMSLE",
        height=450,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Funding Range Classification")

    clf_metrics_path = MODELS_DIR / "classification_metrics.json"
    if clf_metrics_path.exists():
        with open(clf_metrics_path) as f:
            clf_metrics = json.load(f)

        clf_names = [
            k for k in clf_metrics
            if k not in ("best_model", "thresholds", "labels", "selection_metric")
        ]

        clf_df = pd.DataFrame({
            "Model": clf_names,
            "CV F1": [clf_metrics[n].get("cv_f1") for n in clf_names],
            "Test F1": [clf_metrics[n]["f1_weighted"] for n in clf_names],
        })
        st.dataframe(clf_df, use_container_width=True)

        best_clf = clf_metrics.get("best_model", "N/A")
        st.write(f"**Best Classification Model (by inner CV F1):** {best_clf}")
        st.caption(
            "Categories are Low / Medium / High, with thresholds derived from the "
            "training period only and applied unchanged to the test period."
        )
        if best_clf in clf_metrics and clf_metrics[best_clf].get("best_params"):
            st.write("**Tuned hyperparameters:**")
            st.json(clf_metrics[best_clf]["best_params"])

        if "thresholds" in clf_metrics:
            st.write(f"**Category Thresholds (USD):** {clf_metrics['thresholds']}")
        if "labels" in clf_metrics:
            st.write(f"**Categories:** {clf_metrics['labels']}")
    else:
        st.info("Classification model not trained yet.")

    st.markdown("---")
    st.subheader("Training Details")

    st.write("- **Target:** Funding amount (USD)")
    st.write("- **Features:** Industry, Subvertical, City, Investment Type, Year, Month")
    st.write("- **Validation:** Time-based split (80% train, 20% test)")
    st.write("- **Preprocessing:** One-hot encoding for categoricals, median imputation for numerics")
    st.write("- **Best model uses log1p transformation for target variable**")

    st.markdown("---")
    st.warning(
        "Model predictions are exploratory estimates based on historical patterns. "
        "They do not guarantee future funding outcomes."
    )


@st.cache_data
def load_cluster_data():
    return load_cluster_summary()


def page_clustering():
    st.title("Clustering")
    st.markdown("---")

    st.markdown(
        "K-means groups funding events by sector, city, investment type, "
        "`log1p(amount)` and year. The cluster count is chosen by silhouette "
        "score rather than hard-coded."
    )

    summary = load_cluster_data()

    if not summary:
        st.warning("No clustering results found. Run `python run_full.py` first.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters (k)", summary["optimal_k"])
    col2.metric("Silhouette Score", f"{summary['silhouette']:.4f}")
    col3.metric("Events Clustered", f"{summary['events_clustered']:,}")

    st.markdown("---")
    st.subheader("Cluster Profiles")

    clusters_df = pd.DataFrame(summary["clusters"])
    display = clusters_df.rename(columns={
        "cluster": "Cluster",
        "event_count": "Events",
        "median_amount": "Median Amount (USD)",
        "mean_amount": "Mean Amount (USD)",
        "total_amount": "Total (USD)",
        "avg_year": "Avg Year",
        "top_industry": "Top Industry",
        "top_city": "Top City",
    })
    st.dataframe(display, use_container_width=True)

    st.markdown("---")
    st.subheader("k Selection")

    sweep_df = pd.DataFrame(summary["k_sweep"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep_df["k"], y=sweep_df["silhouette"], mode="lines+markers",
        name="Silhouette", marker_color="coral",
    ))
    fig.add_trace(go.Scatter(
        x=sweep_df["k"], y=sweep_df["inertia"], mode="lines+markers",
        name="Inertia", marker_color="steelblue", yaxis="y2",
    ))
    fig.update_layout(
        title="Elbow and Silhouette by Cluster Count",
        xaxis_title="Number of Clusters (k)",
        yaxis=dict(title="Silhouette Score"),
        yaxis2=dict(title="Inertia", overlaying="y", side="right"),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Cluster Composition")

    chart = clusters_df.sort_values("median_amount")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=chart["cluster"].astype(str), y=chart["median_amount"],
        name="Median Amount (USD)", marker_color="coral",
    ))
    fig2.update_layout(
        title="Median Funding Amount by Cluster",
        xaxis_title="Cluster", yaxis_title="Median Amount (USD)", height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=chart["cluster"].astype(str), y=chart["event_count"],
        name="Events", marker_color="steelblue",
    ))
    fig3.update_layout(
        title="Event Count by Cluster",
        xaxis_title="Cluster", yaxis_title="Events", height=400,
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "Clustering groups similar historical events; it does not segment "
        "companies and carries no forward-looking guarantee."
    )


def page_funding_estimator(df):
    st.title("Funding Estimator")
    st.markdown("---")

    predictor = get_predictor()

    if predictor is None:
        st.warning("No trained model found. Please run the training pipeline first.")
        return

    st.subheader("Input Features")

    col1, col2 = st.columns(2)

    with col1:
        industries = sorted(df["industry_clean"].unique())
        industry = st.selectbox("Industry", industries)

        cities = sorted(df["city_clean"].unique())
        city = st.selectbox("City", cities)

    with col2:
        investment_types = sorted(df["investment_type_clean"].unique())
        investment_type = st.selectbox("Investment Type", investment_types)

        year = st.slider("Year", MIN_PREDICTABLE_YEAR, MAX_PREDICTABLE_YEAR, 2020)
        month = st.slider("Month", 1, 12, 6)

    subvertical = st.text_input("Subvertical (optional)", "")
    investors = st.text_input("Investors (optional, comma-separated)", "")

    st.markdown("---")

    if st.button("Estimate Funding Amount"):
        try:
            result = predictor.predict(
                industry, subvertical, city, investment_type, year, month, investors
            )

            st.subheader("Estimated Funding Amount")

            col1, col2 = st.columns(2)
            col1.metric("Estimated Amount (USD)", f"${result['estimated_amount_usd']:,.0f}")
            col2.metric("Estimated Amount (INR Crores)", f"₹{result['estimated_amount_inr_crores']:.2f} Cr")

            st.info(result["disclaimer"])

        except ValueError as e:
            st.error(f"Invalid input: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.subheader("Model Limitations")

    st.markdown(
        "- This model estimates funding amounts based on historical patterns from 2015-2020.\n"
        "- It does not predict whether a startup will receive funding.\n"
        "- It does not predict investment returns or startup success.\n"
        "- Estimates are associations learned from data, not causal relationships.\n"
        "- The model may not generalize to current market conditions."
    )


def page_explainability(df):
    st.title("Explainable AI")
    st.markdown("---")

    st.subheader("Feature Importance")

    st.markdown(
        "Permutation importance measures how much the model's error increases "
        "when a feature is randomly shuffled. Higher values indicate more important features."
    )

    importance_path = PROCESSED_DIR / "permutation_importance.csv"
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)
        st.dataframe(importance_df, use_container_width=True)

        fig = px.bar(
            importance_df.head(15), x="importance_mean", y="feature",
            orientation="h",
            title="Top 15 Feature Importance (Permutation)",
            labels={"importance_mean": "Importance (MAE increase)", "feature": "Feature"},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Permutation importance not computed yet. Run the training pipeline to generate it.")

    st.markdown("---")
    st.subheader("SHAP Analysis")

    shap_candidates = [REPORTS_DIR / "shap_summary.png", PROCESSED_DIR / "shap_summary.png"]
    shap_path = next((p for p in shap_candidates if p.exists()), None)
    if shap_path is not None:
        st.image(str(shap_path), use_container_width=True,
                 caption="SHAP summary: feature impact on funding estimates (HistGradientBoosting)")
    else:
        st.info("SHAP analysis not available. Run the training pipeline to generate it.")

    st.markdown("---")
    st.info(
        "**Important:** Feature importance describes associations learned by the model. "
        "It does not prove that a feature causes funding outcomes."
    )


def page_methodology(df):
    st.title("Methodology")
    st.markdown("---")

    st.subheader("Data Cleaning")

    st.markdown(
        "- Original CSV preserved without modification\n"
        "- Column names normalized to snake_case\n"
        "- Dates parsed using day-first format (dd/mm/yyyy)\n"
        "- Funding amounts parsed with Indian-style comma separator support\n"
        "- 'Undisclosed', 'unknown', 'N/A' treated as missing (not zero)\n"
        "- City names normalized (Bangalore→Bengaluru, Gurgaon→Gurugram)\n"
        "- Industry and investment type labels standardized\n"
        "- Duplicate rows flagged but preserved"
    )

    st.markdown("---")
    st.subheader("Dataset Coverage")

    st.markdown(
        "- **Source:** Indian startup funding events\n"
        "- **Period:** January 2015 – January 2020\n"
        "- **Rows:** {:,}\n"
        "- **Columns:** 10".format(len(df))
    )

    st.markdown("---")
    st.subheader("Modeling Assumptions")

    st.markdown(
        "- **Prediction scenario:** Estimate funding amount for a new funding event\n"
        "- **Features used:** Industry, Subvertical, City, Investment Type, Year, Month\n"
        "- **Excluded:** Investor names, Remarks, Serial numbers\n"
        "- **Validation:** Time-based split to prevent data leakage\n"
        "- **Target transformation:** log1p for HistGradientBoosting model\n"
        "- **Inverse transformation:** expm1 to return to original scale"
    )

    st.markdown("---")
    st.subheader("Evaluation Methodology")

    st.markdown(
        "- Time-based train/test split (80/20)\n"
        "- Metrics: MAE, RMSE, R²\n"
        "- Baseline: DummyRegressor (median)\n"
        "- Preprocessing fit on training data only\n"
        "- One-hot encoding with min_frequency=3 to handle rare categories"
    )

    st.markdown("---")
    st.subheader("Known Limitations")

    st.markdown(
        "- Historical data (2015-2020) does not represent current market conditions\n"
        "- Missing funding amounts reduce the sample size for modeling\n"
        "- Investor information excluded from features (not available at prediction time)\n"
        "- Model estimates are not guarantees of funding\n"
        "- Feature importance does not imply causation\n"
        "- Class imbalance may affect classification performance\n"
        "- Outliers in funding amounts may skew model predictions"
    )

    st.markdown("---")
    st.warning(
        "**Historical Data Warning:** This dataset covers January 2015 to January 2020. "
        "It cannot describe current market conditions or guarantee future outcomes."
    )


def main():
    page = sidebar()

    df = load_data()
    quality_report = load_quality_report()
    db_summary = load_db_summary()

    if page == "Overview":
        page_overview(df, quality_report, db_summary)
    elif page == "Funding Trends":
        page_funding_trends(df)
    elif page == "Industry Insights":
        page_industry_insights(df)
    elif page == "City Insights":
        page_city_insights(df)
    elif page == "Investor Insights":
        page_investor_insights(df)
    elif page == "Machine Learning":
        page_machine_learning(df)
    elif page == "Clustering":
        page_clustering()
    elif page == "Funding Estimator":
        page_funding_estimator(df)
    elif page == "Explainable AI":
        page_explainability(df)
    elif page == "Methodology":
        page_methodology(df)


if __name__ == "__main__":
    main()
