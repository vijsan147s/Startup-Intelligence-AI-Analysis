"""K-means clustering of funding events, with reproducible artifacts.

Promoted out of ``notebooks/04_clustering.ipynb`` so the result is scriptable,
testable and repeatable rather than living only in notebook state.

Two deliberate choices differ from the original notebook:

* Categorical inputs are one-hot encoded rather than label encoded. Label
  encoding puts an arbitrary integer ordering on "Bengaluru" vs "Mumbai" and
  K-means will happily treat that ordering as distance.
* Funding amount is transformed with ``log1p`` before clustering. Raw amounts
  have skew ~22, so unscaled dollars would collapse nearly every point onto a
  handful of mega-rounds and make the clusters a pure amount tiering.
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CLUSTERS_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR

logger = logging.getLogger(__name__)

CATEGORICAL_CLUSTER_FEATURES = ["industry_clean", "city_clean", "investment_type_clean"]
NUMERIC_CLUSTER_FEATURES = ["amount_log", "year"]

K_RANGE = range(2, 11)


def _flag_true(series: pd.Series) -> pd.Series:
    """CSV round-trips booleans as the strings 'True'/'False', and 'False' is
    truthy in Python. Normalise before filtering."""
    return series.astype(str) == "True"


def prepare_clustering_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "has_valid_amount" in df.columns:
        df = df[_flag_true(df["has_valid_amount"])].copy()
    for col in CATEGORICAL_CLUSTER_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    df = df.dropna(subset=["amount_clean", "year"]).copy()
    # KMeans cannot consume NaN.
    df["amount_log"] = np.log1p(df["amount_clean"].astype(float))
    return df


def build_cluster_pipeline(n_clusters: int) -> Pipeline:
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=3)),
    ])
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return Pipeline(steps=[
        ("features", ColumnTransformer(transformers=[
            ("cat", categorical, CATEGORICAL_CLUSTER_FEATURES),
            ("num", numeric, NUMERIC_CLUSTER_FEATURES),
        ])),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)),
    ])


def evaluate_k_range(df: pd.DataFrame, k_range=K_RANGE) -> pd.DataFrame:
    """Elbow and silhouette sweep across candidate cluster counts."""
    rows = []
    for k in k_range:
        pipeline = build_cluster_pipeline(k)
        matrix = pipeline.named_steps["features"].fit_transform(
            df[CATEGORICAL_CLUSTER_FEATURES + ["amount_log", "year"]]
        )
        labels = pipeline.named_steps["kmeans"].fit_predict(matrix)
        rows.append({
            "k": k,
            "inertia": round(float(pipeline.named_steps["kmeans"].inertia_), 2),
            "silhouette": round(float(silhouette_score(matrix, labels)), 4),
        })
        logger.info("k=%d inertia=%.2f silhouette=%.4f", k, rows[-1]["inertia"], rows[-1]["silhouette"])
    return pd.DataFrame(rows)


def plot_k_evaluation(evaluation: pd.DataFrame, output_path=None) -> Path:
    output_path = Path(output_path) if output_path else REPORTS_DIR / "clustering_evaluation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(evaluation["k"], evaluation["inertia"], "bo-")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(evaluation["k"], evaluation["silhouette"], "ro-")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved clustering evaluation to %s", output_path)
    return output_path


def summarize_clusters(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    tagged = df.copy()
    tagged["cluster"] = labels.to_numpy()
    return tagged.groupby("cluster").agg(
        event_count=("amount_clean", "count"),
        median_amount=("amount_clean", "median"),
        mean_amount=("amount_clean", "mean"),
        total_amount=("amount_clean", "sum"),
        avg_year=("year", "mean"),
        top_industry=("industry_clean", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
        top_city=("city_clean", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
    ).reset_index()


def plot_cluster_analysis(summary: pd.DataFrame, output_path=None) -> Path:
    output_path = Path(output_path) if output_path else REPORTS_DIR / "cluster_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    summary.plot(x="cluster", y="event_count", kind="bar", ax=axes[0], color="steelblue", legend=False)
    axes[0].set_title("Event Count by Cluster")

    summary.plot(x="cluster", y="median_amount", kind="bar", ax=axes[1], color="coral", legend=False)
    axes[1].set_title("Median Funding Amount by Cluster")
    axes[1].set_ylabel("Median Amount (USD)")

    summary.plot(x="cluster", y="avg_year", kind="bar", ax=axes[2], color="seagreen", legend=False)
    axes[2].set_title("Average Year by Cluster")
    axes[2].set_ylabel("Year")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved cluster analysis to %s", output_path)
    return output_path


def run_clustering(df: pd.DataFrame, output_dir=None) -> dict:
    """Fit, evaluate, persist. Returns paths plus the chosen k and summary."""
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = prepare_clustering_frame(df)
    logger.info("Clustering over %d funding events", len(frame))

    evaluation = evaluate_k_range(frame)
    optimal_k = int(evaluation.loc[evaluation["silhouette"].idxmax(), "k"])

    pipeline = build_cluster_pipeline(optimal_k)
    features = frame[CATEGORICAL_CLUSTER_FEATURES + ["amount_log", "year"]]
    labels = pipeline.named_steps["kmeans"].fit_predict(
        pipeline.named_steps["features"].fit_transform(features)
    )
    labels = pd.Series(labels, index=frame.index)

    summary = summarize_clusters(frame, labels)

    pipeline_path = output_dir / "clustering_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    summary_path = output_dir / "clustering_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "optimal_k": optimal_k,
                "silhouette": float(evaluation.loc[evaluation["k"] == optimal_k, "silhouette"].iloc[0]),
                "events_clustered": int(len(frame)),
                "features": CATEGORICAL_CLUSTER_FEATURES + NUMERIC_CLUSTER_FEATURES,
                "clusters": json.loads(summary.to_json(orient="records")),
                "k_sweep": json.loads(evaluation.to_json(orient="records")),
            },
            f,
            indent=2,
        )

    assignments_path = CLUSTERS_PATH
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sr_no": frame.get("sr_no"),
        "startup_name": frame.get("startup_name"),
        "industry_clean": frame["industry_clean"],
        "city_clean": frame["city_clean"],
        "amount_clean": frame["amount_clean"],
        "year": frame["year"],
        "cluster": labels.to_numpy(),
    }).to_csv(assignments_path, index=False)

    plot_k_evaluation(evaluation)
    plot_cluster_analysis(summary)

    logger.info("Clustering complete: k=%d, silhouette=%.4f", optimal_k,
                float(evaluation.loc[evaluation["k"] == optimal_k, "silhouette"].iloc[0]))

    return {
        "pipeline_path": str(pipeline_path),
        "summary_path": str(summary_path),
        "assignments_path": str(assignments_path),
        "optimal_k": optimal_k,
        "summary": summary,
        "evaluation": evaluation,
    }


def load_cluster_summary(output_dir=None) -> dict | None:
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    path = output_dir / "clustering_summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)