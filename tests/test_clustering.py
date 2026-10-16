"""Tests for the clustering module."""

import json

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    CATEGORICAL_CLUSTER_FEATURES,
    build_cluster_pipeline,
    evaluate_k_range,
    load_cluster_summary,
    prepare_clustering_frame,
    run_clustering,
    summarize_clusters,
)


def _frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(mean=14.0, sigma=1.5, size=n)
    return pd.DataFrame({
        "industry_clean": rng.choice(["E-Commerce", "FinTech"], n),
        "city_clean": rng.choice(["Bengaluru", "Mumbai"], n),
        "investment_type_clean": rng.choice(["Seed", "Series A"], n),
        "amount_clean": amount,
        "year": rng.choice([2015, 2016, 2017], n),
        "sr_no": np.arange(1, n + 1),
        "startup_name": [f"Startup {i}" for i in range(n)],
    })


class TestPrepareClusteringFrame:
    def test_adds_log_amount(self):
        out = prepare_clustering_frame(_frame())
        assert "amount_log" in out.columns
        assert (out["amount_log"] >= 0).all()

    def test_handles_string_booleans_from_csv(self):
        df = _frame(n=40)
        df["has_valid_amount"] = ["True"] * 20 + ["False"] * 20
        out = prepare_clustering_frame(df)
        # 'False' is truthy as a raw string, so this guards the naive filter.
        assert len(out) == 20

    def test_drops_missing_amounts(self):
        df = _frame(n=40)
        df.loc[df.index[:5], "amount_clean"] = np.nan
        assert len(prepare_clustering_frame(df)) == 35


class TestEvaluateKRange:
    def test_sweep_returns_one_row_per_k(self):
        evaluation = evaluate_k_range(prepare_clustering_frame(_frame()), k_range=range(2, 5))
        assert list(evaluation["k"]) == [2, 3, 4]
        assert (evaluation["inertia"] > 0).all()
        assert evaluation["silhouette"].between(-1, 1).all()

    def test_inertia_decreases_with_more_clusters(self):
        evaluation = evaluate_k_range(prepare_clustering_frame(_frame()), k_range=range(2, 6))
        assert evaluation["inertia"].is_monotonic_decreasing


class TestBuildClusterPipeline:
    def test_pipeline_predicts_labels(self):
        frame = prepare_clustering_frame(_frame())
        pipeline = build_cluster_pipeline(3)
        features = frame[CATEGORICAL_CLUSTER_FEATURES + ["amount_log", "year"]]
        labels = pipeline.named_steps["kmeans"].fit_predict(
            pipeline.named_steps["features"].fit_transform(features)
        )
        assert len(labels) == len(frame)
        assert set(labels) == {0, 1, 2}


class TestSummarizeClusters:
    def test_summary_shape_and_columns(self):
        frame = prepare_clustering_frame(_frame())
        labels = pd.Series(np.arange(len(frame)) % 3, index=frame.index)
        summary = summarize_clusters(frame, labels)
        assert len(summary) == 3
        for col in ["event_count", "median_amount", "top_industry", "top_city"]:
            assert col in summary.columns


class TestRunClustering:
    def test_writes_artifacts(self, tmp_path):
        result = run_clustering(_frame(), output_dir=tmp_path)

        assert (tmp_path / "clustering_pipeline.joblib").exists()
        assert (tmp_path / "clustering_summary.json").exists()

        payload = json.loads((tmp_path / "clustering_summary.json").read_text())
        assert payload["optimal_k"] >= 2
        assert len(payload["clusters"]) == payload["optimal_k"]
        assert len(payload["k_sweep"]) > 0

    def test_summary_round_trips(self, tmp_path):
        run_clustering(_frame(n=80), output_dir=tmp_path)
        loaded = load_cluster_summary(tmp_path)
        assert loaded is not None
        assert "optimal_k" in loaded


def test_load_cluster_summary_missing_returns_none(tmp_path):
    assert load_cluster_summary(tmp_path) is None