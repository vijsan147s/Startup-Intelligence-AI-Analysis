"""Tests for the unified pipeline entry point."""

import inspect

import pytest

from src import pipeline as pipeline_module


def test_run_is_callable_with_expected_signature():
    sig = inspect.signature(pipeline_module.run)
    assert callable(pipeline_module.run)
    for flag in ["skip_shap", "skip_classification", "skip_clustering", "verbose"]:
        assert flag in sig.parameters
        assert sig.parameters[flag].default is False or flag == "verbose"


def test_permutation_output_path_is_data_processed():
    assert pipeline_module.PERMUTATION_PATH.startswith("data/processed/")


def test_entry_points_delegate_to_same_implementation():
    """run_pipeline.py and run_full.py must not reimplement pipeline logic."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for script in ["run_pipeline.py", "run_full.py"]:
        source = (root / script).read_text(encoding="utf-8")
        assert "from src.pipeline import run" in source
        # Regression of the duplication this module replaced.
        assert "run_training(" not in source
        assert "create_database(" not in source
        assert "preprocess(" not in source


def test_pipeline_module_has_no_duplicated_step_logic():
    """The steps must live in src/pipeline.py, not be copy-pasted."""
    source = inspect.getsource(pipeline_module)
    assert source.count("=== Step 1: Data Cleaning ===") == 1