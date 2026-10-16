"""Tests for the unattended refresh wrapper."""

import json

import pytest

from src import scheduled_run


def test_fingerprint_is_content_based(tmp_path):
    csv = tmp_path / "raw.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    first = scheduled_run.input_fingerprint(csv)
    assert first and len(first) == 64

    # Same bytes touched again must not look like new data.
    assert scheduled_run.input_fingerprint(csv) == first

    csv.write_text("a,b\n1,3\n", encoding="utf-8")
    assert scheduled_run.input_fingerprint(csv) != first


def test_fingerprint_of_missing_file_is_none(tmp_path):
    assert scheduled_run.input_fingerprint(tmp_path / "absent.csv") is None


def test_lock_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduled_run, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(scheduled_run, "LOCK_PATH", tmp_path / "run.lock")

    first = scheduled_run.acquire_lock()
    assert first is not None
    assert scheduled_run.acquire_lock() is None

    scheduled_run.release_lock(first)
    assert scheduled_run.acquire_lock() is not None


def test_unchanged_input_skips_without_touching_pipeline(tmp_path, monkeypatch):
    csv = tmp_path / "raw.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(scheduled_run, "RAW_CSV", csv)
    monkeypatch.setattr(scheduled_run, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(scheduled_run, "LOCK_PATH", tmp_path / "logs" / "run.lock")
    monkeypatch.setattr(scheduled_run, "STATE_PATH", tmp_path / "logs" / "hash.txt")
    monkeypatch.setattr(scheduled_run, "STATUS_PATH", tmp_path / "status.json")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "hash.txt").write_text(
        scheduled_run.input_fingerprint(csv), encoding="utf-8"
    )

    def explode(**kwargs):
        raise AssertionError("pipeline must not run when input is unchanged")

    monkeypatch.setattr(scheduled_run, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr("src.pipeline.run", explode)

    assert scheduled_run.main([]) == 0
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "skipped"
    assert status["reason"] == "input_unchanged"


def test_missing_raw_csv_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduled_run, "RAW_CSV", tmp_path / "absent.csv")
    monkeypatch.setattr(scheduled_run, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(scheduled_run, "LOCK_PATH", tmp_path / "logs" / "run.lock")
    monkeypatch.setattr(scheduled_run, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(scheduled_run, "setup_logging", lambda verbose=False: None)

    assert scheduled_run.main([]) == 1
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["reason"] == "raw_csv_missing"


def test_pipeline_failure_writes_failed_status(tmp_path, monkeypatch):
    csv = tmp_path / "raw.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(scheduled_run, "RAW_CSV", csv)
    monkeypatch.setattr(scheduled_run, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(scheduled_run, "LOCK_PATH", tmp_path / "logs" / "run.lock")
    monkeypatch.setattr(scheduled_run, "STATE_PATH", tmp_path / "logs" / "hash.txt")
    monkeypatch.setattr(scheduled_run, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(scheduled_run, "setup_logging", lambda verbose=False: None)

    def boom(**kwargs):
        raise RuntimeError("training blew up")

    monkeypatch.setattr("src.pipeline.run", boom)

    assert scheduled_run.main([]) == 1
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    # The lock must not survive a crash, or every later run is blocked.
    assert not (tmp_path / "logs" / "run.lock").exists()


def test_wrapper_has_no_interactive_prompts():
    """A prompt in a scheduled task hangs until the execution limit kills it."""
    source = open(scheduled_run.__file__, encoding="utf-8").read()
    for banned in ["input(", "click.", "argparse.ArgumentParser(input"]:
        assert banned not in source