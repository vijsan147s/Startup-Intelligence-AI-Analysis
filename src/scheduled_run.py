"""Unattended pipeline refresh for Task Scheduler / cron.

Wraps the same two steps a human runs by hand -- ``src.pipeline.run`` and
``generate_docs.main`` -- with the things a scheduled job needs and a console
run does not:

* a lock file, so an overlapping trigger cannot clobber half-written artifacts
* change detection on the raw CSV, so a daily trigger is cheap when nothing moved
* a rotating log plus a machine-readable status file for the dashboard
* a non-zero exit code on failure, which is what Task Scheduler reports on

    python run_scheduled.py                 # refresh only if the CSV changed
    python run_scheduled.py --force         # refresh regardless
    python run_scheduled.py --skip-shap     # faster, SHAP dominates runtime

Keep this module free of interactive input. Anything that prompts will hang a
scheduled task until it times out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import BASE_DIR, RAW_CSV

LOGS_DIR = BASE_DIR / "logs"
LOG_PATH = LOGS_DIR / "scheduled_run.log"
LOCK_PATH = LOGS_DIR / "scheduled_run.lock"
STATUS_PATH = BASE_DIR / "data" / "processed" / "last_run.json"
STATE_PATH = LOGS_DIR / "last_input_hash.txt"

logger = logging.getLogger("scheduled_run")


def setup_logging(verbose: bool = False) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(stream)


def input_fingerprint(path: Path) -> str | None:
    """Content hash of the raw CSV. Hash rather than mtime so a re-download of
    the same bytes does not trigger a full retrain."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_previous_fingerprint() -> str | None:
    if not STATE_PATH.exists():
        return None
    return STATE_PATH.read_text(encoding="utf-8").strip() or None


def acquire_lock():
    """Create the lock file exclusively. O_EXCL is atomic on both NTFS and ext4,
    so two tasks firing in the same second cannot both win."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, f"{os.getpid()} {datetime.now().isoformat(timespec='seconds')}".encode())
    os.close(fd)
    return LOCK_PATH


def release_lock(handle) -> None:
    try:
        handle.unlink()
    except FileNotFoundError:
        pass


def run_docs() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "generate_docs.py")],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    tail = (result.stdout or result.stderr or "").strip().splitlines()
    return result.returncode == 0, tail[-1] if tail else ""


def write_status(status: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unattended pipeline refresh.")
    parser.add_argument("--force", action="store_true",
                        help="Run even if the raw CSV has not changed.")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip the SHAP step, which dominates runtime.")
    parser.add_argument("--skip-docs", action="store_true",
                        help="Skip regenerating the PDF documentation.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    started = time.time()

    lock = acquire_lock()
    if lock is None:
        logger.warning("Another scheduled run holds the lock; exiting without changes.")
        return 0

    fingerprint = input_fingerprint(RAW_CSV)
    if fingerprint is None:
        logger.error("Raw CSV missing at %s", RAW_CSV)
        write_status({"status": "failed", "reason": "raw_csv_missing",
                      "finished_at": datetime.now(timezone.utc).isoformat()})
        release_lock(lock)
        return 1

    if not args.force and fingerprint == read_previous_fingerprint():
        logger.info("Raw CSV unchanged since last run; nothing to do.")
        write_status({"status": "skipped", "reason": "input_unchanged",
                      "input_sha256": fingerprint,
                      "finished_at": datetime.now(timezone.utc).isoformat()})
        release_lock(lock)
        return 0

    from .pipeline import run as run_pipeline

    try:
        logger.info("Starting pipeline (force=%s, skip_shap=%s)", args.force, args.skip_shap)
        result = run_pipeline(skip_shap=args.skip_shap, verbose=args.verbose)
        logger.info("Pipeline finished; best model = %s", result["best_name"])
    except Exception:
        logger.exception("Pipeline failed")
        write_status({"status": "failed", "reason": "pipeline_error",
                      "finished_at": datetime.now(timezone.utc).isoformat()})
        release_lock(lock)
        return 1

    docs_ok, docs_note = True, "skipped"
    if not args.skip_docs:
        docs_ok, docs_note = run_docs()
        logger.info("Docs: %s %s", "ok" if docs_ok else "failed", docs_note)

    STATE_PATH.write_text(fingerprint, encoding="utf-8")
    write_status({
        "status": "ok" if docs_ok else "ok_docs_failed",
        "reason": None,
        "input_sha256": fingerprint,
        "best_model": result["best_name"],
        "classification_best": result.get("classification_best"),
        "shap": bool(result.get("shap_path")),
        "docs": docs_note,
        "duration_seconds": round(time.time() - started, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    logger.info("Run complete in %.1fs", time.time() - started)
    release_lock(lock)
    return 0 if docs_ok else 2


if __name__ == "__main__":
    sys.exit(main())