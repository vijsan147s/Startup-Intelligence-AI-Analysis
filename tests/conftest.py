import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

# Generated artifacts that the pipeline owns. Tests must never write here.
ARTIFACT_DIRS = [ROOT / "models", ROOT / "data" / "processed"]
ARTIFACT_SUFFIXES = {".json", ".csv", ".joblib", ".db"}


def _snapshot():
    out = {}
    for directory in ARTIFACT_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in ARTIFACT_SUFFIXES:
                out[str(path.relative_to(ROOT))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return out


@pytest.fixture(scope="session", autouse=True)
def _redirect_pipeline_output(tmp_path_factory):
    """Point the pipeline's default output dirs at a throwaway directory.

    This is the preventive half. `save_pipeline` and `save_classification_pipeline`
    fall back to `MODELS_DIR` when no `output_dir` is passed, so a test that
    forgets the argument used to overwrite model_metrics.json with numbers from a
    toy fixture. Redirecting the module-level binding means such a test now
    writes to scratch and passes, instead of silently corrupting artifacts.

    Only `src.train` and `src.classification` are redirected; both bind
    `MODELS_DIR` at import time, so patching `src.config` alone would not
    reach them.
    """
    import src.classification
    import src.train

    scratch = tmp_path_factory.mktemp("artifact_scratch")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(src.train, "MODELS_DIR", scratch)
        mp.setattr(src.classification, "MODELS_DIR", scratch)
        yield


@pytest.fixture(scope="session", autouse=True)
def _artifacts_untouched():
    """Fail the session if any test wrote to models/ or data/processed/.

    The backstop to `_redirect_pipeline_output`: that one keeps the two known
    writers away, this one catches anything else that reaches those directories
    by another route. The README, the generated PDF and the rendered video all
    read these files, so a stray write is invisible until someone compares them.
    """
    before = _snapshot()
    yield
    after = _snapshot()

    changed = sorted(
        name
        for name in set(before) | set(after)
        if before.get(name) != after.get(name)
    )
    assert not changed, (
        "the test suite modified generated artifacts: "
        + ", ".join(changed)
        + ". Tests must pass output_dir=tmp_path."
    )
