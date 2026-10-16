"""Every third-party import must be declared in requirements.txt.

`generate_docs.py` imported reportlab and Pillow without either appearing in
requirements.txt, so the documented install left the PDF build broken on a fresh
clone. Nothing caught it because nothing compared the imports to the manifest.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"

ENTRY_POINTS = [
    ROOT / "run_pipeline.py",
    ROOT / "run_full.py",
    ROOT / "run_scheduled.py",
    ROOT / "app.py",
    ROOT / "generate_docs.py",
    ROOT / "make_video.py",
] + sorted((ROOT / "src").glob("*.py"))

# The suite runs on a fresh clone too, so its imports are dependencies too.
ALL_SOURCES = ENTRY_POINTS + sorted((ROOT / "tests").glob("*.py"))

# Import name -> distribution name, where they differ.
ALIASES = {
    "pil": "pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
}

# First-party modules in this repo, which are not installable dependencies.
LOCAL = {"src", "video_facts", "make_video", "generate_docs", "app", "__pycache__"}


def declared_packages():
    names = set()
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        names.add(re.split(r"[<>=!~\[; ]", line)[0].strip().lower())
    return names


def top_level_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


class TestRequirementsDeclared:
    def test_requirements_exists(self):
        assert REQUIREMENTS.exists(), "requirements.txt is missing"

    def test_entry_points_exist(self):
        missing = [p.name for p in ENTRY_POINTS if not p.exists()]
        assert not missing, f"declared entry points are missing: {missing}"

    def test_third_party_imports_are_declared(self):
        declared = declared_packages()
        allowed = set(sys.stdlib_module_names) | LOCAL
        undeclared = {}

        for path in ALL_SOURCES:
            if not path.exists():
                continue
            for module in top_level_imports(path):
                key = module.lower()
                if key in allowed or key in declared:
                    continue
                dist = ALIASES.get(key, key)
                if dist in declared:
                    continue
                undeclared.setdefault(f"{module} -> {dist}", []).append(path.name)

        assert not undeclared, (
            "these imports are not in requirements.txt: "
            + ", ".join(f"{k} (in {', '.join(v)})" for k, v in undeclared.items())
        )

    def test_reportlab_and_pillow_are_declared(self):
        """The two that were missing; keep them named explicitly."""
        declared = declared_packages()
        assert "reportlab" in declared, "generate_docs.py imports reportlab"
        assert "pillow" in declared, "generate_docs.py and make_video.py import PIL"

    def test_no_requirements_without_imports(self):
        """Flag declared packages nothing imports, so the list cannot rot."""
        imported = set()
        for path in ALL_SOURCES:
            if path.exists():
                for module in top_level_imports(path):
                    key = module.lower()
                    imported.add(ALIASES.get(key, key))

        stdlib = set(sys.stdlib_module_names) | LOCAL
        unused = sorted(declared_packages() - imported - stdlib)
        # Notebooks legitimately use seaborn/plotly, which entry points do not.
        assert set(unused) <= {"seaborn", "plotly"}, (
            f"declared but not imported by any entry point: {unused}"
        )