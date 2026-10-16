"""Every local path the README links must exist and must not be gitignored.

Two failures look identical from a screenshot - a broken image - and have
different causes:

  1. the file is missing from disk
  2. the file exists but .gitignore excludes it, so a clone never sees it

Case 2 is the one that bit this repo: all eleven charts under "Results &
Screenshots" were real files matched by `reports/*.png`, so they rendered fine
locally and appeared as eleven broken images on GitHub.

The matcher below covers the subset of gitignore syntax this repo actually
uses (comments, directory suffixes, leading slash, globs). It is not a general
gitignore implementation and is not trying to be.
"""

import fnmatch
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
GITIGNORE = ROOT / ".gitignore"

EXTERNAL = re.compile(r"^(https?:|mailto:|ftp:|#)")


def gitignore_patterns():
    if not GITIGNORE.exists():
        return []
    out = []
    for raw in GITIGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def is_ignored(rel_path: str, patterns) -> str | None:
    """Return the matching pattern, or None if the path is tracked."""
    rel = rel_path.lstrip("/")
    for pattern in patterns:
        p = pattern.rstrip("/")
        if pattern.endswith("/") and not rel.endswith("/"):
            # A directory pattern only matches a directory, or something under it.
            if not rel.startswith(p + "/"):
                continue
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel + "/", pattern):
            return pattern
    return None


def readme_references():
    if not README.exists():
        return []
    text = README.read_text(encoding="utf-8")
    found, seen = [], set()
    for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", text):
        target = target.split("#")[0].strip()
        if not target or EXTERNAL.match(target) or target in seen:
            continue
        seen.add(target)
        found.append(target)
    return found


class TestReadmeLinks:
    def test_readme_exists(self):
        assert README.exists(), "README.md is missing"

    def test_readme_has_references(self):
        assert readme_references(), "no local references found; the parser is broken"

    def test_referenced_paths_exist(self):
        missing = [r for r in readme_references() if not (ROOT / r).exists()]
        assert not missing, f"README links to paths that do not exist: {missing}"

    def test_referenced_paths_are_not_gitignored(self):
        patterns = gitignore_patterns()
        hidden = [
            (r, is_ignored(r, patterns))
            for r in readme_references()
            if is_ignored(r, patterns)
        ]
        assert not hidden, (
            "README links to paths that .gitignore excludes, so they render as "
            "broken images or dead links on GitHub: "
            + ", ".join(f"{r} (matched {pat!r})" for r, pat in hidden)
        )

    def test_embedded_images_exist_and_are_pngs(self):
        text = README.read_text(encoding="utf-8")
        images = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text)
        for img in images:
            if EXTERNAL.match(img):
                continue
            path = ROOT / img.split("#")[0]
            assert path.exists(), f"embedded image missing: {img}"
            assert path.suffix.lower() == ".png", f"unexpected image format: {img}"

    def test_embedded_images_are_not_empty(self):
        """A zero-byte or stub PNG renders as a broken image too."""
        text = README.read_text(encoding="utf-8")
        for img in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text):
            if EXTERNAL.match(img):
                continue
            path = ROOT / img.split("#")[0]
            if path.exists():
                assert path.stat().st_size > 1_000, f"suspiciously small image: {img}"

    @pytest.mark.parametrize(
        "referenced", ["reports/funding_by_year.png", "reports/model_comparison.png"]
    )
    def test_known_chart_regressions(self, referenced):
        """The two charts from the original bug report, pinned explicitly."""
        path = ROOT / referenced
        assert path.exists(), f"{referenced} missing"
        assert is_ignored(referenced, gitignore_patterns()) is None, (
            f"{referenced} is gitignored again; the README screenshot breaks"
        )