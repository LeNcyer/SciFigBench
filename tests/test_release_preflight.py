import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_preflight", ROOT / ".github/scripts/verify_dist.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_package_source_versions_agree():
    from scifigbench import __version__
    assert preflight.source_version(ROOT) == __version__


def test_mismatched_source_version_rejected(tmp_path):
    (tmp_path / "src/scifigbench").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="scifigbench"\nversion="0.1.0"\n', encoding="utf-8")
    (tmp_path / "src/scifigbench/_version.py").write_text('__version__="0.2.0"\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="Source versions differ"):
        preflight.source_version(tmp_path)


def test_release_tag_matches_exact_source_commit(monkeypatch):
    commit = "a" * 40
    monkeypatch.setattr(preflight, "git", lambda *args: commit)
    preflight.validate_release(ROOT, "0.1.0", commit, {"release": {"tag_name": "v0.1.0"}})
    with pytest.raises(AssertionError, match="different source Commit"):
        preflight.validate_release(ROOT, "0.1.0", "b" * 40, {"release": {"tag_name": "v0.1.0"}})


@pytest.mark.parametrize("release", [
    {"tag_name": "v0.2.0"}, {"tag_name": "0.1.0"}, {"tag_name": "v0.1.0", "draft": True},
    {"tag_name": "v0.1.0", "prerelease": True},
])
def test_invalid_release_cannot_pass_preflight(monkeypatch, release):
    monkeypatch.setattr(preflight, "git", lambda *args: "a" * 40)
    with pytest.raises(AssertionError):
        preflight.validate_release(ROOT, "0.1.0", "a" * 40, {"release": release})
