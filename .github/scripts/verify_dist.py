"""Validate source identity, release tags and both distribution archives."""
from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
from email.policy import default
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import zipfile

if not __debug__:
    raise RuntimeError("Release checks require assertions enabled; unset PYTHONOPTIMIZE.")

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def source_version(root: Path) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "scifigbench"
    assignments = ast.parse((root / "src/scifigbench/_version.py").read_text(encoding="utf-8"))
    versions = [ast.literal_eval(node.value) for node in assignments.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)]
    assert versions == [project["version"]], "Source versions differ"
    assert re.fullmatch(r"\d+\.\d+\.\d+", project["version"]), "Expected a stable version"
    return project["version"]


def validate_release(root: Path, version: str, commit: str, event: dict) -> None:
    release = event.get("release", {})
    assert not release.get("draft") and not release.get("prerelease"), "Only a final published Release may publish"
    tag = release.get("tag_name")
    assert tag == "v" + version, "Tag and package version differ"
    assert git(root, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}") == commit, "Tag points to a different source Commit"


def metadata_check(data: bytes, version: str, readme: str):
    metadata = BytesParser(policy=default).parsebytes(data)
    assert metadata["Name"] == "scifigbench" and metadata["Version"] == version
    assert metadata["License-Expression"] == "MIT", "Missing MIT License metadata"
    assert metadata["Description-Content-Type"].startswith("text/markdown")
    description = data.replace(b"\r\n", b"\n").partition(b"\n\n")[2].decode("utf-8")
    assert isinstance(description, str) and description.rstrip("\n") == readme.rstrip("\n"), "Long description differs from source README"
    return metadata


def verify(root: Path, dist: Path, *, allow_dirty: bool = False) -> dict:
    version = source_version(root)
    commit = git(root, "rev-parse", "HEAD")
    dirty = bool(git(root, "status", "--porcelain"))
    assert allow_dirty or not dirty, "Release builds require a clean source Commit"
    if not dirty:
        names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode("utf-8").split("\0")
        for name in filter(None, names):
            blob = subprocess.check_output(["git", "-C", str(root), "show", "HEAD:" + name])
            assert (root / name).read_bytes() == blob, f"Checkout bytes differ from the Git object: {name}"
    expected_commit = os.environ.get("GITHUB_SHA")
    if expected_commit:
        assert commit == expected_commit, "Checkout differs from the Workflow Commit"
    if os.environ.get("GITHUB_EVENT_NAME") == "release":
        assert os.environ.get("GITHUB_REPOSITORY") == "LeNcyer/SciFigBench"
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        assert event.get("action") == "published"
        validate_release(root, version, commit, event)
        assert not dirty, "Release source is dirty"
    wheels, sdists = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "Expected exactly one wheel and one sdist"
    assert sorted(p.name for p in dist.iterdir()) == sorted([wheels[0].name, sdists[0].name]), "Unexpected distribution files"
    readme = (root / "README.md").read_text(encoding="utf-8")
    license_bytes = (root / "LICENSE").read_bytes()
    package = root / "src/scifigbench"
    source = {p.relative_to(package).as_posix(): p.read_bytes() for p in package.rglob("*.py")}
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        metadata_check(wheel.read(meta_name), version, readme)
        licenses = [n for n in names if n.endswith(".dist-info/licenses/LICENSE")]
        assert len(licenses) == 1 and wheel.read(licenses[0]) == license_bytes
        runtime = {n.removeprefix("scifigbench/"): wheel.read(n) for n in names if n.startswith("scifigbench/") and n.endswith(".py")}
        assert runtime == source, "Wheel source bytes differ"
        assert not any(n.endswith((".pyc", ".pyo")) for n in names)
    with tarfile.open(sdists[0]) as sdist:
        members = {m.name: m for m in sdist.getmembers() if m.isfile()}
        prefix = f"scifigbench-{version}/"
        def read(name):
            return sdist.extractfile(members[prefix + name]).read()
        metadata_check(read("PKG-INFO"), version, readme)
        assert read("LICENSE") == license_bytes
        runtime = {n.removeprefix(prefix + "src/scifigbench/"): sdist.extractfile(m).read()
                   for n, m in members.items() if n.startswith(prefix + "src/scifigbench/") and n.endswith(".py")}
        assert runtime == source, "sdist source bytes differ"
        assert read("tests/runner_support.py") == (root / "tests/runner_support.py").read_bytes()
    return {
        "schema_version": 1, "package": "scifigbench", "version": version,
        "source_commit": commit, "source_tree": git(root, "rev-parse", "HEAD^{tree}"),
        "source_dirty": dirty,
        "source_matches_git_objects": not dirty, "python": sys.version.split()[0],
        "runtime_sha256": {name: sha256(data) for name, data in sorted(source.items())},
        "artifacts": {p.name: sha256(p.read_bytes()) for p in (wheels[0], sdists[0])},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true", help="Local development only; never valid for publishing")
    args = parser.parse_args()
    report = verify(Path(__file__).resolve().parents[2], args.dist.resolve(), allow_dirty=args.allow_dirty)
    args.provenance.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
