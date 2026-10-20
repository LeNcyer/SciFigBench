"""Explicit, versioned Hub downloads. No online repository is assumed to exist."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
import json, os, re
from .data import DataError, resolve_data_root, load_qa, file_sha256
from .tasks import TASKS

class DownloadError(DataError):
    """Download configuration or integrity verification failed."""

def _checksums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Download is missing its checksum manifest: {path}")
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            raise DownloadError(f"{path}:{number}: invalid SHA256 entry")
        name = fields[1].strip().removeprefix("*")
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name or name in result:
            raise DownloadError(f"{path}:{number}: unsafe or duplicate checksum path {name!r}")
        result[name] = fields[0].lower()
    return result

def verify_download(data_root: str | Path, *, annotations: bool = False, qa_only: bool = False) -> dict:
    """Verify only the requested subset; annotations are optional, never implicit."""
    if annotations and qa_only:
        raise DownloadError("annotations and qa_only are mutually exclusive")
    root = resolve_data_root(data_root)
    manifest_path = root/"QA/manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Download is missing QA/manifest.json: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DownloadError(f"Invalid QA manifest: {manifest_path}: {exc}") from exc
    checks = _checksums(root/"checksums.sha256")
    qa = load_qa(root)
    required = {"QA/manifest.json"} | {"QA/"+spec.filename for spec in TASKS.values()}
    for name, spec in TASKS.items():
        entry = manifest.get("tasks", {}).get(spec.filename)
        actual = sum(q["task"] == name for q in qa)
        if not isinstance(entry, dict) or entry.get("total_questions") != actual or entry.get("sha256") != checks.get("QA/"+spec.filename):
            raise DownloadError(f"QA manifest count/hash mismatch for {spec.filename}")
    if not qa_only:
        required.update(q["image_path"] for q in qa)
    if annotations:
        required.update(f"annotations/{q['sample_id']}.json" for q in qa)
    if (root/"QA/checksums.sha256").is_file():
        required.add("QA/checksums.sha256")
    for rel in sorted(required):
        path = root/rel
        if not path.is_file():
            raise FileNotFoundError(f"Downloaded subset is missing required file: {path}")
        if rel not in checks or file_sha256(path) != checks[rel]:
            raise DownloadError(f"SHA256 integrity check failed: {rel}")
    return {"data_root": str(root), "dataset_version": manifest.get("version"), "questions": len(qa),
            "images": len({q["image_path"] for q in qa}) if not qa_only else 0,
            "checked_files": len(required), "scope": "QA only" if qa_only else "QA + Images + Annotations" if annotations else "QA + Images"}

def download_data(*, repo_id: str | None = None, revision: str | None = None,
                  local_dir: str | Path | None = None, cache_dir: str | Path | None = None,
                  annotations: bool = False, qa_only: bool = False,
                  local_files_only: bool = False) -> Path:
    """Download and verify a dataset subset, or reuse verified local files.

    Supply repo_id and revision (or SCIFIGBENCH_REPO_ID/SCIFIGBENCH_REVISION)
    until published defaults are explicitly configured. Offline local reuse
    does not require a Hub account, repository name, images in QA-only mode,
    or annotations in standard mode. No dataset is downloaded on import.
    """
    if annotations and qa_only:
        raise DownloadError("annotations and qa_only are mutually exclusive")
    root = resolve_data_root(local_dir)
    repo_id = repo_id or os.environ.get("SCIFIGBENCH_REPO_ID")
    revision = revision or os.environ.get("SCIFIGBENCH_REVISION")
    state_path = root/".scifigbench_download.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if local_files_only:
        if state and ((repo_id and state.get("repo_id") != repo_id) or (revision and state.get("revision") != revision)):
            raise DownloadError("Offline cache source/revision differs from the requested dataset")
        verify_download(root, annotations=annotations, qa_only=qa_only)
        return root
    if not repo_id or not revision:
        raise DownloadError("No published dataset defaults are configured. Supply --repo-id and --revision (or SCIFIGBENCH_REPO_ID/SCIFIGBENCH_REVISION), or reuse local data with --local-files-only.")
    if state.get("repo_id") == repo_id and state.get("revision") == revision:
        try:
            verify_download(root, annotations=annotations, qa_only=qa_only)
        except FileNotFoundError:
            pass  # A broader explicitly requested subset may require more files.
        else:
            return root
    from huggingface_hub import snapshot_download
    patterns = ["QA/qa_*.jsonl", "QA/manifest.json", "QA/checksums.sha256", "checksums.sha256"]
    if not qa_only:
        patterns.append("images/*.jpg")
    if annotations:
        patterns.append("annotations/*.json")
    try:
        snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                          local_dir=str(root), cache_dir=str(Path(cache_dir).expanduser()) if cache_dir is not None else None,
                          allow_patterns=patterns, local_files_only=False)
    except Exception as exc:
        raise DownloadError(f"Dataset download failed: {exc}") from exc
    verified = verify_download(root, annotations=annotations, qa_only=qa_only)
    state_path.write_text(json.dumps({"repo_id": repo_id, "revision": revision, "scope": verified["scope"],
                                     "dataset_version": verified["dataset_version"]}, indent=2)+"\n", encoding="utf-8")
    return root
