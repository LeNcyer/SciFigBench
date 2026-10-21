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


# The high-level runner uses pinned reference manifests and a sparse image subset.
# The existing download_data API and its full-subset verification remain unchanged.
def _resolve_evaluation_revision(repo_id: str, revision: str, *, cache_dir=None,
                                 local_files_only: bool = False) -> str:
    if not isinstance(repo_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise DownloadError("repo_id must be a Hugging Face owner/repository name")
    if not isinstance(revision, str) or not revision.strip():
        raise DownloadError("revision must be a nonempty branch, tag or full Commit SHA")
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    import hashlib
    from ._run_storage import write_json
    directory = (Path(cache_dir).expanduser() / ".scifigbench_refs" if cache_dir is not None
                 else Path.home() / ".cache/scifigbench/revisions")
    key = hashlib.sha256((repo_id + "\n" + revision).encode("utf-8")).hexdigest()
    record = directory / (key + ".json")
    if local_files_only:
        try:
            state = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DownloadError("Offline revision is not cached. Use a full Commit SHA or resolve the revision online first.") from exc
        sha = state.get("commit") if isinstance(state, dict) else None
        if not isinstance(state, dict) or state.get("repo_id") != repo_id or state.get("revision") != revision or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise DownloadError("Invalid cached revision identity")
        return sha
    from huggingface_hub import HfApi
    try:
        sha = HfApi().repo_info(repo_id=repo_id, repo_type="dataset", revision=revision).sha
    except Exception as exc:
        raise DownloadError("Could not resolve the dataset revision to a Commit SHA.") from exc
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise DownloadError("Hub returned an invalid dataset Commit SHA")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(record, {"repo_id": repo_id, "revision": revision, "commit": sha})
    return sha


def _hub_evaluation_file(repo_id: str, commit: str, filename: str, *,
                         cache_dir=None, local_files_only: bool = False) -> Path:
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(
            repo_id=repo_id, repo_type="dataset", revision=commit, filename=filename,
            cache_dir=str(Path(cache_dir).expanduser()) if cache_dir is not None else None,
            local_files_only=local_files_only,
        ))
    except Exception as exc:
        mode = "offline cache" if local_files_only else "Hub"
        raise DownloadError(f"Required pinned file is unavailable in the {mode}: {filename}. "
                            "Prepare this Commit's reference manifests and required files online first.") from exc


def _existing_evaluation_files(root: Path, expected: dict[str, str]) -> set[str]:
    """Compare evaluation entries only; unrelated README checksum changes are valid."""
    local_checks = _checksums(root / "checksums.sha256") if (root / "checksums.sha256").is_file() else None
    missing = set()
    for rel, digest in expected.items():
        if local_checks is not None and local_checks.get(rel) != digest:
            raise DownloadError(f"Local checksum entry differs from the pinned evaluation data: {rel}")
        path = root / rel
        if path.exists() or path.is_symlink():
            if not path.is_file() or file_sha256(path) != digest:
                raise DownloadError(f"SHA256 integrity check failed for pinned evaluation data: {rel}. Existing files were not overwritten.")
        else:
            missing.add(rel)
    return missing


def _install_evaluation_file(source: Path, root: Path, rel: str, digest: str) -> None:
    """Install a verified missing file, never replace an existing user's file."""
    import shutil
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.parent.resolve().is_relative_to(root.resolve()):
        raise DownloadError(f"Download destination leaves data_root: {rel}")
    if file_sha256(source) != digest:
        raise DownloadError(f"SHA256 integrity check failed for pinned download: {rel}")
    try:
        with target.open("xb") as dest, source.open("rb") as src:
            shutil.copyfileobj(src, dest)
            dest.flush()
            os.fsync(dest.fileno())
    except FileExistsError:
        if not target.is_file() or file_sha256(target) != digest:
            raise DownloadError(f"Existing file conflicts with pinned download: {rel}") from None


def _prepare_evaluation_data(*, repo_id: str, commit: str, data_root=None,
                             cache_dir=None, local_files_only: bool = False) -> tuple[Path, dict, dict]:
    """Reuse matching QA even when other documents belong to a newer Hub commit."""
    checksum_file = _hub_evaluation_file(repo_id, commit, "checksums.sha256",
                                         cache_dir=cache_dir, local_files_only=local_files_only)
    checks = _checksums(checksum_file)
    manifest_file = _hub_evaluation_file(repo_id, commit, "QA/manifest.json",
                                         cache_dir=cache_dir, local_files_only=local_files_only)
    if file_sha256(manifest_file) != checks.get("QA/manifest.json"):
        raise DownloadError("Pinned QA manifest SHA256 mismatch")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DownloadError("Invalid pinned QA manifest") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tasks"), dict):
        raise DownloadError("Invalid pinned QA manifest tasks")
    required = {"QA/manifest.json", "QA/checksums.sha256"} | {"QA/" + spec.filename for spec in TASKS.values()}
    if any(rel not in checks for rel in required):
        raise DownloadError("Pinned checksum manifest is missing required QA entries")
    for spec in TASKS.values():
        entry = manifest["tasks"].get(spec.filename)
        if (not isinstance(entry, dict) or entry.get("sha256") != checks["QA/" + spec.filename]
                or not isinstance(entry.get("total_questions"), int)
                or isinstance(entry["total_questions"], bool) or entry["total_questions"] <= 0):
            raise DownloadError(f"Invalid pinned QA manifest entry: {spec.filename}")
    default_root = Path.home() / ".cache/scifigbench/datasets" / repo_id.replace("/", "--") / commit
    configured = data_root if data_root is not None else os.environ.get("SCIFIGBENCH_DATA_DIR")
    root = Path(configured).expanduser().resolve() if configured is not None else default_root.resolve()
    if root.exists() and not root.is_dir():
        raise DownloadError("data_root must be a directory")
    expected = {rel: checks[rel] for rel in required}
    missing = _existing_evaluation_files(root, expected)
    if missing:
        if local_files_only:
            raise DownloadError(f"Offline data_root is missing required QA files: {', '.join(sorted(missing))}")
        # Use the established QA downloader in a private versioned staging area.
        # It never writes into a supplied existing dataset root.
        staging = (Path(cache_dir).expanduser() / ".scifigbench_qa" if cache_dir is not None
                   else Path.home() / ".cache/scifigbench/qa")
        staging = staging / repo_id.replace("/", "--") / commit
        _existing_evaluation_files(staging, expected)
        download_data(repo_id=repo_id, revision=commit, local_dir=staging,
                      cache_dir=cache_dir, qa_only=True)
        _existing_evaluation_files(staging, expected)
        root.mkdir(parents=True, exist_ok=True)
        for rel in sorted(missing):
            _install_evaluation_file(staging / rel, root, rel, checks[rel])
    if not (root / "checksums.sha256").exists():
        _install_evaluation_file(checksum_file, root, "checksums.sha256", file_sha256(checksum_file))
    _existing_evaluation_files(root, expected)
    rows = load_qa(root)
    for name, spec in TASKS.items():
        if sum(q["task"] == name for q in rows) != manifest["tasks"][spec.filename]["total_questions"]:
            raise DownloadError(f"Pinned QA count mismatch: {name}")
    return root, checks, manifest


def _ensure_evaluation_images(root: Path, rows: list[dict], checks: dict, *,
                              repo_id: str, commit: str, cache_dir=None,
                              local_files_only: bool = False) -> dict[str, str]:
    paths = sorted({q["image_path"] for q in rows})
    if any(not re.fullmatch(r"images/F[0-9]{6}\.jpg", rel) or rel not in checks for rel in paths):
        raise DownloadError("Selected image is absent from the pinned checksum manifest")
    expected = {rel: checks[rel] for rel in paths}
    missing = _existing_evaluation_files(root, expected)
    if missing and local_files_only:
        raise DownloadError(f"Offline data_root is missing {len(missing)} selected images; no model was called.")
    for rel in sorted(missing):
        source = _hub_evaluation_file(repo_id, commit, rel, cache_dir=cache_dir,
                                      local_files_only=local_files_only)
        _install_evaluation_file(source, root, rel, expected[rel])
    _existing_evaluation_files(root, expected)
    return expected
