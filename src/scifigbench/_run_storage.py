"""Durable run records and strict resume identity; no model or dataset construction."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile
import uuid


class RunError(RuntimeError):
    """A run could not complete; saved progress is available at run_dir."""

    def __init__(self, message: str, *, run_dir: Path | None = None):
        super().__init__(message)
        self.run_dir = run_dir


class ResumeError(RunError):
    """Saved predictions cannot safely be reused for this request."""


class ModelCallError(RunError):
    """The model raised an exception or violated the raw-string protocol."""

    def __init__(self, message: str, *, run_dir: Path, qid: str):
        super().__init__(message, run_dir=run_dir)
        self.qid = qid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Replace a JSON record atomically, keeping the previous record on failure."""
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def run_lock(root: Path):
    """OS locks are released after process death; an existing lock file is harmless."""
    stream = (root / ".run.lock").open("a+b")
    try:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ResumeError("Another process is writing this run.", run_dir=root) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def create_run(output_dir: str | Path | None) -> Path:
    parent = Path(output_dir).expanduser().resolve() if output_dir is not None else Path.cwd() / "scifigbench_runs"
    parent.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    root = parent / name
    root.mkdir()
    return root


def read_record(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResumeError(f"Missing or invalid run record: {path.name}", run_dir=path.parent) from exc
    if not isinstance(value, dict):
        raise ResumeError(f"Invalid run record: {path.name}", run_dir=path.parent)
    return value


def check_identity(root: Path, identity: dict) -> dict:
    saved = read_record(root / "run.json")
    previous = saved.get("identity")
    if not isinstance(previous, dict) or saved.get("identity_sha256") != canonical_hash(previous):
        raise ResumeError("Saved run identity is invalid.", run_dir=root)
    if not identity.get("model_id") or not previous.get("model_id"):
        raise ResumeError("Resume requires the same explicit model_id, set when the original run was created.", run_dir=root)
    if previous != identity:
        fields = sorted(key for key in previous.keys() | identity.keys() if previous.get(key) != identity.get(key))
        raise ResumeError(f"Run identity mismatch: {', '.join(fields)}. Start a new run; predictions were not merged.", run_dir=root)
    return saved


def read_predictions(root: Path, qids: list[str]) -> dict[str, str]:
    """Recover only an incomplete final record, retaining a byte-for-byte backup."""
    path = root / "predictions.jsonl"
    if not path.is_file():
        raise ResumeError("Run is missing predictions.jsonl.", run_dir=root)
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    predictions: dict[str, str] = {}
    wanted = set(qids)
    valid_end = 0
    recover_tail = False
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                recover_tail = True
                break
            raise ResumeError(f"Corrupt prediction record at line {index + 1}.", run_dir=root) from exc
        if (not isinstance(row, dict) or set(row) != {"qid", "response"}
                or not isinstance(row.get("qid"), str) or not isinstance(row.get("response"), str)):
            raise ResumeError(f"Invalid prediction record at line {index + 1}.", run_dir=root)
        qid = row["qid"]
        if qid not in wanted or qid in predictions:
            raise ResumeError(f"Unknown or duplicate prediction QID at line {index + 1}.", run_dir=root)
        predictions[qid] = row["response"]
        valid_end += len(line)
    # Predictions are written serially in the fixed plan order. Reject reordered
    # or spliced files even if their individual QIDs would otherwise be valid.
    if list(predictions) != qids[:len(predictions)]:
        raise ResumeError("Predictions are not a prefix of the saved evaluation plan.", run_dir=root)
    if recover_tail or (raw and not raw.endswith(b"\n")):
        backup = root / ("predictions.recovery-" + uuid.uuid4().hex + ".jsonl")
        with backup.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        recovered = raw[:valid_end]
        if recovered and not recovered.endswith(b"\n"):
            recovered += b"\n"
        with path.open("wb") as stream:
            stream.write(recovered)
            stream.flush()
            os.fsync(stream.fileno())
        append_jsonl(root / "errors.jsonl", {
            "time": utc_now(), "type": "PredictionTailRecovery",
            "backup": backup.name, "discarded_bytes": len(raw) - valid_end,
        })
    return predictions
