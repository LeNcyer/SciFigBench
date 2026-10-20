"""Explicit UTF-8 input contracts. Loading local data never connects to the Hub."""
from __future__ import annotations
from pathlib import Path
from collections.abc import Iterable, Mapping
from numbers import Real
import hashlib, json, math, os, re
from .tasks import TASKS, COLORS, select_tasks

class DataError(ValueError):
    """A dataset or prediction does not satisfy the documented input contract."""

class PredictionError(DataError):
    """A prediction is missing, ambiguous, or cannot be aligned to the selected QA."""

def resolve_data_root(data_root: str | Path | None = None) -> Path:
    if data_root is None:
        data_root = os.environ.get("SCIFIGBENCH_DATA_DIR") or Path.home()/".cache/scifigbench/data"
    return Path(data_root).expanduser().resolve()

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise DataError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise DataError(f"{path}:{line_number}: each JSONL row must be an object")
                rows.append(row)
    except UnicodeError as exc:
        raise DataError(f"{path}: file is not valid UTF-8: {exc}") from exc
    if not rows:
        raise DataError(f"JSONL file is empty: {path}")
    return rows

def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0

def _box(box, size, context: str) -> None:
    if not isinstance(box, list) or len(box) != 4 or any(not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(v) for v in box):
        raise DataError(f"{context}: expected a finite four-number bounding box")
    if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
        raise DataError(f"{context}: bounding box must have positive area and lie inside image_size")

def validate_render_fields(row: Mapping) -> None:
    """Validate only rendering fields; never inspect answer or object metadata."""
    if not isinstance(row, Mapping):
        raise DataError("QA row must be an object")
    task = row.get("task")
    if not isinstance(task, str) or task not in TASKS:
        raise DataError(f"Unknown QA task {task!r}")
    path = row.get("image_path")
    if not isinstance(path, str) or not re.fullmatch(r"images/F[0-9]{6}\.jpg", path):
        raise DataError("image_path must be an anonymous relative path: images/Fxxxxxx.jpg")
    size = row.get("image_size")
    if not isinstance(size, list) or len(size) != 2 or not all(_integer(v) and v > 0 for v in size):
        raise DataError("image_size must be [positive width, positive height]")
    if task == "dense_content_perception":
        boxes, colors = row.get("boxes_xyxy"), row.get("colors")
        if not isinstance(boxes, list) or not 1 <= len(boxes) <= 6 or colors != COLORS[:len(boxes)]:
            raise DataError("DCP needs 1..6 boxes and their fixed red-to-violet color order")
        for box in boxes:
            _box(box, size, "boxes_xyxy")
    else:
        for key in (["red_box_xyxy", "blue_box_xyxy"] if task in ("edge_level_verification", "path_level_reachability") else ["red_box_xyxy"] if task == "node_level_degree_counting" else []):
            _box(row.get(key), size, key)

def validate_qa(row: dict) -> None:
    validate_render_fields(row)
    common = {"qid", "task", "sample_id", "image_path", "image_size", "prompt", "answer"}
    task = row["task"]
    if not common <= row.keys():
        raise DataError(f"Missing QA fields: {sorted(common-row.keys())}")
    for key in ("qid", "prompt"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise DataError(f"QA {key} must be a nonempty string")
    if not isinstance(row["sample_id"], str) or not re.fullmatch(r"F[0-9]{6}", row["sample_id"]) or row["image_path"] != f"images/{row['sample_id']}.jpg":
        raise DataError("sample_id and anonymous image_path do not match")
    extra = {"dense_content_perception": {"boxes_xyxy", "colors"}, "edge_level_verification": {"red_box_xyxy", "blue_box_xyxy"}, "path_level_reachability": {"red_box_xyxy", "blue_box_xyxy"}, "node_level_degree_counting": {"red_box_xyxy"}, "information_flow_tracing": set()}[task]
    if task == "dense_content_perception":
        meta_keys = {"node_ids"} if len(row["boxes_xyxy"]) > 1 else set()
    else:
        meta_keys = {"edge_level_verification": {"src_node_id", "dst_node_id"}, "path_level_reachability": {"src_node_id", "dst_node_id"}, "node_level_degree_counting": {"target_node_id", "variant"}, "information_flow_tracing": {"target_node_id", "target_text", "variant"}}[task]
    expected = common | extra | ({"meta"} if meta_keys else set())
    if set(row) != expected:
        raise DataError(f"QA fields differ from the Public whitelist: extra={sorted(row.keys()-expected)}, missing={sorted(expected-row.keys())}")
    meta = row.get("meta", {})
    if not isinstance(meta, dict) or set(meta) != meta_keys:
        raise DataError(f"Public meta fields must be exactly {sorted(meta_keys)}")
    for key in meta_keys & {"src_node_id", "dst_node_id", "target_node_id"}:
        if not _integer(meta[key]):
            raise DataError(f"meta.{key} must be a nonnegative integer")
    if "variant" in meta and meta["variant"] not in ("in", "out"):
        raise DataError("meta.variant must be 'in' or 'out'")
    if "target_text" in meta and (not isinstance(meta["target_text"], str) or not meta["target_text"].strip()):
        raise DataError("meta.target_text must be a nonempty string")
    answer = row["answer"]
    if task == "dense_content_perception":
        k = len(row["boxes_xyxy"])
        if not isinstance(answer, list) or len(answer) != k or not all(isinstance(x, str) and x.strip() for x in answer):
            raise DataError("DCP answer must contain one nonempty string per box")
        if k > 1 and (not isinstance(meta["node_ids"], list) or len(meta["node_ids"]) != k or not all(_integer(v) for v in meta["node_ids"]) or len(set(meta["node_ids"])) != k):
            raise DataError("DCP flow node_ids must contain K distinct integer IDs")
    elif task in ("edge_level_verification", "path_level_reachability"):
        if answer not in ("true", "false"):
            raise DataError("Boolean-task answer must be the string 'true' or 'false'")
    elif task == "node_level_degree_counting":
        if not isinstance(answer, str) or not re.fullmatch(r"[0-9]+", answer):
            raise DataError("Degree answer must be a nonnegative integer string")
    else:
        try:
            value = json.loads(answer) if isinstance(answer, str) else None
        except ValueError as exc:
            raise DataError("IFT answer must be a JSON-encoded string list") from exc
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise DataError("IFT answer must be a JSON-encoded string list")

def load_qa(data_root: str | Path | None = None, task: str = "all") -> list[dict]:
    root = resolve_data_root(data_root)
    result, seen = [], set()
    for name in select_tasks(task):
        path = root/"QA"/TASKS[name].filename
        for index, row in enumerate(read_jsonl(path), 1):
            try:
                validate_qa(row)
            except DataError as exc:
                raise DataError(f"{path}, row {index}: {exc}") from exc
            if row["task"] != name:
                raise DataError(f"{path}: wrong task {row['task']!r}; expected {name!r}")
            if row["qid"] in seen:
                raise DataError(f"Duplicate QA QID: {row['qid']}")
            seen.add(row["qid"])
            result.append(row)
    return result

def load_predictions(predictions, prediction_field: str = "response") -> dict[str, str]:
    if not isinstance(prediction_field, str) or not prediction_field:
        raise PredictionError("prediction_field must be an explicit nonempty field name")
    if isinstance(predictions, (str, Path)):
        rows = read_jsonl(predictions)
    elif isinstance(predictions, Mapping):
        rows = [{"qid": qid, prediction_field: value} for qid, value in predictions.items()]
    else:
        try:
            rows = list(predictions)
        except TypeError as exc:
            raise PredictionError("Predictions must be a JSONL path, a QID-to-text mapping, or an iterable of objects") from exc
    if not rows:
        raise PredictionError("Predictions are empty")
    by_qid = {}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping) or not isinstance(row.get("qid"), str) or not row["qid"].strip():
            raise PredictionError(f"Prediction row {index}: qid must be a nonempty string")
        qid = row["qid"]
        if prediction_field not in row:
            raise PredictionError(f"Prediction row {index} ({qid}): missing explicit field {prediction_field!r}; GT fields are never guessed")
        if not isinstance(row[prediction_field], str):
            raise PredictionError(f"Prediction row {index} ({qid}): {prediction_field} must be raw text")
        if qid in by_qid:
            raise PredictionError(f"Duplicate prediction QID: {qid}")
        by_qid[qid] = row[prediction_field]
    return by_qid
