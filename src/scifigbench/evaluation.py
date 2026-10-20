"""Score explicitly supplied model text; never derive predictions from annotations."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
from .tasks import TASKS, select_tasks
from .data import load_qa, load_predictions, resolve_data_root, file_sha256, PredictionError
from .scoring import SCORERS
from ._version import __version__

def scoring_view(q):
    t=q['task'];v={'answer':q['answer']}
    if t=='dense_content_perception':
        K=len(q['boxes_xyxy']);assert 1<=K<=6
        v.update(subtask='single' if K==1 else 'flow_path',meta={'K':K})
    elif t in ('node_level_degree_counting','information_flow_tracing'):
        v['meta']={'variant':q['meta']['variant']}
    return v

def score_question(qa: dict, response, *, no_latex_normalize: bool = False) -> dict:
    """Apply the approved scorer to one QA/response; no files or network access."""
    select_tasks(qa["task"])
    result = SCORERS[qa["task"]](scoring_view(qa), response, SimpleNamespace(no_latex_normalize=no_latex_normalize))
    result.pop("stratum", None)
    return result

def evaluate(predictions, *, data_root: str | Path | None = None, task: str = "all",
             allow_partial: bool = False, prediction_field: str = "response",
             include_details: bool = False, no_latex_normalize: bool = False) -> dict:
    """Return per-task scores. Full selected-QA coverage is required by default.

    Predictions may be a JSONL path, a QID-to-raw-text mapping, or iterable of
    objects. Unknown/unselected and duplicate QIDs are errors. Partial mode
    retains the frozen mean over matched questions and reports its coverage.
    """
    selected = select_tasks(task)
    root = resolve_data_root(data_root)
    qa = load_qa(root, task)
    pred = load_predictions(predictions, prediction_field)
    qids = {q["qid"] for q in qa}
    unknown = sorted(pred.keys()-qids)
    if unknown:
        raise PredictionError(f"Unknown or unselected prediction QIDs ({len(unknown)}): {unknown[:10]}. Select 'all' or provide predictions for the selected task(s).")
    missing = sorted(qids-pred.keys())
    if missing and not allow_partial:
        raise PredictionError(f"Full Evaluation requires every selected QA: missing {len(missing)}/{len(qa)} predictions, e.g. {missing[:10]}. Use allow_partial=True / --allow-partial for an explicit Partial Evaluation.")
    tasks, details = {}, []
    for name in selected:
        rows = [q for q in qa if q["task"] == name]
        scored = []
        for q in rows:
            if q["qid"] not in pred:
                continue
            result = score_question(q, pred[q["qid"]], no_latex_normalize=no_latex_normalize)
            scored.append(result)
            if include_details:
                details.append({"qid": q["qid"], "task": name, **result})
        n = len(scored)
        report = {"n_qa_total": len(rows), "n_total": n, "n_matched": n,
                  "n_missing_pred": len(rows)-n, "coverage": n/len(rows), "coverage_pct": 100*n/len(rows),
                  "evaluation_mode": "Partial Evaluation" if n != len(rows) else "Full Evaluation",
                  "n_parse_fail": sum(not r["parse_ok"] for r in scored),
                  "primary_metric": {"name": TASKS[name].metric, "value": sum(r["score"] for r in scored)/max(n, 1)}}
        if name == "dense_content_perception":
            report["additional_metrics"] = {"set-ANLS*": {"value": sum(r["set_score"] for r in scored)/max(n, 1)}}
        tasks[name] = report
    manifest_path = root/"QA/manifest.json"
    version = json.loads(manifest_path.read_text(encoding="utf-8")).get("version") if manifest_path.is_file() else None
    report = {"package_version": __version__, "dataset_version": version,
              "evaluation_mode": "Partial Evaluation" if missing else "Full Evaluation", "task": task,
              "n_qa_total": len(qa), "n_predictions": len(pred), "n_matched": len(pred), "n_missing_pred": len(missing),
              "coverage": len(pred)/len(qa), "coverage_pct": 100*len(pred)/len(qa),
              "prediction_field": prediction_field, "latex_normalize": not no_latex_normalize,
              "qa_sha256": {TASKS[t].filename: file_sha256(root/"QA"/TASKS[t].filename) for t in selected},
              "tasks": tasks}
    if include_details:
        report["per_question"] = details
    return report
