"""One-call synchronous VLM evaluation with explicit Quick/Partial/Full modes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Literal

from PIL import Image

from ._version import __version__
from ._run_storage import (
    RunError, ResumeError, ModelCallError, append_jsonl, canonical_hash, check_identity,
    create_run, read_predictions, read_record, run_lock, utc_now, write_json,
)
from .data import file_sha256, load_qa
from .download import (
    _resolve_evaluation_revision, _prepare_evaluation_data, _ensure_evaluation_images,
    _existing_evaluation_files, DownloadError,
)
from .evaluation import evaluate
from .rendering import prepare_model_payload
from .tasks import TASKS, select_tasks

DEFAULT_DATASET_COMMIT = "a884c09770b99e6337c4cb27cee4696773e27d6e"
_SAMPLING_VERSION = "sha256-json-v1"


class VLMCallable(Protocol):
    """A user-owned synchronous model accepting only prompt and rendered RGB image."""

    def __call__(self, *, prompt: str, image: Image.Image) -> str:
        """Return unmodified raw model text, including text-form JSON outputs."""
        ...


class FullEvaluationRequired(ValueError):
    """A full selected-task run was requested without both explicit opt-in settings."""


@dataclass(frozen=True)
class BenchmarkResult:
    """Saved per-task scores; quick/partial results are not full Benchmark scores."""

    report: dict
    run_dir: Path

    @property
    def scores(self) -> dict[str, float | None]:
        """Primary metric per selected task, in [0, 1]; no aggregate score."""
        return {name: row["primary_metric"]["value"] for name, row in self.report["tasks"].items()}

    @property
    def evaluation_mode(self) -> Literal["quick", "partial", "full"]:
        return self.report["evaluation_mode"]

    @property
    def is_full_benchmark(self) -> bool:
        return self.report["is_full_benchmark"]

    @property
    def predictions_path(self) -> Path:
        return self.run_dir / "predictions.jsonl"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.json"


def _sample(rows: list[dict], selected: tuple[str, ...], cap: int | None, seed: int) -> list[dict]:
    result = []
    for task in selected:
        task_rows = [q for q in rows if q["task"] == task]
        # JSON with fixed separators makes seed/task/QID boundaries unambiguous.
        task_rows.sort(key=lambda q: (
            canonical_hash([_SAMPLING_VERSION, seed, task, q["qid"]]), q["qid"],
        ))
        result.extend(task_rows if cap is None else task_rows[:cap])
    return result


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(root.rglob("*.py"))}


def SciFigBench(
    model: VLMCallable,
    *,
    task: str = "all",
    max_samples_per_task: int | None = 5,
    confirm_full_run: bool = False,
    seed: int = 0,
    repo_id: str = "LeNcyer/SciFigBench",
    revision: str = DEFAULT_DATASET_COMMIT,
    data_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_id: str | None = None,
    resume_from: str | Path | None = None,
    local_files_only: bool = False,
) -> BenchmarkResult:
    """Run a user's VLM through the existing input and scoring pipeline.

    Quick Test is the default: at most 5 QA per selected task and 25 model
    Callable calls for task="all". Quick/Partial scores are NOT official full
    Benchmark results. Installation and import never download data; this
    function downloads verified QA and only the images needed by the plan.

    Full Evaluation requires BOTH max_samples_per_task=None and
    confirm_full_run=True. All five tasks currently contain 40,670 QA pairs:
    a full run may make 40,670 model calls and incur substantial API costs.
    Single-task counts come from the verified QA. A finite cap covering every
    selected QA cannot bypass confirmation. Interrupted Full Evaluation is
    saved as incomplete/partial, never completed full.

    The model is called synchronously with exactly prompt and image keyword
    arguments. Image is a canonical RGB PIL image, closed after the call.
    Return raw text (including JSON encoded as text); no GT, answers, object
    IDs, metadata, paths or annotations are passed. There are no automatic
    retries. The user's wrapper controls SDK timeouts and any extra requests.

    New runs require no model_id, cache path or run identifier. To resume,
    set model_id from the FIRST call, then supply the same model_id and
    configuration with resume_from=result.run_dir (or an exception's run_dir).
    The identifier must accurately describe the user's model/checkpoint and
    decoding configuration; arbitrary Callable weights cannot be inferred.
    Resume skips durable responses and rejects different models, data commits,
    plans or implementations. A charged response lost before local persistence
    may have to be called again. output_dir and resume_from are mutually
    exclusive. output_dir is the parent of a newly created unique run directory.

    The default data revision is an immutable accepted Hugging Face commit.
    Branches/tags are resolved once per online call and then pinned. Offline
    calls require cached reference manifests and all selected QA/images.
    data_root may reuse matching evaluation data despite unrelated README or
    other document changes; data hash conflicts are errors, never overwritten.

    Example:
        result = SciFigBench(model=model)  # Quick Test: <=25 calls, partial scores
        full = SciFigBench(
            model=model, max_samples_per_task=None, confirm_full_run=True,
            model_id="model-checkpoint-and-decoding-v1",
        )  # Full: up to 40,670 calls and substantial API costs
    """
    selected = select_tasks(task)
    if not callable(model):
        raise TypeError("model must be callable with prompt and image keyword arguments")
    if max_samples_per_task is not None and (
            not isinstance(max_samples_per_task, int) or isinstance(max_samples_per_task, bool)
            or max_samples_per_task <= 0):
        raise ValueError("max_samples_per_task must be a positive integer or None")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(confirm_full_run, bool) or not isinstance(local_files_only, bool):
        raise ValueError("confirm_full_run and local_files_only must be booleans")
    if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
        raise ValueError("model_id must be a nonempty string when provided")
    if resume_from is not None and output_dir is not None:
        raise ValueError("output_dir and resume_from are mutually exclusive")
    if resume_from is not None and model_id is None:
        raise ResumeError("Resume requires an explicit model_id set when the original run was created.")
    commit = _resolve_evaluation_revision(repo_id, revision, cache_dir=cache_dir,
                                           local_files_only=local_files_only)
    root, checks, manifest = _prepare_evaluation_data(
        repo_id=repo_id, commit=commit, data_root=data_root, cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    rows = load_qa(root, task=task)
    totals = {name: sum(q["task"] == name for q in rows) for name in selected}
    plan = _sample(rows, selected, max_samples_per_task, seed)
    if (max_samples_per_task is None and not confirm_full_run) or (
            max_samples_per_task is not None and len(plan) == len(rows)):
        raise FullEvaluationRequired(
            f"Full Evaluation selects {len(rows):,} QA pairs for {task}, potentially "
            f"{len(rows):,} model Callable calls and substantial API costs. "
            "Set max_samples_per_task=None AND confirm_full_run=True. "
            "A finite cap covering all selected QA cannot bypass confirmation. No model was called. "
            "Quick Test defaults to at most 5 QA per task (25 calls for task='all'); "
            "Quick/Partial scores are NOT full Benchmark results."
        )
    requested_mode = "full" if max_samples_per_task is None else "quick" if max_samples_per_task == 5 else "partial"
    qids = [q["qid"] for q in plan]
    image_hashes = {}
    for q in plan:
        rel = q["image_path"]
        if rel not in checks:
            raise DownloadError(f"Selected image is absent from the pinned manifest: {rel}")
        image_hashes[rel] = checks[rel]
    qa_hashes = {"QA/" + spec.filename: checks["QA/" + spec.filename] for spec in TASKS.values()}
    identity = {
        "schema_version": 1, "package_version": __version__, "model_id": model_id,
        "repo_id": repo_id, "dataset_commit": commit, "dataset_version": manifest.get("version"),
        "qa_sha256": qa_hashes, "qa_manifest_sha256": checks["QA/manifest.json"],
        "qa_checksums_sha256": checks["QA/checksums.sha256"],
        "image_sha256": image_hashes, "task": task, "selected_tasks": list(selected),
        "max_samples_per_task": max_samples_per_task, "seed": seed,
        "sampling_version": _SAMPLING_VERSION, "selected_qids": qids,
        "implementation_sha256": _implementation_hashes(),
        "scoring_options": {"no_latex_normalize": False},
    }
    run_root = Path(resume_from).expanduser().resolve() if resume_from is not None else create_run(output_dir)
    if not run_root.is_dir():
        raise ResumeError("resume_from is not an existing run directory.", run_dir=run_root)
    with run_lock(run_root):
        if resume_from is not None:
            saved = check_identity(run_root, identity)
            predictions = read_predictions(run_root, qids)
            progress = read_record(run_root / "progress.json")
            calls = progress.get("model_calls")
            if not isinstance(calls, int) or isinstance(calls, bool) or calls < len(predictions):
                raise ResumeError("Invalid saved model call count.", run_dir=run_root)
        else:
            saved = {"schema_version": 1, "created_at": utc_now(),
                     "identity": identity, "identity_sha256": canonical_hash(identity)}
            write_json(run_root / "run.json", saved)
            (run_root / "predictions.jsonl").touch(exist_ok=False)
            predictions = {}
            calls = 0
            write_json(run_root / "progress.json", {"model_calls": calls, "updated_at": utc_now()})
        previous_calls = calls
        planned_counts = {name: sum(q["task"] == name for q in plan) for name in selected}

        def save_report(status: str, *, score: bool = True) -> dict:
            evaluation = None
            if predictions and score:
                # Revalidate scoring inputs before reading GT again in evaluate.
                _existing_evaluation_files(root, {
                    **qa_hashes, "QA/manifest.json": checks["QA/manifest.json"],
                    "QA/checksums.sha256": checks["QA/checksums.sha256"],
                })
                evaluation = evaluate(predictions, data_root=root, task=task, allow_partial=status != "completed" or requested_mode != "full")
            complete = status == "completed" and len(predictions) == len(plan)
            mode = requested_mode if complete else "partial"
            tasks = {}
            for name in selected:
                n = sum(q["qid"] in predictions for q in plan if q["task"] == name)
                metric = evaluation["tasks"][name] if evaluation else {
                    "n_qa_total": totals[name], "n_total": n, "n_matched": n,
                    "n_missing_pred": totals[name] - n, "coverage": n / totals[name],
                    "coverage_pct": 100 * n / totals[name],
                    "primary_metric": {"name": TASKS[name].metric, "value": None},
                }
                tasks[name] = {**metric, "n_planned": planned_counts[name],
                               "n_evaluated": n, "plan_completion": n / planned_counts[name]}
            report = {
                "schema_version": 1, "created_at": saved["created_at"], "updated_at": utc_now(),
                "status": status, "requested_mode": requested_mode, "evaluation_mode": mode,
                "is_full_benchmark": complete and mode == "full" and task == "all",
                "score_notice": ("Full selected-task evaluation." if complete and mode == "full" else
                                 "Quick/Partial scores are NOT official full Benchmark results."),
                "n_qa_total": len(rows), "n_planned": len(plan), "n_evaluated": len(predictions),
                "n_model_calls": calls, "n_model_calls_this_invocation": calls - previous_calls,
                "coverage": len(predictions) / len(rows), "plan_completion": len(predictions) / len(plan),
                "dataset": {"repo_id": repo_id, "requested_revision": revision, "commit": commit, "data_root": str(root),
                            "version": manifest.get("version"), "qa_sha256": qa_hashes,
                            "image_sha256": image_hashes},
                "tasks": tasks, "evaluation": evaluation,
            }
            write_json(run_root / "report.json", report)
            return report

        phase, current_qid = "data", None
        try:
            _ensure_evaluation_images(root, plan, checks, repo_id=repo_id, commit=commit,
                                      cache_dir=cache_dir, local_files_only=local_files_only)
            save_report("running", score=False)
            for qa in plan:
                current_qid = qa["qid"]
                if current_qid in predictions:
                    continue
                phase = "render"
                image_path = root / qa["image_path"]
                if file_sha256(image_path) != image_hashes[qa["image_path"]]:
                    raise DownloadError("Selected image changed before rendering; no prediction was recorded for this QA.")
                payload = prepare_model_payload(qa, data_root=root)
                try:
                    if set(payload) != {"prompt", "image"}:
                        raise RunError("Model payload must contain exactly prompt and image.", run_dir=run_root)
                    if file_sha256(image_path) != image_hashes[qa["image_path"]]:
                        raise DownloadError("Selected image changed during rendering; the model was not called for this QA.")
                    phase = "model"
                    calls += 1
                    write_json(run_root / "progress.json", {
                        "model_calls": calls, "last_qid": current_qid, "updated_at": utc_now(),
                    })
                    try:
                        response = model(**payload)
                        if not isinstance(response, str):
                            raise TypeError("Model response must be raw text, not a native list/bool/int or other object.")
                    except Exception as exc:
                        raise ModelCallError(
                            f"Model call failed ({type(exc).__name__}); raw-string protocol required. "
                            f"Saved progress: {run_root}", run_dir=run_root, qid=current_qid,
                        ) from exc
                finally:
                    payload["image"].close()
                phase = "save"
                append_jsonl(run_root / "predictions.jsonl", {"qid": current_qid, "response": response})
                predictions[current_qid] = response
            phase, current_qid = "evaluation", None
            _existing_evaluation_files(root, image_hashes)
            report = save_report("completed")
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            append_jsonl(run_root / "errors.jsonl", {
                "time": utc_now(), "type": type(exc).__name__, "phase": phase, "qid": current_qid,
            })
            try:
                save_report(status)
            except Exception:
                # A changed/corrupt dataset must not produce scores from different GT.
                failed_report = save_report(status, score=False)
                failed_report["score_notice"] = "Scores unavailable; the run failed before valid evaluation."
                write_json(run_root / "report.json", failed_report)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                exc.run_dir = run_root
                raise
            if isinstance(exc, RunError):
                raise
            raise RunError(f"Evaluation stopped ({type(exc).__name__}: {exc}). Saved progress: {run_root}", run_dir=run_root) from exc
    return BenchmarkResult(report=report, run_dir=run_root)
