import json
import shutil
import sys

import pytest
from PIL import Image

from scifigbench import (
    SciFigBench, BenchmarkResult, FullEvaluationRequired, ModelCallError,
    RunError, ResumeError, DownloadError, DataError, evaluate, load_predictions,
)
from scifigbench.runner import _sample
from runner_support import (
    COMMIT, REPO_ID, TASK_NAMES, HubStub, make_runner_dataset,
    refresh_root_checksums,
)


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    source, rows = make_runner_dataset(tmp_path / "source")
    hub = HubStub(source, tmp_path / "hub-files")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    kwargs = dict(repo_id=REPO_ID, revision=COMMIT, data_root=tmp_path / "data",
                  cache_dir=tmp_path / "cache", output_dir=tmp_path / "runs")
    return source, rows, hub, kwargs


class RecordingModel:
    def __init__(self, *, fail_at=None, response="invalid-output", interrupt=False):
        self.calls = []
        self.fail_at, self.response, self.interrupt = fail_at, response, interrupt

    def __call__(self, **payload):
        assert set(payload) == {"prompt", "image"}
        assert isinstance(payload["prompt"], str) and "REFERENCE-ONLY" not in payload["prompt"]
        assert isinstance(payload["image"], Image.Image) and payload["image"].mode == "RGB"
        self.calls.append(payload)
        if len(self.calls) == self.fail_at:
            if self.interrupt:
                raise KeyboardInterrupt()
            raise RuntimeError("provider-secret-must-not-be-logged")
        return self.response


def run(model, setup, **overrides):
    kwargs = {**setup[3], **overrides}
    if kwargs.get("resume_from") is not None:
        kwargs.pop("output_dir", None)
    return SciFigBench(model=model, **kwargs)


def test_default_quick_is_simple_and_matches_original_evaluator(setup_run):
    model = RecordingModel()
    result = run(model, setup_run)
    assert isinstance(result, BenchmarkResult)
    assert len(model.calls) == 25
    assert result.evaluation_mode == "quick" and not result.is_full_benchmark
    assert result.report["n_evaluated"] == 25 and result.report["n_qa_total"] == 30
    assert result.report["coverage"] == pytest.approx(25 / 30)
    assert "NOT official" in result.report["score_notice"]
    assert set(result.scores) == set(TASK_NAMES) and set(result.scores.values()) == {0.0}
    assert result.report["evaluation"] == evaluate(
        result.predictions_path, data_root=setup_run[3]["data_root"], allow_partial=True,
    )
    for row in result.report["tasks"].values():
        assert row["n_planned"] == row["n_evaluated"] == 5
        assert row["n_qa_total"] == 6 and row["coverage"] == pytest.approx(5 / 6)
    for line in result.predictions_path.read_text(encoding="utf-8").splitlines():
        assert set(json.loads(line)) == {"qid", "response"}
    with pytest.raises(ValueError):
        model.calls[0]["image"].getpixel((0, 0))


@pytest.mark.parametrize("task,cap,expected,mode", [
    (TASK_NAMES[1], 5, 5, "quick"), ("all", 2, 10, "partial"),
])
def test_single_and_partial_modes(setup_run, task, cap, expected, mode):
    model = RecordingModel()
    result = run(model, setup_run, task=task, max_samples_per_task=cap)
    assert len(model.calls) == expected and result.evaluation_mode == mode
    assert not result.is_full_benchmark


@pytest.mark.parametrize("cap,confirmation,task,count", [
    (None, False, "all", 30), (6, False, "all", 30), (100, True, "all", 30),
    (6, True, TASK_NAMES[1], 6),
])
def test_full_guard_precedes_models_and_images(setup_run, cap, confirmation, task, count):
    model = RecordingModel()
    with pytest.raises(FullEvaluationRequired) as error:
        run(model, setup_run, task=task, max_samples_per_task=cap, confirm_full_run=confirmation)
    message = str(error.value)
    assert f"{count:,} QA" in message and "API costs" in message and "No model was called" in message
    assert not model.calls and not (setup_run[3]["data_root"] / "images").exists()
    assert not any(q["filename"].startswith("images/") for q in setup_run[2].requests)


def test_confirmation_alone_does_not_expand_default_run(setup_run):
    model = RecordingModel()
    result = run(model, setup_run, confirm_full_run=True)
    assert len(model.calls) == 25 and result.evaluation_mode == "quick"


@pytest.mark.parametrize("task,expected,is_benchmark", [("all", 30, True), (TASK_NAMES[1], 6, False)])
def test_explicit_full(setup_run, task, expected, is_benchmark):
    model = RecordingModel()
    result = run(model, setup_run, task=task, max_samples_per_task=None, confirm_full_run=True)
    assert len(model.calls) == expected
    assert result.evaluation_mode == "full" and result.is_full_benchmark is is_benchmark
    assert result.report["coverage"] == 1 and result.report["evaluation"]["evaluation_mode"] == "Full Evaluation"


def test_real_selected_count_is_not_hardcoded(tmp_path, monkeypatch):
    counts = {name: 6 for name in TASK_NAMES}
    counts[TASK_NAMES[1]] = 17
    source, _ = make_runner_dataset(tmp_path / "source", counts=counts)
    hub = HubStub(source, tmp_path / "hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    model = RecordingModel()
    with pytest.raises(FullEvaluationRequired, match="17 QA pairs"):
        SciFigBench(model, task=TASK_NAMES[1], max_samples_per_task=None, repo_id=REPO_ID,
                    revision=COMMIT, data_root=tmp_path / "data", cache_dir=tmp_path / "cache")
    assert not model.calls


def test_sampling_is_reproducible_and_independent_of_file_order(setup_run):
    rows = setup_run[1]
    first = _sample(rows, TASK_NAMES, 5, 0)
    assert [q["qid"] for q in first] == [q["qid"] for q in _sample(list(reversed(rows)), TASK_NAMES, 5, 0)]
    assert [q["qid"] for q in first] != [q["qid"] for q in _sample(rows, TASK_NAMES, 5, 1)]
    one = run(RecordingModel(), setup_run)
    two = run(RecordingModel(), setup_run)
    assert one.predictions_path.read_bytes() == two.predictions_path.read_bytes()
    assert one.run_dir != two.run_dir


def test_exact_images_cache_reuse_and_no_annotations(setup_run):
    one = run(RecordingModel(), setup_run)
    hub = setup_run[2]
    wanted = set(one.report["dataset"]["image_sha256"])
    downloaded = {q["filename"] for q in hub.requests if q["filename"].startswith("images/")}
    assert downloaded == wanted and len(wanted) == 25
    assert len(list((setup_run[3]["data_root"] / "images").glob("*.jpg"))) == 25
    assert not any("annotations" in str(q) for q in hub.requests + hub.snapshot_calls)
    assert all("images/*.jpg" not in q["allow_patterns"] for q in hub.snapshot_calls)
    count, snapshots = len(hub.requests), len(hub.snapshot_calls)
    run(RecordingModel(), setup_run, model_id="another-model")
    assert len(hub.requests) == count and len(hub.snapshot_calls) == snapshots


def test_shared_images_download_once(tmp_path, monkeypatch):
    source, _ = make_runner_dataset(tmp_path / "source", shared_image=True)
    hub = HubStub(source, tmp_path / "hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    result = SciFigBench(RecordingModel(), repo_id=REPO_ID, revision=COMMIT,
                        data_root=tmp_path / "data", cache_dir=tmp_path / "cache", output_dir=tmp_path / "runs")
    assert len(result.report["dataset"]["image_sha256"]) == 1
    assert sum(q["filename"].startswith("images/") for q in hub.requests) == 1


def test_reuse_existing_data_with_changed_documents(setup_run, tmp_path):
    local = tmp_path / "existing"
    shutil.copytree(setup_run[0], local)
    (local / "README.md").write_text("Newer document, unchanged benchmark data.\n", encoding="utf-8")
    refresh_root_checksums(local)
    before = {p.relative_to(local).as_posix(): p.read_bytes() for p in local.rglob("*") if p.is_file()}
    result = run(RecordingModel(), setup_run, data_root=local)
    assert result.report["n_evaluated"] == 25
    assert not setup_run[2].snapshot_calls
    assert not any(q["filename"].startswith("images/") for q in setup_run[2].requests)
    assert before == {p.relative_to(local).as_posix(): p.read_bytes() for p in local.rglob("*") if p.is_file()}


@pytest.mark.parametrize("rel", ["QA/qa_edge_level_verification.jsonl", "QA/manifest.json", "images/F000001.jpg"])
def test_hash_mismatch_never_overwrites_existing_data(setup_run, tmp_path, rel):
    local = tmp_path / "existing"
    shutil.copytree(setup_run[0], local)
    path = local / rel
    path.write_bytes(b"corrupt-existing-file")
    # Select full coverage so even F000001 is definitely required.
    model = RecordingModel()
    with pytest.raises((DownloadError, RunError), match="SHA256"):
        run(model, setup_run, data_root=local, max_samples_per_task=None, confirm_full_run=True)
    assert path.read_bytes() == b"corrupt-existing-file" and not model.calls


def test_offline_reuses_exact_cached_data(setup_run):
    run(RecordingModel(), setup_run)
    hub = setup_run[2]
    before = (len(hub.requests), len(hub.api_calls), len(hub.snapshot_calls))
    result = run(RecordingModel(), setup_run, local_files_only=True)
    assert result.report["n_evaluated"] == 25
    assert before == (len(hub.requests), len(hub.api_calls), len(hub.snapshot_calls))


def test_offline_missing_references_fails_without_calling_model(setup_run):
    model = RecordingModel()
    with pytest.raises(DownloadError, match="offline cache"):
        run(model, setup_run, local_files_only=True)
    assert not model.calls


def test_offline_missing_selected_image_does_not_fetch_it(setup_run):
    first = run(RecordingModel(), setup_run)
    rel = next(iter(first.report["dataset"]["image_sha256"]))
    (setup_run[3]["data_root"] / rel).unlink()
    model = RecordingModel()
    requests = len(setup_run[2].requests)
    with pytest.raises(RunError):
        run(model, setup_run, local_files_only=True)
    assert not model.calls and len(setup_run[2].requests) == requests


def test_mutable_revision_resolved_once_then_pinned_and_offline_reused(setup_run):
    first = run(RecordingModel(), setup_run, revision="main")
    hub = setup_run[2]
    assert len(hub.api_calls) == 1 and first.report["dataset"]["commit"] == COMMIT
    assert all(q["revision"] == COMMIT for q in hub.requests + hub.snapshot_calls)
    result = run(RecordingModel(), setup_run, revision="main", local_files_only=True)
    assert len(hub.api_calls) == 1 and result.report["dataset"]["requested_revision"] == "main"


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5"])
def test_invalid_sample_caps_fail_before_network(setup_run, value):
    with pytest.raises(ValueError, match="positive integer"):
        run(RecordingModel(), setup_run, max_samples_per_task=value)
    assert not setup_run[2].requests and not setup_run[2].snapshot_calls


def test_call_failure_preserves_partial_scores_and_null_tasks(setup_run):
    with pytest.raises(ModelCallError) as error:
        run(RecordingModel(fail_at=3), setup_run, model_id="test-model")
    root = error.value.run_dir
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["evaluation_mode"] == "partial"
    assert report["n_evaluated"] == 2 and report["n_model_calls"] == 3
    assert report["tasks"][TASK_NAMES[0]]["primary_metric"]["value"] == 0.0
    assert all(report["tasks"][t]["primary_metric"]["value"] is None for t in TASK_NAMES[1:])
    assert len(load_predictions(root / "predictions.jsonl")) == 2
    assert "provider-secret" not in (root / "errors.jsonl").read_text(encoding="utf-8")


@pytest.mark.parametrize("response", [None, True, 2, ["raw"], {"text": "x"}])
def test_non_string_outputs_are_protocol_errors_not_wrong_answers(setup_run, response):
    with pytest.raises(ModelCallError) as error:
        run(RecordingModel(response=response), setup_run)
    report = json.loads((error.value.run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["n_evaluated"] == 0 and report["evaluation"] is None
    assert all(q["primary_metric"]["value"] is None for q in report["tasks"].values())
    with pytest.raises(DataError):
        load_predictions(error.value.run_dir / "predictions.jsonl")


def test_resume_skips_saved_responses_and_retains_attempt_counts(setup_run):
    with pytest.raises(ModelCallError) as error:
        run(RecordingModel(fail_at=3), setup_run, model_id="stable-model")
    saved = load_predictions(error.value.run_dir / "predictions.jsonl")
    model = RecordingModel()
    result = run(model, setup_run, model_id="stable-model", resume_from=error.value.run_dir)
    assert len(model.calls) == 23 and result.report["n_model_calls"] == 26
    assert result.report["n_model_calls_this_invocation"] == 23
    assert all(load_predictions(result.predictions_path)[qid] == text for qid, text in saved.items())
    assert result.report["status"] == "completed" and result.evaluation_mode == "quick"


def test_interrupted_full_is_partial_until_successful_resume(setup_run):
    options = dict(max_samples_per_task=None, confirm_full_run=True, model_id="stable")
    with pytest.raises(KeyboardInterrupt) as error:
        run(RecordingModel(fail_at=3, interrupt=True), setup_run, **options)
    root = error.value.run_dir
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "interrupted" and report["requested_mode"] == "full"
    assert report["evaluation_mode"] == "partial" and not report["is_full_benchmark"]
    with pytest.raises(FullEvaluationRequired):
        run(RecordingModel(), setup_run, resume_from=root, **{**options, "confirm_full_run": False})
    model = RecordingModel()
    result = run(model, setup_run, resume_from=root, **options)
    assert len(model.calls) == 28 and result.is_full_benchmark


@pytest.mark.parametrize("change", [
    {"model_id": "other"}, {"seed": 1}, {"max_samples_per_task": 2},
    {"task": TASK_NAMES[0]}, {"revision": "b" * 40},
])
def test_resume_rejects_mixed_run_identities(setup_run, change):
    with pytest.raises(ModelCallError) as error:
        run(RecordingModel(fail_at=2), setup_run, model_id="stable")
    model = RecordingModel()
    with pytest.raises(ResumeError, match="identity mismatch"):
        run(model, setup_run, resume_from=error.value.run_dir, **{"model_id": "stable", **change})
    assert not model.calls


def test_resume_rejects_changed_implementation(setup_run, monkeypatch):
    with pytest.raises(ModelCallError) as error:
        run(RecordingModel(fail_at=2), setup_run, model_id="stable")
    monkeypatch.setattr("scifigbench.runner._implementation_hashes", lambda: {"changed.py": "x"})
    with pytest.raises(ResumeError, match="implementation_sha256"):
        run(RecordingModel(), setup_run, resume_from=error.value.run_dir, model_id="stable")


@pytest.mark.parametrize("original_id,resume_id", [(None, None), (None, "new"), ("stable", None)])
def test_resume_requires_model_id_from_first_run(setup_run, original_id, resume_id):
    first = run(RecordingModel(), setup_run, model_id=original_id)
    with pytest.raises(ResumeError, match="model_id"):
        run(RecordingModel(), setup_run, resume_from=first.run_dir, model_id=resume_id)


@pytest.mark.parametrize("kind", ["duplicate", "unknown", "middle-corrupt", "reordered"])
def test_invalid_prediction_files_cannot_be_resumed(setup_run, kind):
    first = run(RecordingModel(), setup_run, model_id="stable")
    lines = first.predictions_path.read_text(encoding="utf-8").splitlines()
    if kind == "duplicate":
        lines.append(lines[0])
    elif kind == "unknown":
        lines[0] = json.dumps({"qid": "not-selected", "response": "x"})
    elif kind == "middle-corrupt":
        lines[0] = "invalid-json"
    else:
        lines[0], lines[1] = lines[1], lines[0]
    first.predictions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    model = RecordingModel()
    with pytest.raises(ResumeError):
        run(model, setup_run, resume_from=first.run_dir, model_id="stable")
    assert not model.calls


def test_truncated_tail_keeps_backup_and_resumes(setup_run):
    with pytest.raises(ModelCallError) as error:
        run(RecordingModel(fail_at=3), setup_run, model_id="stable")
    path = error.value.run_dir / "predictions.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"qid":"unfinished')
    before = path.read_bytes()
    result = run(RecordingModel(), setup_run, resume_from=error.value.run_dir, model_id="stable")
    backups = list(result.run_dir.glob("predictions.recovery-*.jsonl"))
    assert len(backups) == 1 and backups[0].read_bytes() == before
    assert len(load_predictions(result.predictions_path)) == 25


def test_single_writer_lock_blocks_concurrent_resume(setup_run):
    from scifigbench._run_storage import run_lock
    first = run(RecordingModel(), setup_run, model_id="stable")
    with run_lock(first.run_dir):
        with pytest.raises(ResumeError, match="Another process"):
            run(RecordingModel(), setup_run, resume_from=first.run_dir, model_id="stable")
    result = run(RecordingModel(), setup_run, resume_from=first.run_dir, model_id="stable")
    assert result.report["n_model_calls_this_invocation"] == 0


def test_changed_image_cannot_reach_model_or_contaminate_resumed_predictions(setup_run):
    plan = _sample(setup_run[1], TASK_NAMES, 5, 0)
    victim = setup_run[3]["data_root"] / plan[1]["image_path"]
    original = (setup_run[0] / plan[1]["image_path"]).read_bytes()
    calls = []
    def mutating_model(*, prompt, image):
        calls.append(1)
        victim.write_bytes(b"changed-during-run")
        return "invalid-output"
    with pytest.raises(RunError, match="changed before rendering") as error:
        run(mutating_model, setup_run, model_id="stable")
    assert len(calls) == 1 and len(load_predictions(error.value.run_dir / "predictions.jsonl")) == 1
    victim.write_bytes(original)
    model = RecordingModel()
    result = run(model, setup_run, model_id="stable", resume_from=error.value.run_dir)
    assert len(model.calls) == 24 and result.report["n_evaluated"] == 25
