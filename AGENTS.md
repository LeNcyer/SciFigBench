# SciFigBench Agent Guide

## Project Overview

This toolkit downloads/loads benchmark data, prepares canonical visual inputs,
and scores predictions from the user's own VLM. It does not load or run a model.

The benchmark measures three capabilities: content perception (DCP), topology
reasoning (Edge, Degree and Reach), and information-flow tracing (IFT).

| Task argument | Main metric |
|---|---|
| `dense_content_perception` | Position-wise ANLS |
| `edge_level_verification` | Accuracy |
| `node_level_degree_counting` | Accuracy |
| `path_level_reachability` | Accuracy |
| `information_flow_tracing` | Set-based ANLS* |

## Repository Structure

- `src/scifigbench/tasks.py`: public task/file registry.
- `data.py`, `download.py`: local UTF-8 data contracts and explicit Hub access.
- `rendering.py`: canonical image rendering and safe model payloads.
- `evaluation.py`, `scoring/`: alignment, reports, approved parsers and metrics.
- `cli.py`, `__main__.py`: console and module entry points using the same API.
- `tests/`: offline tests using small synthetic data, without GPU or model APIs.

## Installation

From a checkout, using an activated virtual environment:

```bash
python -m pip install .
scifigbench --help
python -m scifigbench --help
```

For development and local distribution builds:

```bash
python -m pip install '.[test]'
python -m build
```

The evaluation toolkit uses the MIT License in `LICENSE`. This license does not
automatically apply to the separately distributed benchmark dataset. An online
PyPI release is not assumed.

## Dataset Download

Supply the dataset source with `--repo-id LeNcyer/SciFigBench` and an explicit
`--revision` (a branch, tag or commit), or configure `SCIFIGBENCH_REPO_ID` and
`SCIFIGBENCH_REVISION`. Pin a commit for reproducible downloads.

Once those variables contain the intended dataset source:

```bash
scifigbench download --data-root ./data
scifigbench download --data-root ./data --qa-only
scifigbench download --data-root ./data --annotations
```

The default subset is QA + Images. `--annotations` adds annotations; `--qa-only`
excludes images. The two flags are mutually exclusive. Use `--cache-dir` for the
Hub cache, or `--data-root` for the download destination. Explicit `--repo-id`
and `--revision` override environment configuration.

Use existing data offline without configuring an online source:

```bash
scifigbench download --data-root ./data --local-files-only
scifigbench download --data-root ./qa-only-data --local-files-only --qa-only
```

These verify the requested files against checksum and QA manifests. Only the
requested subset is claimed verified. Importing or evaluating never downloads.
Without `--data-root`, use `SCIFIGBENCH_DATA_DIR` or the user's local
`.cache/scifigbench/data` directory. Paths do not depend on the working directory.

## Evaluation Workflow

1. Load the selected Public QA using `load_qa(data_root, task=...)`.
2. Prepare each model input with `prepare_model_payload(qa, data_root=...)`.
3. Call the user's VLM with only the returned `prompt` and `image`.
4. Save each QID and the model's raw response as UTF-8 JSONL.
5. Evaluate the predictions. QA alone is sufficient for scoring; images are
   needed for visual input preparation. Annotations are never required.

Do not read answers or annotations to fabricate model responses. This package
provides no model framework, API credentials, or GPU runner.

## Prediction Format

Each row must contain a unique string QID and raw string response:

```json
{"qid":"cf32f75bc3d9","response":"true"}
```

Duplicate, unknown/unselected QIDs and invalid field types are errors. Supply
predictions for the selected task(s); select `all` for a combined file. Historical
field names require an explicit `--prediction-field` or API `prediction_field`.
Ground truth fields such as `answer`, `gt`, and `gt_answer` are never guessed.

## CLI and Python API

```bash
scifigbench evaluate --predictions predictions.jsonl --task all --data-root ./data
scifigbench evaluate --predictions predictions.jsonl --task all --data-root ./data --output report.json
scifigbench evaluate --predictions partial.jsonl --task all --data-root ./data --allow-partial
python -m scifigbench evaluate --predictions legacy.jsonl --task all --data-root ./data --prediction-field model_answer --allow-partial
```

Full coverage of selected QA is required by default. Partial mode reports matched
and missing counts and coverage; its denominator remains the matched questions.
No undefined overall five-task score is produced. `--include-details` adds the
per-question results. `--no-latex-normalize` explicitly selects the existing
normalization-off scoring option.

```python
from scifigbench import download_data, load_qa, render, prepare_model_payload, evaluate

root = "./data"
rows = load_qa(root, task="dense_content_perception")
image = render(rows[0], data_root=root)
payload = prepare_model_payload(rows[0], data_root=root)  # exactly prompt/image
# response = your_vlm(**payload)  # user-provided callable, not supplied here
report = evaluate("predictions.jsonl", task="all", data_root=root)
```

`download_data` accepts `repo_id`, `revision`, `local_dir`, `cache_dir`,
`annotations`, `qa_only`, and `local_files_only`; it returns a `Path`.
`evaluate` accepts a JSONL path, QID-to-text mapping, or iterable of prediction
objects and returns a dictionary. It also accepts `allow_partial`,
`prediction_field`, `include_details`, and `no_latex_normalize`.
`render` returns a new RGB PIL image. Close images when no longer needed.

## Testing

```bash
python -m pytest
```

Tests mock Hub access; no live repository, GPU, annotations, or model service is
needed. They cover parser boundaries, numerical edge cases, prediction alignment,
canonical rendering, input isolation, downloads and CLI/API consistency.

## Critical Constraints

- Preserve approved formulas, LaTeX normalization and JSON-first parsing.
- Determine DCP K from `len(boxes_xyxy)`, ignoring deleted construction tags.
  Single uses prediction item 0; Multi keeps its existing length penalties.
- Preserve IFT text multiplicity, empty-set and parse-failure behavior. SciPy is
  required so matching cannot silently fall back to another algorithm.
- Keep prompt text, color order, outward boxes, thickness, clipping and order.
- Never pass answers, object IDs, paths, GT or metadata to the user's model.
- Never modify official QA answers, images, annotations or benchmark files.
- Record the data version and QA hashes in reports. Do not represent old-input
  predictions as fresh inference on changed benchmark inputs.
- Keep this package focused on public data loading, visual query preparation
  and evaluation; model inference and dataset construction are outside its scope.
