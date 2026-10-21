<h1 align="center">🔬 SciFigBench</h1>

<p align="center">
  <strong>Evaluating VLMs' Understanding of Scientific Method Figures.</strong>
</p>

<p align="center">
  <a href="https://github.com/LeNcyer/SciFigBench/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-blue"
       alt="MIT License">
</a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <a href="https://arxiv.org/abs/2610.27087">
    <img src="https://img.shields.io/badge/arXiv-2610.27087-b31b1b" alt="Paper on arXiv">
  </a>
  <a href="https://github.com/LeNcyer/SciFigBench/blob/main/assets/SciFigBench%20Poster.pdf">
  <img src="https://img.shields.io/badge/Poster-PDF-D14836"
       alt="SciFigBench Conference Poster">
</a>
  <a href="https://huggingface.co/datasets/LeNcyer/SciFigBench">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-FFD21E" alt="Hugging Face Dataset">
  </a>
  <img src="https://img.shields.io/badge/PyPI-v0.1.0%20Coming%20Soon-blue" alt="PyPI">
</p>

<p align="center">
  <a href="#news">News</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#license-and-citation">License & Citation</a>
</p>

---

<a id="news"></a>

## 🔥 News

- **[Oct. 20, 2026]** 🎉 **SciFigBench is officially released!** Following a second round of human verification, we further strengthened the benchmark's reliability and revised approximately 5.8% of its reference answers.
- **[Aug. 21, 2026]** 🎉 Our paper, **SciFigBench: A Benchmark for Scientific Method Figure Understanding**, was accepted to **EMNLP 2026 Main Conference**!
- **[May 26, 2026]** 📝 We submitted SciFigBench to the **May 2026 ARR cycle**, targeting EMNLP 2026.

<a id="overview"></a>

## 👋 Overview

**SciFigBench** is a diagnostic benchmark for evaluating Vision-Language Models (VLMs) on scientific method figure understanding. It consists of **4,352 real-world scientific method figures** and **40,670 QA pairs**, organized into five fine-grained tasks across three core capabilities: **Dense Content Perception**, **Visual Relation Comprehension**, and **Information-Flow Tracing**.

By separately evaluating text recognition, arrow-induced structural relations, and information-flow reasoning, SciFigBench helps identify where VLMs struggle to understand scientific method figures.

<p align="center">
  <img src="https://raw.githubusercontent.com/LeNcyer/SciFigBench/main/assets/benchmark_overview.png" alt="SciFigBench Benchmark Overview" width="500">
</p>

<p align="center">
  <em>Five diagnostic tasks across three complementary capabilities.</em>
</p>

<div align="center">

| Task | Evaluation Focus | QA Pairs | Metric |
|---|---|---:|---|
| **Dense Content Perception** | Reading text across highlighted regions | 8,367 | Position-wise ANLS |
| **Edge-Level Verification** | Verifying directed edges between nodes | 8,665 | Accuracy |
| **Node-Level Degree Counting** | Counting incoming or outgoing edges | 7,848 | Accuracy |
| **Path-Level Reachability** | Reasoning about directed paths | 8,144 | Accuracy |
| **Information-Flow Tracing** | Tracing direct neighbors through text and connections | 7,646 | Set-based ANLS* |
| **Total** | | **40,670** | |

</div>

SciFigBench also provides human-verified content and structural annotations for further research. The full dataset is available on [Hugging Face](https://huggingface.co/datasets/LeNcyer/SciFigBench).

**Note on reported results.** The paper reports results on an earlier QA version. Retrospective evaluations show modest score changes, with relative model performance trends and main conclusions remaining largely consistent.

<a id="quick-start"></a>

## 🚀 Quick Start

**🤖 Using a Coding Agent?** Follow [AGENTS.md](https://github.com/LeNcyer/SciFigBench/blob/main/AGENTS.md) to quickly set up SciFigBench and evaluate your VLM with tools like Codex, Claude Code, or Cursor.

### 1. Installation

Install SciFigBench from source:

```bash
git clone https://github.com/LeNcyer/SciFigBench.git
cd SciFigBench
pip install .
```

Install via PyPI (Coming Soon):

```bash
pip install scifigbench
```

### 2. One-call Evaluation

Provide your own synchronous callable model with the signature
**model(prompt: str, image: PIL.Image.Image) -> str**. It receives only the
official prompt and rendered RGB image and returns raw text. Wrap your preferred
model SDK yourself; no model vendor SDK is required by SciFigBench.

    from scifigbench import SciFigBench

    # model = MyVLM(...)  # Your callable model.

    # Quick Test (default):
    # Evaluate at most 5 QA per task (25 Callable calls for all five tasks).
    # Results are partial scores, NOT official full Benchmark results.
    result = SciFigBench(model=model, task="all")
    print(result.scores)

Quick Test automatically downloads verified QA and only the selected images on
first use. Installation and import do not download data. Sampling is deterministic
for the same data Commit, seed and sample limit. Use **seed=0** (the default) or
provide another integer.

**Full Evaluation requires explicit confirmation:**

    # Full Evaluation: all 40,670 QA pairs across five tasks.
    # WARNING: May trigger 40,670 model Callable calls and substantial API costs.
    result = SciFigBench(
        model=model,
        task="all",
        max_samples_per_task=None,
        confirm_full_run=True,
        model_id="my-model-version-and-decoding-config",
    )

A single-task full run uses that task's actual QA count (for example, 8,665 for
edge-level verification). Without both full-run settings, a request covering all
selected QA raises an error before any model call. A finite limit covering every
selected QA cannot bypass confirmation. Setting the confirmation flag alone
does not expand the default Quick Test. Interrupted full runs remain incomplete
Partial Evaluation, never completed full Benchmark results.

**Scores and coverage.** The result exposes per-task primary scores in the range
0–1, with no overall five-task score. Its report records **quick / partial / full**,
actual evaluated counts, each task's total QA count, Coverage and sampling-plan
completion. Coverage uses the entire selected task as its denominator. Unassessed
metrics are **None**; actual zero scores remain **0.0**. A single-task full run is
not the complete five-task Benchmark.

**Data and outputs.** The default dataset Commit is
**a884c09770b99e6337c4cb27cee4696773e27d6e**. An explicit branch or tag is resolved
once to a Commit, which is recorded in the report and used for every download.
Data is cached by repository and Commit. An existing **data_root** is reused
when its QA, manifests and required images match; unrelated README changes do
not invalidate evaluation data. Conflicting hashes raise errors without
overwriting existing data. Annotations are not required or downloaded.

Each call creates a unique directory under **./scifigbench_runs/**, or under
the parent specified by **output_dir**. It contains **run.json**,
**predictions.jsonl**, **progress.json**, **report.json**, and **errors.jsonl**
when an error occurs. Predictions contain only QID and raw response and work
with the existing evaluator. Obtain paths from **result.run_dir**,
**result.predictions_path**, and **result.report_path**.

**Errors and resume.** Model exceptions and non-string outputs stop the run,
save completed responses and raise **ModelCallError** with a **run_dir**
attribute. Invalid answer text is still scored by the standard parser; it is
not silently removed from the denominator. No automatic model retries occur.

New runs need no model identifier or storage configuration. To enable resume,
set a stable **model_id** on the first call, including the checkpoint/model
version and decoding configuration. Resume with the same parameters and
**resume_from=previous_run_dir**, omitting **output_dir**. The Toolkit verifies
the model ID, data Commit/hashes, sampling plan and implementation hashes before
reusing predictions. Full-run resume still requires explicit confirmation.
Already saved responses are skipped; a charged response lost before persistence
may need another call. A model's actual weights cannot be inferred from an
arbitrary callable, so provide an accurate identifier. Never include credentials
in it. Only one writer can use a run directory at a time.

Use **local_files_only=True** to avoid network access after the pinned reference
manifests, QA and selected images have been cached. Missing offline inputs fail
clearly. Existing low-level loading/scoring remains local-only.

### 3. Download the Dataset

The benchmark dataset is hosted on [Hugging Face](https://huggingface.co/datasets/LeNcyer/SciFigBench). Download the QA pairs and images from the latest public version:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LeNcyer/SciFigBench",
    repo_type="dataset",
    revision="main",  # Latest public version; this branch can change.
    local_dir="./data",
    allow_patterns=["QA/*", "images/*.jpg", "checksums.sha256"],
)
```

The toolkit also supports dataset downloads:

```bash
scifigbench download --repo-id LeNcyer/SciFigBench --revision main --data-root ./data
```

For reproducibility, replace `revision="main"` above (or CLI `--revision main`)
with the full accepted-data commit `a884c09770b99e6337c4cb27cee4696773e27d6e`.
`QA/*` and `images/*.jpg` supply the standard evaluation inputs; the checksum
files verify their integrity. Optional `annotations/*.json`, `schema.md`, and
`manifest.jsonl` can be added to `allow_patterns` for annotation research and
schema/reference information. They are not required for standard evaluation.

### 4. Run Your VLM

Use your preferred VLM to answer the benchmark questions. SciFigBench provides the official prompts and a rendering utility for generating query-specific visual inputs.

```python
import json
from scifigbench import prepare_model_payload

with open("data/QA/qa_edge_level_verification.jsonl", encoding="utf-8") as f:
    qa = json.loads(next(f))

payload = prepare_model_payload(qa, data_root="./data")

# Pass payload["prompt"] and payload["image"] to your VLM.
```

Save your model's raw responses as a JSONL file, with one prediction per line:

```json
{"qid": "cf32f75bc3d9", "response": "true"}
{"qid": "0aaf24e73a36", "response": "1"}
```

Expected answer text by task:

| Task | Model output format |
|---|---|
| Dense Content Perception | JSON list of strings, in query color order |
| Edge-Level Verification | `true` or `false` |
| Node-Level Degree Counting | Integer |
| Path-Level Reachability | `true` or `false` |
| Information-Flow Tracing | JSON list of strings |

The JSONL `response` field always contains the model's **raw text string**, even
when that text is a JSON list; for example, `"response": "[\"Encoder\", \"Decoder\"]"`.

Models should receive only the task prompt and rendered figure, without reference answers or ground-truth annotations.

### 5. Evaluate Predictions

Evaluate all five tasks using the official SciFigBench metrics:

```bash
scifigbench evaluate \
    --predictions predictions.jsonl \
    --task all \
    --data-root ./data
```

For evaluation on a subset of questions, append `--allow-partial`. The evaluator reports task-specific scores and prediction coverage, clearly distinguishing partial evaluations from full benchmark results. A task with `n_matched=0` has a metric value of JSON `null` (Python `None`); a matched task can legitimately score `0.0`. Only matched questions enter each task's score.

For additional commands and Python API details, refer to [AGENTS.md](https://github.com/LeNcyer/SciFigBench/blob/main/AGENTS.md) or run `scifigbench --help`.

<a id="license-and-citation"></a>

## ✍️ License & Citation

The SciFigBench Evaluation Toolkit is released under the **MIT License**. See [LICENSE](https://github.com/LeNcyer/SciFigBench/blob/main/LICENSE).

If you find our work helpful, please use the following citation.

```bibtex
@inproceedings{wu2026scifigbench,
  title     = {SciFigBench: A Benchmark for Scientific Method Figure Understanding},
  author    = {Wu, Lijia and Li, Zheng and Liu, Yuxuan and Zhang, Jingwen and Zhao, Penghai and Li, Xiang},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  eprint    = {2610.27087},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2610.27087}
}
```
