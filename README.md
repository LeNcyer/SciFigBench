<h1 align="center">🔬 SciFigBench</h1>

<p align="center">
  <strong>Evaluating VLMs' Understanding of Scientific Method Figures.</strong>
</p>

<p align="center">
  <a href="LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-blue"
       alt="MIT License">
</a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Paper-Coming%20Soon-grey" alt="Paper">
  <a href="assets/SciFigBench%20Poster.pdf">
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
  <img src="assets/benchmark_overview.png" alt="SciFigBench Benchmark Overview" width="500">
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

**🤖 Using a Coding Agent?** Follow [AGENTS.md](AGENTS.md) to quickly set up SciFigBench and evaluate your VLM with tools like Codex, Claude Code, or Cursor.

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

### 2. Download the Dataset

The benchmark dataset is hosted on [Hugging Face](https://huggingface.co/datasets/LeNcyer/SciFigBench). Download the QA pairs and images:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LeNcyer/SciFigBench",
    repo_type="dataset",
    local_dir="./data",
    allow_patterns=["QA/*", "images/*.jpg", "checksums.sha256"],
)
```

The toolkit also supports dataset downloads:

```bash
scifigbench download --repo-id LeNcyer/SciFigBench --revision main --data-root ./data
```

Use a commit SHA as `--revision` for reproducibility. Ground-truth annotations are optional and not required for standard evaluation.

### 3. Run Your VLM

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

Models should receive only the task prompt and rendered figure, without reference answers or ground-truth annotations.

### 4. Evaluate Predictions

Evaluate all five tasks using the official SciFigBench metrics:

```bash
scifigbench evaluate \
    --predictions predictions.jsonl \
    --task all \
    --data-root ./data
```

For evaluation on a subset of questions, append `--allow-partial`. The evaluator reports task-specific scores and prediction coverage, clearly distinguishing partial evaluations from full benchmark results.

For additional commands and Python API details, refer to [AGENTS.md](AGENTS.md) or run `scifigbench --help`.

<a id="license-and-citation"></a>

## ✍️ License & Citation

The SciFigBench Evaluation Toolkit is released under the **MIT License**. See [LICENSE](LICENSE).

If you find our work helpful, please use the following citation.

```bibtex
@inproceedings{wu2026scifigbench,
  title     = {SciFigBench: A Benchmark for Scientific Method Figure Understanding},
  author    = {Wu, Lijia and Li, Zheng and Liu, Yuxuan and Zhang, Jingwen and Zhao, Penghai and Li, Xiang},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```