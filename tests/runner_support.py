"""Small public fixtures for offline runner tests and installed-package smoke checks."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import fnmatch
import hashlib
import json
import shutil

from PIL import Image

TASK_NAMES = (
    "dense_content_perception", "edge_level_verification", "node_level_degree_counting",
    "path_level_reachability", "information_flow_tracing",
)
COMMIT = "a" * 40
REPO_ID = "tests/benchmark"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refresh_manifests(root):
    files = sorted((root / "QA").glob("qa_*.jsonl"))
    tasks = {}
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        tasks[path.name] = {"total_questions": len(rows), "sha256": digest(path)}
    manifest = {"version": "public-synthetic-fixture", "tasks": tasks,
                "total_questions": sum(e["total_questions"] for e in tasks.values())}
    (root / "QA/manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (root / "QA/checksums.sha256").write_text(
        "".join(digest(p) + "  " + p.name + "\n" for p in files), encoding="utf-8",
    )
    refresh_root_checksums(root)


def refresh_root_checksums(root):
    (root / "checksums.sha256").write_text("".join(
        digest(p) + "  " + p.relative_to(root).as_posix() + "\n"
        for p in sorted(root.rglob("*")) if p.is_file() and p.name != "checksums.sha256"
    ) + "".join(
        digest(p) + "  " + p.relative_to(root).as_posix() + "\n"
        for p in sorted(root.glob("QA/checksums.sha256"))
    ), encoding="utf-8")


def make_runner_dataset(root, *, counts=None, shared_image=False, images=True):
    root = Path(root)
    (root / "QA").mkdir(parents=True)
    counts = counts or {task: 6 for task in TASK_NAMES}
    rows = []
    image_ids = set()
    index = 0
    for ti, task in enumerate(TASK_NAMES):
        task_rows = []
        for qi in range(counts[task]):
            index += 1
            image_id = 1 if shared_image else index
            image_ids.add(image_id)
            q = {
                "qid": f"{ti + 1:x}{qi:011x}", "task": task,
                "sample_id": f"F{image_id:06d}", "image_path": f"images/F{image_id:06d}.jpg",
                "image_size": [180, 120], "prompt": "Read the figure. Return the requested answer.",
            }
            if ti == 0:
                q.update(answer=["REFERENCE-ONLY"], boxes_xyxy=[[10, 10, 40, 30]], colors=["red"])
            elif ti in (1, 3):
                q.update(answer="true", red_box_xyxy=[10, 10, 40, 30],
                         blue_box_xyxy=[100, 60, 150, 95], meta={"src_node_id": 0, "dst_node_id": 1})
            elif ti == 2:
                q.update(answer="2", red_box_xyxy=[10, 10, 40, 30],
                         meta={"target_node_id": 0, "variant": "in"})
            else:
                q.update(answer='["REFERENCE-ONLY"]',
                         meta={"target_node_id": 0, "target_text": "Target", "variant": "in"})
            task_rows.append(q)
        path = root / "QA" / f"qa_{task}.jsonl"
        path.write_text("".join(json.dumps(q) + "\n" for q in task_rows), encoding="utf-8")
        rows.extend(task_rows)
    if images:
        (root / "images").mkdir()
        for image_id in sorted(image_ids):
            with Image.new("RGB", (180, 120), "white") as image:
                image.save(root / f"images/F{image_id:06d}.jpg")
    (root / "README.md").write_text("Fixture documentation.\n", encoding="utf-8")
    refresh_manifests(root)
    return root, rows


class HubStub:
    """Simulate immutable Hub snapshots and an initially empty download cache."""

    def __init__(self, source, cache):
        self.source, self.cache = Path(source), Path(cache)
        self.requests = []
        self.snapshot_calls = []
        self.api_calls = []
        self.commit = COMMIT

    def repo_info(self, **kwargs):
        self.api_calls.append(kwargs)
        return SimpleNamespace(sha=self.commit)

    def hf_hub_download(self, **kwargs):
        target = self.cache / kwargs["revision"] / kwargs["filename"]
        if target.is_file():
            return str(target)
        if kwargs["local_files_only"]:
            raise FileNotFoundError("Not cached")
        self.requests.append(kwargs)
        source = self.source / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return str(target)

    def snapshot_download(self, **kwargs):
        self.snapshot_calls.append(kwargs)
        assert kwargs["revision"] == self.commit
        dest = Path(kwargs["local_dir"])
        for source in self.source.rglob("*"):
            rel = source.relative_to(self.source).as_posix()
            if source.is_file() and any(fnmatch.fnmatch(rel, p) for p in kwargs["allow_patterns"]):
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return str(dest)

    def module(self):
        return SimpleNamespace(
            hf_hub_download=self.hf_hub_download,
            snapshot_download=self.snapshot_download,
            HfApi=lambda: self,
        )
