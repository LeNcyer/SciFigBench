import hashlib
import json
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

TASKS = ["dense_content_perception", "edge_level_verification", "node_level_degree_counting", "path_level_reachability", "information_flow_tracing"]

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")

def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def synthetic_dataset(root, images=True, annotations=False):
    rows = []
    def row(task, index, answer, **extra):
        q = dict(qid=f"{index:012x}", task=task, sample_id="F000001", image_path="images/F000001.jpg", image_size=[180,120], prompt="Read α → β.\nReturn the requested answer.", answer=answer, **extra)
        rows.append(q)
    red, blue = [10,10,40,30], [100,60,150,95]
    row(TASKS[0],1,["Model"],boxes_xyxy=[red],colors=["red"])
    row(TASKS[0],2,["A","B"],boxes_xyxy=[red,blue],colors=["red","orange"],meta={"node_ids":[0,1]})
    row(TASKS[1],3,"true",red_box_xyxy=red,blue_box_xyxy=blue,meta={"src_node_id":0,"dst_node_id":1})
    row(TASKS[2],4,"2",red_box_xyxy=red,meta={"target_node_id":0,"variant":"in"})
    row(TASKS[2],5,"0",red_box_xyxy=red,meta={"target_node_id":0,"variant":"out"})
    row(TASKS[3],6,"false",red_box_xyxy=red,blue_box_xyxy=blue,meta={"src_node_id":0,"dst_node_id":1})
    row(TASKS[4],7,json.dumps(["Same","Same"]),meta={"target_node_id":0,"target_text":"Target","variant":"in"})
    row(TASKS[4],8,"[]",meta={"target_node_id":0,"target_text":"Target","variant":"out"})
    for task in TASKS:
        write_rows(root/"QA"/f"qa_{task}.jsonl", [r for r in rows if r["task"]==task])
    if images:
        p = root/"images/F000001.jpg";p.parent.mkdir(parents=True)
        image=Image.new("RGB",(180,120),"white");draw=ImageDraw.Draw(image);draw.line((40,20,100,80),fill="black",width=2);image.save(p)
    if annotations:
        p=root/"annotations/F000001.json";p.parent.mkdir(parents=True);p.write_text('{"sample_id":"F000001"}\n',encoding="utf-8")
    manifest=dict(version="synthetic-test-fixture",total_questions=len(rows),total_images=1,tasks={})
    for task in TASKS:
        p=root/"QA"/f"qa_{task}.jsonl";manifest["tasks"][p.name]=dict(total_questions=sum(r["task"]==task for r in rows),sha256=hash_file(p))
    (root/"QA/manifest.json").write_text(json.dumps(manifest)+"\n",encoding="utf-8")
    (root/"QA/checksums.sha256").write_text("".join(hash_file(p)+"  "+p.name+"\n" for p in sorted((root/"QA").glob("qa_*.jsonl"))),encoding="utf-8")
    (root/"checksums.sha256").write_text("".join(hash_file(p)+"  "+p.relative_to(root).as_posix()+"\n" for p in sorted(root.rglob("*")) if p.is_file()),encoding="utf-8")
    return root,rows

@pytest.fixture
def dataset_factory(tmp_path):
    def make(name="data", **kwargs):
        return synthetic_dataset(tmp_path/name, **kwargs)
    return make

@pytest.fixture
def dataset(dataset_factory):
    return dataset_factory()

@pytest.fixture
def predictions(dataset):
    _,rows=dataset
    return {q["qid"]:json.dumps(q["answer"]) if q["task"]==TASKS[0] else q["answer"] for q in rows}

@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    for name in ["SCIFIGBENCH_DATA_DIR","SCIFIGBENCH_REPO_ID","SCIFIGBENCH_REVISION"]:
        monkeypatch.delenv(name, raising=False)
