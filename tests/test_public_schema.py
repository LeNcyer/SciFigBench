import copy
import json
import pytest
from scifigbench import load_qa,DataError
from scifigbench.data import read_jsonl,validate_qa

def test_all_five_public_files_load_without_annotations(dataset):
    root,rows=dataset
    assert load_qa(root)==rows
    assert not (root/'annotations').exists()

def test_QA_only_loading_and_scoring(dataset_factory):
    from scifigbench import evaluate
    root,rows=dataset_factory(images=False)
    pred={q['qid']:json.dumps(q['answer']) if q['task']=='dense_content_perception' else q['answer'] for q in rows}
    r=evaluate(pred,data_root=root)
    assert r['n_matched']==len(rows) and r['coverage']==1
    assert all(t['primary_metric']['value']==1 for t in r['tasks'].values())

@pytest.mark.parametrize("change",[
    lambda q:q.pop('qid'),lambda q:q.update(extra_private='leak'),lambda q:q.update(subtask='single'),
    lambda q:q.update(image_path='../escape.jpg'),lambda q:q.update(image_size=[0,120]),
    lambda q:q.update(boxes_xyxy=[[0,0,float('nan'),10]]),lambda q:q.update(colors=['blue']),
    lambda q:q.update(answer='Model'),lambda q:q.update(task='unknown'),lambda q:q.update(task=[]),
])
def test_bad_public_rows_are_clear_errors(dataset,change):
    q=copy.deepcopy(dataset[1][0]);change(q)
    with pytest.raises(DataError):validate_qa(q)

def test_bad_JSON_non_object_empty_and_missing_file(tmp_path):
    p=tmp_path/'bad.jsonl'
    for text in ['\n','{bad}\n','[]\n']:
        p.write_text(text,encoding='utf-8')
        with pytest.raises(DataError):read_jsonl(p)
    with pytest.raises(FileNotFoundError):read_jsonl(tmp_path/'missing')
    p.write_bytes(b'\xff\xfe\x80')
    with pytest.raises(DataError,match='UTF-8'):read_jsonl(p)

def test_UTF8_blank_lines_and_duplicate_QA(dataset):
    root,rows=dataset;p=root/'QA/qa_dense_content_perception.jsonl'
    text=p.read_text(encoding='utf-8');p.write_text('\n'+text+'\n',encoding='utf-8')
    assert load_qa(root)[0]['prompt']==rows[0]['prompt']
    p.write_text(text+json.dumps(rows[0])+'\n',encoding='utf-8')
    with pytest.raises(DataError,match='Duplicate QA'):load_qa(root)
