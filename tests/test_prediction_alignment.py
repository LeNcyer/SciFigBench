import json
import pytest
from scifigbench import evaluate,load_predictions,PredictionError,DataError

def test_full_and_explicit_partial_denominators(dataset,predictions):
    root,rows=dataset;r=evaluate(predictions,data_root=root)
    assert r['evaluation_mode']=='Full Evaluation' and r['n_matched']==len(rows) and r['coverage_pct']==100
    assert not any(k in r for k in ['overall_score','average_score'])
    one={rows[0]['qid']:predictions[rows[0]['qid']]}
    with pytest.raises(PredictionError,match='--allow-partial'):evaluate(one,data_root=root)
    r=evaluate(one,data_root=root,allow_partial=True,include_details=True)
    assert r['evaluation_mode']=='Partial Evaluation' and r['n_missing_pred']==len(rows)-1
    assert r['tasks']['dense_content_perception']['n_total']==1
    assert r['tasks']['dense_content_perception']['primary_metric']=={'name':'Position-wise ANLS','value':1}
    assert len(r['per_question'])==1

@pytest.mark.parametrize('rows',[
    [],[{'qid':'x','response':'true'},{'qid':'x','response':'false'}],[{'qid':1,'response':'true'}],
    [{'qid':'x','response':True}],[{'qid':'x','answer':'true'}],[{'qid':'x','gt':'true'}],
    [{'qid':'x','gt_answer':'true'}],[{'qid':'x','response':'true'},{'qid':'y','prediction':'false'}],
])
def test_ambiguous_missing_duplicate_and_wrong_types(rows):
    with pytest.raises((PredictionError,DataError)):load_predictions(rows)

def test_unknown_task_unknown_QID_and_explicit_legacy_field(dataset):
    root,rows=dataset
    with pytest.raises(PredictionError,match='Unknown'):evaluate({'unknown':'true'},data_root=root,allow_partial=True)
    with pytest.raises(ValueError,match='Unknown task'):evaluate({'x':'true'},data_root=root,task='topo_task_a')
    assert load_predictions([{'qid':'x','model_answer':'raw','gt_answer':'GT'}],prediction_field='model_answer')=={'x':'raw'}

def test_unparseable_text_keeps_matched_denominator(dataset,predictions):
    root,rows=dataset;predictions[rows[0]['qid']]='bad output'
    r=evaluate(predictions,data_root=root)
    d=r['tasks']['dense_content_perception']
    assert d['n_total']==2 and d['n_parse_fail']==1 and d['primary_metric']['value']==.5


@pytest.mark.parametrize('matched_task,wrong_response', [
    ('dense_content_perception', '["Wrong"]'),
    ('edge_level_verification', 'false'),
    ('node_level_degree_counting', '999'),
    ('path_level_reachability', 'true'),
    ('information_flow_tracing', '[]'),
])
def test_partial_unassessed_tasks_are_distinct_from_scored_zero(dataset, matched_task, wrong_response):
    root, rows = dataset
    question = next(q for q in rows if q['task'] == matched_task)
    report = evaluate({question['qid']: wrong_response}, data_root=root,
                      allow_partial=True, include_details=True)
    assert report['n_matched'] == 1
    assert len(report['per_question']) == 1
    for name, task_report in report['tasks'].items():
        matched = name == matched_task
        assert task_report['n_matched'] == int(matched)
        assert task_report['n_missing_pred'] == task_report['n_qa_total'] - int(matched)
        assert task_report['coverage'] == int(matched) / task_report['n_qa_total']
        value = task_report['primary_metric']['value']
        if matched:
            assert value == 0.0 and isinstance(value, float)
        else:
            assert value is None
        if name == 'dense_content_perception':
            assert task_report['additional_metrics']['set-ANLS*']['value'] == (0.0 if matched else None)
    encoded = json.dumps(report, allow_nan=False)
    assert 'null' in encoded and json.loads(encoded) == report
