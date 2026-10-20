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
