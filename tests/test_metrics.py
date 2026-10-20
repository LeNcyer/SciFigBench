import copy
import json
import pytest
from scifigbench.evaluation import score_question,scoring_view
from scifigbench.scoring._latex_norm import anls,set_anls,_self_test

@pytest.mark.parametrize("raw,expected",[('["Model"]',1),('["Model","EXTRA"]',1),('[]',0),('invalid',0),('"Model"',0),('["wrong"]',0)])
def test_single_historical_length_semantics(dataset,raw,expected):
    q=dataset[1][0]
    q={**q,"subtask":"flow_path","K":99,"meta":{"K":99}}
    assert scoring_view(q)["subtask"]=="single"
    assert score_question(q,raw)["score"]==expected

@pytest.mark.parametrize("raw,expected",[('["A","B"]',1),('["A"]',.5),('["A","B","EXTRA"]',2/3),('[]',0),('invalid',0),('["B","A"]',0)])
def test_multi_historical_length_and_position(dataset,raw,expected):
    q=dataset[1][1]
    assert scoring_view(q)["subtask"]=="flow_path"
    assert score_question(q,raw)["score"]==pytest.approx(expected)

@pytest.mark.parametrize("raw,expected",[('["Same","Same"]',1),('["Same"]',.5),('["Same","Same","EXTRA"]',2/3),('[]',0),('invalid',0)])
def test_IFT_multiplicity(dataset,raw,expected):
    q=dataset[1][-2]
    assert score_question(q,raw)["score"]==pytest.approx(expected)

def test_empty_IFT_and_frozen_failure_behavior(dataset):
    q=dataset[1][-1]
    assert score_question(q,'[]')["score"]==1
    failed=score_question(q,'not JSON')
    assert failed["parse_ok"] is False and failed["score"]==1

@pytest.mark.parametrize("text",['``` "dog on a rocky shore" ```','```python def f(x): return x ```',r'Prompt ``` \`{}``python ``` return x','Template ```python model() ```'])
def test_four_backtick_content_patterns(dataset,text):
    dcp=copy.deepcopy(dataset[1][0]);dcp["answer"]=[text]
    ift=copy.deepcopy(dataset[1][-2]);ift["answer"]=json.dumps([text])
    for q in [dcp,ift]:
        result=score_question(q,json.dumps([text]))
        assert result["parse_ok"] and result["score"]==1

def test_normalization_and_frozen_self_tests():
    assert anls(r'x^{2}',r'x^2')==1
    assert anls(r'\frac{a}{b}',r'\dfrac{a}{b}')==1
    _self_test()

def test_exact_Hungarian_assignment_above_fallback_threshold():
    exact=[c*8 for c in 'qrstuvw']
    assert set_anls(['aaaaa','aaaab']+exact,['aaaac','baaaa']+exact)==pytest.approx(8.6/9)
