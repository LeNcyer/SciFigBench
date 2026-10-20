import json
import pytest
from scifigbench.scoring.parsing import parse_json_list, parse_bool, parse_int, _strip_md_fence

@pytest.mark.parametrize("raw,expected", [
    ('["```literal```","tail"]',["```literal```","tail"]),
    ('```json\n["x"]\n```',["x"]),('```\n["x"]\n```',["x"]),
    ('```json\n["```literal```","tail"]\n```',["```literal```","tail"]),
    ('````json\n["```literal```"]\n````',["```literal```"]),
    ('["\\u0060\\u0060\\u0060"]',["```"]),
    (' \r\n\t["x"]\t\n ',["x"]),('```json\n["x"]',["x"]),('["x"]\n```',["x"]),
    ('["x]',None),('```json\n["x]\n```',None),('Answer: ["x"] Thanks.',["x"]),
    ('```json ["x"] ```',["x"]),('{"answer":["x",2,null]}',["x","2","None"]),
    ('[]',[]),('{"x":1}',None),('true',None),('42',None),('',None),('null',None),(None,None),
])
def test_approved_json_cases(raw,expected):
    assert parse_json_list(raw)==expected

def test_escaped_content_and_legacy_item_coercion():
    value=["line\nbreak",r"\nabla \frac{a}{b}",r"\\","α","```"]
    assert parse_json_list(json.dumps(value,ensure_ascii=True))==value
    mixed=[1,True,None,{"x":1},["a"]]
    assert parse_json_list(json.dumps(mixed))==[str(x) for x in mixed]

@pytest.mark.parametrize("text,expected", [
    ('```json\n["x"]\n```','["x"]'),('```\n["x"]\n```','["x"]'),
    ('["```x```"]','["```x```"]'),('prefix ```json\n["x"]\n``` suffix','prefix ```json\n["x"]\n``` suffix'),
    ('```json\n["x"]','```json\n["x"]'),('````json\n["x"]\n```','````json\n["x"]\n```'),
])
def test_only_outer_complete_fences_are_removed(text,expected):
    assert _strip_md_fence(text)==expected

@pytest.mark.parametrize("raw,expected",[(None,None),("",None),("true","true"),("FALSE","false"),("Answer: true","true"),("untrue",None),("```json\nfalse\n```","false")])
def test_boolean_semantics(raw,expected):
    assert parse_bool(raw)==expected

@pytest.mark.parametrize("raw,expected",[(None,None),("",None),("42",42),("-7",-7),("Answer: +3",3),("3.9",3),("none",None)])
def test_integer_semantics(raw,expected):
    assert parse_int(raw)==expected
