import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import pytest
from scifigbench import evaluate

def run(*args,cwd):
    return subprocess.run([sys.executable,'-m','scifigbench',*map(str,args)],cwd=cwd,capture_output=True,text=True,encoding='utf-8')

def test_module_console_and_API_equivalence(dataset,predictions,tmp_path):
    root,_=dataset;p=tmp_path/'predictions.jsonl';p.write_text(''.join(json.dumps({'qid':k,'response':v})+'\n' for k,v in predictions.items()),encoding='utf-8')
    expected=evaluate(p,data_root=root)
    proc=run('evaluate','--predictions',p,'--data-root',root,'--task','all',cwd=tmp_path)
    assert proc.returncode==0,proc.stderr
    assert json.loads(proc.stdout)==expected
    out=tmp_path/'reports/report.json'
    console=Path(sysconfig.get_path('scripts'))/('scifigbench.exe' if os.name=='nt' else 'scifigbench')
    proc=subprocess.run([str(console),'evaluate','--predictions',str(p),'--data-root',str(root),'--output',str(out)],cwd=tmp_path,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert json.loads(out.read_text(encoding='utf-8'))==expected

def test_help_version_errors_and_offline_download(dataset,tmp_path):
    root,_=dataset
    for args in [('--help',),('--version',),('evaluate','--help'),('download','--help')]:
        assert run(*args,cwd=tmp_path).returncode==0
    proc=run('download','--local-files-only','--data-root',root,cwd=tmp_path)
    assert proc.returncode==0 and json.loads(proc.stdout)['data_root']==str(root)
    assert run('download',cwd=tmp_path).returncode==2
    assert run('evaluate','--predictions','missing.jsonl','--task','invalid',cwd=tmp_path).returncode==2

def test_partial_CLI_and_explicit_field(dataset,tmp_path):
    root,rows=dataset;p=tmp_path/'partial.jsonl';p.write_text(json.dumps({'qid':rows[0]['qid'],'model_answer':'["Model"]','gt':'DO NOT USE'})+'\n',encoding='utf-8')
    args=['evaluate','--predictions',p,'--data-root',root,'--prediction-field','model_answer']
    assert run(*args,cwd=tmp_path).returncode==2
    proc=run(*args,'--allow-partial',cwd=tmp_path)
    assert proc.returncode==0 and json.loads(proc.stdout)['evaluation_mode']=='Partial Evaluation'
    report = json.loads(proc.stdout)
    assert report['tasks']['edge_level_verification']['primary_metric']['value'] is None
    assert report == evaluate(p, data_root=root, prediction_field='model_answer', allow_partial=True)


@pytest.mark.parametrize('exists', [False, True])
def test_invalid_data_root_CLI_is_actionable_and_fails(tmp_path, exists):
    root = tmp_path / 'wrong-data'
    if exists:
        root.mkdir()
    predictions = tmp_path / 'predictions.jsonl'
    predictions.write_text('{"qid":"example","response":"true"}\n', encoding='utf-8')
    proc = run('evaluate', '--predictions', predictions, '--data-root', root, cwd=tmp_path)
    assert proc.returncode == 2
    assert not proc.stdout
    assert '--data-root' in proc.stderr and 'QA/' in proc.stderr and 'download' in proc.stderr
    assert 'Traceback' not in proc.stderr

def test_import_has_no_network_and_missing_SciPy_cannot_silently_fallback(tmp_path):
    script="import socket; socket.socket.connect=lambda *a: (_ for _ in ()).throw(RuntimeError('network forbidden')); import scifigbench; print(scifigbench.__version__)"
    assert subprocess.run([sys.executable,'-c',script],cwd=tmp_path,capture_output=True).returncode==0
    blocked="import builtins; original=builtins.__import__; exec(\"def guarded(name,*a,**k):\\n if name.startswith('scipy'): raise ImportError('blocked')\\n return original(name,*a,**k)\"); builtins.__import__=guarded; import scifigbench"
    proc=subprocess.run([sys.executable,'-c',blocked],cwd=tmp_path,capture_output=True,text=True)
    assert proc.returncode!=0 and 'requires SciPy' in proc.stderr
