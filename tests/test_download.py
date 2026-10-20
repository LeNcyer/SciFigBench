import fnmatch
import shutil
from pathlib import Path
from types import SimpleNamespace
import sys
import inspect
import pytest
from scifigbench import download_data,DownloadError
from scifigbench.download import verify_download

def hub_stub(monkeypatch,source,calls):
    def snapshot_download(**kwargs):
        calls.append(kwargs);dest=Path(kwargs['local_dir']);dest.mkdir(parents=True,exist_ok=True)
        for p in source.rglob('*'):
            if p.is_file() and any(fnmatch.fnmatch(p.relative_to(source).as_posix(),pattern) for pattern in kwargs['allow_patterns']):
                target=dest/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        return str(dest)
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=snapshot_download))

@pytest.mark.parametrize('mode',[{}, {'qa_only':True}, {'annotations':True}])
def test_explicit_revision_download_subsets_and_cache(dataset_factory,tmp_path,monkeypatch,mode):
    source,_=dataset_factory('hub-source',annotations=True);calls=[];hub_stub(monkeypatch,source,calls)
    root=download_data(repo_id='tests/synthetic-fixture',revision='fixed-test-revision',local_dir=tmp_path/'cache',cache_dir=tmp_path/'hub-cache',**mode)
    assert calls[0]['repo_type']=='dataset' and calls[0]['revision']=='fixed-test-revision'
    assert calls[0]['cache_dir']==str(tmp_path/'hub-cache')
    assert (root/'images').exists() != mode.get('qa_only',False)
    assert (root/'annotations').exists()==mode.get('annotations',False)
    verify_download(root,**mode)
    assert download_data(repo_id='tests/synthetic-fixture',revision='fixed-test-revision',local_dir=root,**mode)==root
    assert len(calls)==1
    assert download_data(local_dir=root,local_files_only=True,**mode)==root and len(calls)==1

def test_no_invented_defaults_and_no_annotations_required(dataset,tmp_path,monkeypatch):
    root,_=dataset
    with pytest.raises(DownloadError,match='No published'):download_data(local_dir=tmp_path/'empty')
    assert download_data(local_dir=root,local_files_only=True)==root
    with pytest.raises(DownloadError,match='mutually exclusive'):download_data(annotations=True,qa_only=True)

def test_integrity_failure_and_revision_mismatch(dataset_factory,tmp_path,monkeypatch):
    source,_=dataset_factory('source');calls=[];hub_stub(monkeypatch,source,calls)
    root=download_data(repo_id='tests/fixture',revision='v1',local_dir=tmp_path/'dest')
    with pytest.raises(DownloadError,match='revision'):download_data(repo_id='tests/fixture',revision='v2',local_dir=root,local_files_only=True)
    p=root/'images/F000001.jpg';p.write_bytes(b'corrupted')
    with pytest.raises(DownloadError,match='SHA256'):download_data(local_dir=root,local_files_only=True)
    assert len(calls)==1

def test_missing_download_file_clear_error(dataset_factory,tmp_path,monkeypatch):
    source,_=dataset_factory('source');(source/'images/F000001.jpg').unlink();calls=[];hub_stub(monkeypatch,source,calls)
    with pytest.raises(FileNotFoundError,match='required file'):download_data(repo_id='tests/fixture',revision='v1',local_dir=tmp_path/'dest')

def test_installed_official_Hub_API_accepts_the_used_arguments():
    from huggingface_hub import snapshot_download
    inspect.signature(snapshot_download).bind(repo_id='tests/synthetic-fixture',repo_type='dataset',revision='fixed-test-revision',
        local_dir='unused',cache_dir=None,allow_patterns=['QA/*.jsonl'],local_files_only=True)
