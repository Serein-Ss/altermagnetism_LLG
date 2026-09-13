from argparse import Namespace
from pathlib import Path
import json
import pytest
import yaml

from scripts.literature import workflow
from scripts.workflow.submit_remaining_validation import task_specs


def test_validation_matrix():
    tasks=task_specs()
    assert len(tasks)==65
    assert sum(t['pool']=='cpu' for t in tasks)==32
    assert sum(t['pool']=='gpu' for t in tasks)==33
    assert len({t['paper'] for t in tasks})==4
    assert all(t['protocol']!='smoke' for t in tasks)


def config(tmp_path,allowed):
    source=Path(workflow.__file__).resolve().parents[2]/'conf/literature/bauer_2011.yaml'
    doc=yaml.safe_load(source.read_text())
    doc['numerics']['validation_extension']['validation_protocols']=allowed
    dest=tmp_path/'input.yaml';dest.write_text(yaml.safe_dump(doc))
    return Namespace(config=dest,protocol='trajectory',run_id='test_validation',
                     validation=True,project_root=tmp_path,device='cpu')


def test_validation_requires_explicit_config_opt_in(tmp_path):
    with pytest.raises(RuntimeError,match='explicitly listed'):
        workflow.prepare(config(tmp_path,[]),'bauer_2011')


def test_validation_output_isolated_and_atomic(tmp_path):
    args=config(tmp_path,['trajectory'])
    r,p,raw,out,manifest=workflow.prepare(args,'bauer_2011')
    assert raw.name=='test_validation.partial'
    assert manifest['configuration']['numerics']['validation_extension']['production_enabled'] is False
    (raw/'example.json').write_text('{}')
    workflow.complete(raw,out,manifest)
    assert not raw.exists()
    assert raw.with_name('test_validation').is_dir()
    assert json.loads((out/'manifest.json').read_text())['stage']=='numerical_validation'
    with pytest.raises(FileExistsError): workflow.prepare(args,'bauer_2011')


def test_validation_opt_in_does_not_unlock_production(tmp_path):
    args=config(tmp_path,['trajectory']);args.validation=False
    with pytest.raises(RuntimeError,match='Production disabled'):
        workflow.prepare(args,'bauer_2011')
