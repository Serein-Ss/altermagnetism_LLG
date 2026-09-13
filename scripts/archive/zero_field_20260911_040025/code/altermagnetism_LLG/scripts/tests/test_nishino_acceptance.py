import copy
import json
from pathlib import Path
import pytest
from scripts.workflow.resume_nishino_primary import accepted_report

ROOT=Path(__file__).resolve().parents[2]


def test_acceptance_is_specific_and_fail_closed():
    report=json.loads((ROOT/'output/literature_reproduction/nishino_miyashita_2015/20260909_8df33ac_bb48dc64/report.json').read_text())
    assert accepted_report(report)==1
    assert report['status']=='fail'
    for kind in ['mean','cdf','norm','paired','identity']:
        altered=copy.deepcopy(report)
        if kind=='mean': altered['rows'][0]['mean_check']['passed']=False
        if kind=='cdf': altered['rows'][0]['cdf_check']['passed']=False
        if kind=='norm': altered['rows'][0]['norm_error']=1e-3
        if kind=='paired': altered['paired_convergence'][0]['passed']=False
        if kind=='identity': altered['run_id']='unreviewed_run'
        with pytest.raises(ValueError): accepted_report(altered)
