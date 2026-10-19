import json
from pathlib import Path
from tests.experiments.stable_independent_reference import cases


def test_frozen_reference_cases_and_noise_disjointness():
    cfg=json.loads(Path('tests/conf/stable_independent_reference.json').read_text())
    rows=cases(cfg)
    assert rows==cases(cfg) and len(rows)==12
    assert {r['case_id'] for r in rows}==set(range(12))
    assert {(r['theta'],r['family']) for r in rows if r['role']=='primary'}=={(t,i) for t in (.11,.13) for i in range(4)}
    assert {(r['theta'],r['method'],r['dt'],r['family']) for r in rows if r['role']=='numerical_control'}=={(t,m,.01,0) for t in (.11,.13) for m in ('mt','heun')}
    seeds=[s for r in rows for s in range(r['noise_start'],r['noise_start']+2*cfg['n_per_half'])]
    assert len(seeds)==len(set(seeds))==3072
