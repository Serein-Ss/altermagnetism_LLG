from collections import Counter
import json
import pytest
from tests.miltat_library_20260920.generate import HERE,run
from tests.miltat_library_20260920.data import validate_manifest,MiltatWindows


def test_frozen_assignment_counts_and_unique_streams():
    rows=json.loads((HERE/'layout.json').read_text())['parents']
    assert len(rows)==60
    for shape in [(64,38),(96,56),(128,74)]:
        assert Counter(r['split'] for r in rows if tuple(r['cell_shape'])==shape)==dict(train=12,dev=4,test=4)
    seeds=[s for r in rows for s in [r['preparation_seed'],*r['noise_seeds']]]
    assert len(seeds)==540 and len(set(seeds))==540


def test_end_to_end_smoke_is_not_silently_used_for_training():
    root=HERE/'smoke_library'
    with pytest.raises(ValueError,match='smoke'):validate_manifest(root)
    rows=validate_manifest(root,allow_smoke=True)
    ds=MiltatWindows(root,rows[0]['split'],.005,allow_smoke=True)
    assert ds[0]['spins'].shape==(3,6,4,2,3)
    assert ds[0]['time_ps'][-1]==pytest.approx(.005)


def test_formal_generation_requires_release(monkeypatch,tmp_path):
    import tests.miltat_library_20260920.generate as module
    monkeypatch.setattr(module,'HERE',tmp_path)
    monkeypatch.setattr(module,'allocation',lambda device:None)
    with pytest.raises(FileNotFoundError):run('cold',0,'cpu')
