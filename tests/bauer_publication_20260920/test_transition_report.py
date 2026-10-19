import json
import h5py
import numpy as np
import pytest
from tests.bauer_publication_20260920 import transition_report as report


def test_complete_archive_and_time_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'EXPERIMENT', tmp_path)
    monkeypatch.setattr(report, 'REFERENCE', tmp_path/'reference')
    folder = tmp_path/'evaluation'/'test'; folder.mkdir(parents=True)
    (folder/'complete.json').write_text(json.dumps({'model': 'test'}))
    refdir = tmp_path/'reference'/'L8'; refdir.mkdir(parents=True)
    rng = np.random.default_rng(5)
    with h5py.File(folder/'L8.h5', 'w') as g:
        g.attrs.update(complete=True, model='test')
        g['time_ps'] = np.arange(21)*.1
        for parent in range(6):
            r = rng.normal(size=(32,21,8,3)); r[:,0] = 0
            g[f'parent_{parent}'] = r[:16]
            with h5py.File(refdir/f'test_future_{parent}.h5', 'w') as h:
                h.attrs['complete'] = True
                h['spins'] = r; h['seeds'] = np.arange(32); h['time_ps'] = np.arange(21)*.1
    cfg = dict(evaluation_sizes=[8], evaluation_duration_ps=2., save_ps=.1, branches=16)
    rows = report.collect_model('test', cfg)
    assert len(rows) == 12
    assert all(r['model_vs_reference_a']['increment_power_relative'] == 0 for r in rows)
    with h5py.File(folder/'L8.h5', 'r+') as g:
        g['time_ps'][-1] = 3.
    with pytest.raises(ValueError, match='Time grid'):
        report.collect_model('test', cfg)
