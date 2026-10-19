from copy import deepcopy
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from tests.experiments.real_anchor_analysis import validate_report, main
from scripts.workflow.bauer_campaign import save, digest


def test_complete_batch_and_duplicates():
    cfg = dict(start_frames=[50, 150])
    report = dict(status='COMPLETE_REAL_ANCHOR_PROBE_NOT_CERTIFIED', config=cfg,
                  rows=[dict(family=f, theta=t, start_frame=s)
                        for f in range(4) for t in (.11, .13) for s in cfg['start_frames']])
    validate_report(report, cfg)
    bad = deepcopy(report); bad['rows'].pop()
    with pytest.raises(ValueError): validate_report(bad, cfg)
    bad = deepcopy(report); bad['rows'][-1] = bad['rows'][0]
    with pytest.raises(ValueError): validate_report(bad, cfg)
    bad = deepcopy(report); bad['status'] = 'RUNNING'
    with pytest.raises(ValueError): validate_report(bad, cfg)
    with pytest.raises(ValueError): validate_report(report, dict(start_frames=[40, 150]))


def test_end_to_end_all_48_comparisons(tmp_path, monkeypatch):
    source = Path('tests/experiments/restart_observables.py').resolve()
    monkeypatch.chdir(tmp_path)
    source_copy = Path('tests/experiments/restart_observables.py')
    source_copy.parent.mkdir(parents=True)
    shutil.copyfile(source, source_copy)
    save('audit.json', {'fixture': True})
    cfg = dict(start_frames=[50, 150], training_seeds=[11, 22, 33],
               n_parents=2, frames=11, audit='audit.json')
    cfg_path = Path('tests/conf/real_anchor_probe.json'); save(cfg_path, cfg)
    # Software fixture only: each continuation exactly equals its reference.
    spins = np.zeros((2, 201, 25, 3), dtype='f4'); spins[..., 2] = 1
    spins[1] *= -1
    np.savez_compressed('reference.npz', spins=spins, reduced_time=np.arange(201)*80)
    np.savez_compressed('generated.npz', spins=spins[:, :11],
                        parent_index=np.arange(2), reduced_time=np.arange(11)*80)
    for seed in cfg['training_seeds']:
        rows = [dict(family=f, theta=t, start_frame=s, path='generated.npz',
                     sha256=digest('generated.npz'), reference_path='reference.npz',
                     reference_sha256=digest('reference.npz'))
                for f in range(4) for t in (.11, .13) for s in cfg['start_frames']]
        save(f'tests/results/real_anchor_probe/{seed}/summary.json',
             dict(status='COMPLETE_REAL_ANCHOR_PROBE_NOT_CERTIFIED', config=cfg, rows=rows,
                  config_sha256=digest(cfg_path), audit_sha256=digest('audit.json')))
    main()
    out = Path('tests/results/real_anchor_analysis')
    report = json.loads((out/'summary.json').read_text())
    assert report['status'] == 'COMPLETE_RESTART_MOMENTS_NOT_CERTIFIED'
    assert len(report['rows']) == 48 and len(list(out.glob('*.npz'))) == 48
    for row in report['rows']:
        assert digest(row['per_parent_path']) == row['sha256']
        with np.load(row['per_parent_path']) as data:
            np.testing.assert_array_equal(data['parent_index'], [0, 1])
            assert data['generated/magnetization'].shape == (2, 11, 3)
            for key in data.files:
                if key.startswith('differences/'):
                    np.testing.assert_array_equal(data[key], np.zeros_like(data[key]))
    with pytest.raises(FileExistsError):
        main()
