"""Tiny CPU prerequisite workflow checks, never a physical certification."""
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
import yaml
from scripts.workflow.gate_f_prerequisites import (configuration, mean_interval, equivalent,
    fine_worker, mc_worker, calibration_config, review)
from scripts.generation.generate_zero_field_gomonay import run, validate_config

ROOT = Path(__file__).resolve().parents[2]


def test_independent_chain_interval_uses_actual_count():
    assert mean_interval(np.ones(8), .05) == [1., 1.]
    short = mean_interval(np.array([-1., 1.]*2), .05)
    long = mean_interval(np.array([-1., 1.]*4), .05)
    assert long[1] < short[1]
    with pytest.raises(ValueError):
        mean_interval([1., 2.], .05)


def test_equivalence_is_not_nonsignificance():
    settings = dict(confidence=.95, multiplicity_family_bound=512, absolute_band=.05)
    assert equivalent(np.ones(8), np.ones(8), settings)['passed']
    assert not equivalent(np.arange(8.), np.arange(8.), settings)['passed']


def fixture_manifest(tmp_path):
    design = yaml.safe_load((ROOT/'conf/gate_f/prerequisites.yaml').read_text())
    design['calibration']['sizes'] = [4, 4]
    design['fine_window'].update(duration=.04, dt=.005, save_dt=.01)
    design['kernel'].update(size=4, chains=8, mc_sweeps=20, mc_stride=2)
    (tmp_path/'raw').mkdir()
    output = tmp_path/'out'; output.mkdir()
    (output/'kernel_gate.json').write_text(json.dumps(dict(status='pass', provisional_theta=[.1, .3, .6])))
    return dict(design=design, run_id='synthetic', material=str(ROOT/'conf/literature/gomonay_2024.yaml'),
                raw=str(tmp_path/'raw'), output=str(output), manifest_hash='synthetic_not_certificate')


def test_configuration_and_gate_block(tmp_path):
    m = fixture_manifest(tmp_path)
    c = calibration_config(m, 0)
    validate_config(c)
    assert c['alpha'] == .05 and c['batch'] == 8
    assert all(value == 0 for value in c['drives'].values())
    (Path(m['output'])/'kernel_gate.json').write_text(json.dumps(dict(status='inconclusive')))
    with pytest.raises(ValueError, match='passed'):
        calibration_config(m, 0)


def test_tiny_fine_half_step_paths_and_initial_identity(tmp_path):
    m = fixture_manifest(tmp_path)
    c = configuration(m['design'], 'fixture', m['material'], 4, .1, .02, .005, .01, 8,
                      'heun_projected_predictor', 12)
    config = tmp_path/'config.yaml'; config.write_text(yaml.safe_dump(c))
    run(config, Path(m['raw'])/'calibration_0.h5', device='cpu')
    fine_worker(m, 0)
    fine_worker(m, 0, half=True)
    with h5py.File(Path(m['raw'])/'fine_0.h5') as coarse, h5py.File(Path(m['raw'])/'fine_half_0.h5') as fine:
        assert coarse.attrs['complete'] and fine.attrs['complete']
        np.testing.assert_array_equal(coarse['initial_state'][:], fine['initial_state'][:])
        np.testing.assert_allclose(coarse['time'][:], fine['time'][:])
        assert coarse['spins'].shape == fine['spins'].shape == (5, 12, 4, 4, 2, 3)
        assert np.max(np.abs(np.linalg.norm(fine['spins'][:], axis=-1)-1)) < 1e-10
        assert not np.array_equal(coarse['noise_seed'][:], fine['noise_seed'][:])


def test_mc_eight_independent_chains(tmp_path):
    m = fixture_manifest(tmp_path)
    mc_worker(m, 0)
    with np.load(Path(m['raw'])/'mc_0.npz') as h:
        assert h['spins'].shape == (8, 10, 32, 3)
        assert np.isfinite(h['energy']).all()
    with pytest.raises(FileExistsError):
        mc_worker(m, 0)


def test_missing_results_never_pass(tmp_path):
    m = fixture_manifest(tmp_path)
    with pytest.raises(SystemExit) as error:
        review(m)
    assert error.value.code == 3
    result = json.loads((Path(m['output'])/'prerequisite_review.json').read_text())
    assert result['status'] == 'inconclusive_missing_prerequisites'
    assert result['production_enabled'] is False
