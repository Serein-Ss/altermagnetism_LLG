"""Descriptive fixed-region audit of the completed boundary-loss intervention."""
import json
from pathlib import Path

import numpy as np

from scripts.workflow.bauer_campaign import digest, save


def regional_errors(candidate, reference):
    """Errors of ensemble means and neighbor dot products at the final time."""
    mean_error = candidate.mean(0) - reference.mean(0)
    bond_error = (candidate[:, :-1] * candidate[:, 1:]).sum(-1).mean(0)
    bond_error -= (reference[:, :-1] * reference[:, 1:]).sum(-1).mean(0)
    sites = {'ends': np.r_[0:3, 22:25], 'interior': np.arange(3, 22)}
    bonds = {'ends': np.array([0, 1, 22, 23]), 'interior': np.arange(2, 22)}
    return {region: {
        'mean_rmse': float(np.sqrt(np.mean(mean_error[index] ** 2))),
        'bond_rmse': float(np.sqrt(np.mean(bond_error[bonds[region]] ** 2))),
    } for region, index in sites.items()}


def run():
    root = Path('tests/results/boundary_weighted_analysis')
    source = root / 'summary.json'
    summary = json.loads(source.read_text())
    verification = json.loads((root / 'verification.json').read_text())
    assert verification['status'] == 'VERIFIED_RAW_BOUNDARY_COMPARISON_NOT_CERTIFIED'
    assert verification['source_sha256'] == digest(source)
    reference_path = Path('tests/results/fresh_rollout_reference/summary.json')
    assert digest(reference_path) == summary['reference_sha256']
    refs = {r['anchor']: r for r in json.loads(reference_path.read_text())['rows'] if r['case'] == 1}
    rows = []
    for row in summary['rows']:
        seed, anchor = row['seed'], row['anchor']
        items = {'reference': refs[anchor]}
        for label, directory in [('control', 'coverage_intervention_training'),
                                 ('boundary', 'boundary_weighted_training')]:
            report_path = Path(f'tests/results/{directory}/{seed}/summary.json')
            assert digest(report_path) == row['source_sha256'][label]
            report = json.loads(report_path.read_text())
            items[label] = next(r for r in report['rows'] if r['anchor'] == anchor)
        data = {}
        for label, item in items.items():
            assert digest(item['path']) == item['sha256']
            with np.load(item['path']) as z:
                data[label] = z['spins'][:, -1].astype(float)
            assert data[label].shape == (256, 25, 3)
            assert np.isfinite(data[label]).all()
        rows.append(dict(seed=seed, anchor=anchor, state=row['state'],
            metrics={label: regional_errors(data[label], data['reference'])
                     for label in ['control', 'boundary']}))
    assert len(rows) == 36
    groups = []
    for kind in ['aligned', 'tilted', 'wall']:
        for theta in [.11, .13]:
            for seed in [11, 22, 33]:
                selected = [r for r in rows if r['state']['kind'] == kind
                            and r['state']['theta'] == theta and r['seed'] == seed]
                assert len(selected) == 2
                groups.append(dict(kind=kind, theta=theta, seed=seed,
                    metrics={label: {region: {key: float(np.mean([
                        r['metrics'][label][region][key] for r in selected]))
                        for key in ['mean_rmse', 'bond_rmse']}
                        for region in ['ends', 'interior']}
                        for label in ['control', 'boundary']}))
    save(root / 'regional_audit.json', dict(rows=rows, groups=groups,
        source_sha256=digest(source), code_sha256=digest(__file__),
        scope='Exploratory terminal-time region errors. Each seed group averages two fixed initial families. '
              'End sites: first/last three; end bonds: first/last two. Remaining bonds include region interfaces. '
              'Descriptive only; no confidence interval, independence, equivalence or path-law certification.'))
    for group in groups:
        print(group)


if __name__ == '__main__':
    run()
