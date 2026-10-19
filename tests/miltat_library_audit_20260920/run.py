"""Read-only first-library content audit, including all retained core outcomes."""
import json
from collections import Counter
from pathlib import Path
import h5py
import yaml
from tests.miltat_library_20260920.data import validate_manifest

HERE = Path(__file__).resolve().parent
LIBRARY = HERE.parent/'miltat_library_20260920'


def inspect_parent(path, row, cfg):
    with h5py.File(path) as h:
        if int(h.attrs['preparation_seed']) != row['preparation_seed']:
            raise ValueError('Preparation seed differs from frozen allocation')
        if h['noise_seeds'][:].tolist() != row['noise_seeds']:
            raise ValueError('Noise seeds differ from frozen allocation')
        stored = json.loads(h.attrs['config_json'])
        for key in ['dt_fs', 'save_ps', 'cold_duration_ps', 'parent_preparation_ps', 'future_duration_ps', 'temperature_K', 'alpha']:
            if stored[key] != cfg[key]:
                raise ValueError('Configuration mismatch: '+key)
        expected = (cfg['branches_per_parent'], round(cfg['future_duration_ps']/cfg['save_ps'])+1, *row['cell_shape'], 2, 3)
        if h['spins'].shape != expected:
            raise ValueError('Trajectory shape differs from frozen design')
        initial = json.loads(h.attrs['initial_core_json'])
        final = json.loads(h.attrs['final_core_json'])
        if len(final) != expected[0]:
            raise ValueError('Missing branch outcomes')
        return dict(parent_id=row['parent_id'], split=row['split'], shape=row['cell_shape'],
                    initial_core=initial, final_cores=final,
                    final_status_counts=dict(Counter(x['status'] for x in final)),
                    branches=expected[0], frames=expected[1], horizon_ps=float(h['time_ps'][-1]))


def main():
    cfg = yaml.safe_load((LIBRARY/'config.yaml').read_text())
    complete = json.loads((LIBRARY/'data/complete.json').read_text())
    if complete['smoke']:
        raise ValueError('Smoke data are not the formal library')
    records = validate_manifest(LIBRARY/'data')
    planned = json.loads((LIBRARY/'layout.json').read_text())['parents']
    if records != planned or len(records) != complete['parents']:
        raise ValueError('Library manifest differs from frozen allocation')
    rows = [inspect_parent(LIBRARY/'data'/r['file'], r, cfg) for r in records]
    summary = []
    for shape in cfg['cell_shapes']:
        for split in ['train', 'dev', 'test']:
            selected = [r for r in rows if r['shape']==shape and r['split']==split]
            counts = Counter(x['status'] for r in selected for x in r['final_cores'])
            summary.append(dict(shape=shape, split=split, parents=len(selected),
                                branches=sum(r['branches'] for r in selected), final_status_counts=dict(counts)))
    result = dict(integrity_audit_complete=True, scientific_certification=False, summary=summary, parents=rows,
        interpretation='All retained endpoint outcomes, not survival curves or equilibrium samples. Shared-parent branches are not independent parent replicates. No core-loss filtering.')
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/content_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Audited parents', len(rows))


if __name__ == '__main__':
    main()
