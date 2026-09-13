"""Run unchanged frozen literature kernels with the new artifact directories.

Only the I/O preparation is adapted. Checkpoint protocol, kernel bytes, RNG and
environment identities must match; scientific settings are never relaxed.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT/'scripts/archive/zero_field_20260911_040025/code/altermagnetism_LLG'
FROZEN = ROOT/'conf/archive/frozen'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specification(index):
    rows = [
        ('bauer_2011', 'trajectory', '20260911_040025_014', 'zero_field_gomonay/20260911_040025/014.yaml'),
        ('bauer_2011', 'trajectory', '20260913_023043_bauer_dt_0', 'bauer_revision4/20260913_023043/dt_0.yaml'),
        ('bauer_2011', 'trajectory', '20260913_023043_bauer_dt_1', 'bauer_revision4/20260913_023043/dt_1.yaml'),
        ('laliena_crnb3s6_2020', 'branch', '20260911_040025_031', 'zero_field_gomonay/20260911_040025/031.yaml'),
        ('laliena_crnb3s6_2020', 'branch', '20260911_040025_035', 'zero_field_gomonay/20260911_040025/035.yaml'),
        ('laliena_crnb3s6_2020', 'helix', '20260911_040025_039', 'zero_field_gomonay/20260911_040025/039.yaml'),
        ('laliena_crnb3s6_2020', 'current', '20260911_040025_043', 'zero_field_gomonay/20260911_040025/043.yaml'),
    ]
    paper, protocol, run, config = rows[index]
    return argparse.Namespace(paper=paper, protocol=protocol, run_id=run,
                              config=FROZEN/config, device='cpu', validation=True)


def load_frozen():
    # Script entry only: imports must resolve to the byte-preserved snapshot.
    sys.path.insert(0, str(SNAPSHOT))
    import scripts.literature.workflow as workflow
    assert Path(workflow.__file__).is_relative_to(SNAPSHOT)
    return workflow


def prepare(args, paper, *, check_only=False):
    import h5py
    import numpy as np
    import torch
    from scripts.core.literature_config import read_document, load_runtime
    document = read_document(args.config)
    assert document['paper_id'] == paper
    runtime = load_runtime(args.config)
    p = runtime.numerics['protocols'][args.protocol]
    assert args.protocol in runtime.numerics['validation_extension']['validation_protocols']
    code = {str(path.relative_to(SNAPSHOT)): digest(path)
            for folder in (SNAPSHOT/'scripts/core', SNAPSHOT/'scripts/literature')
            for path in folder.rglob('*.py')}
    identity = dict(config=digest(args.config), code=code)
    if p.get('resumable'):
        p = dict(p, resume_identity=identity)
    raw = ROOT/'data/literature_reproduction'/paper/'raw'/(args.run_id+'.partial')
    out = ROOT/'assets/literature_reproduction'/paper/args.run_id
    if raw.with_name(args.run_id).exists():
        raise FileExistsError('completed run exists; do not duplicate: '+args.run_id)
    checkpoint = raw/'trajectory.h5.checkpoint.h5'
    progress = None
    if checkpoint.exists():
        with h5py.File(checkpoint, 'r') as c, h5py.File(raw/'trajectory.h5.partial', 'r') as h:
            stored = json.loads(c.attrs['identity'])
            assert stored['scope'] == identity, 'frozen physics/configuration changed'
            assert stored['protocol'] == p, 'protocol changed'
            assert stored['torch'] == torch.__version__ and stored['numpy'] == np.__version__
            assert stored['device'] == 'cpu' and stored['device_name'] == 'cpu'
            assert c.attrs['identity'] == h.attrs['identity']
            assert list(c['spins'].shape) == stored['shape']
            assert np.isfinite(c['spins'][:]).all()
            assert int(c.attrs['frame']) == int(h.attrs['written_frames'])
            assert len(c['rng']) == len(c['spins'])
            for i in range(len(c['rng'])):
                torch.Generator().set_state(torch.from_numpy(c[f'rng/{i}'][:]))
            progress = dict(step=int(c.attrs['step']), frame=int(c.attrs['frame']),
                            checkpoint_sha256=digest(checkpoint))
    elif args.protocol == 'trajectory':
        raise RuntimeError('expected existing Bauer checkpoint')
    manifest = dict(paper_id=paper, protocol=args.protocol, configuration=document,
                    config_sha256=identity['config'], code_sha256=code, device='cpu',
                    stage='numerical_validation', run_id=args.run_id,
                    status='running', claim='numerical_experiment_not_literature_certification',
                    resumed_from=progress, layout_adapter=digest(__file__))
    if check_only:
        return manifest
    raw.mkdir(parents=True, exist_ok=bool(p.get('resumable')))
    out.mkdir(parents=True, exist_ok=bool(p.get('resumable')))
    with (out/'manifest.json').open('w') as stream:
        json.dump(manifest, stream, indent=2)
    return runtime.reduced, p, raw, out, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=range(7))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    workflow = load_frozen()
    if args.self_test:
        import tempfile
        import h5py
        import numpy as np
        import torch
        from scripts.core.streaming_llg import simulate_resumable
        from scripts.core.literature_config import load_runtime
        from scripts.literature.bauer_2011.model import OpenChain
        from scripts.literature.bauer_2011.integrator import BauerWeakRK
        runtime = load_runtime(specification(0).config)
        p = dict(runtime.numerics['protocols']['trajectory'], steps=19, save_every=3, checkpoint_every=5)
        model = OpenChain(runtime.reduced)
        llg = BauerWeakRK(model, alpha=.1, theta=.11, project=True)
        s = torch.zeros((8,100,3), dtype=torch.float64); s[...,2] = 1.
        with tempfile.TemporaryDirectory(prefix='llg_resume_test_') as folder:
            a, b = Path(folder)/'a.h5', Path(folder)/'b.h5'
            expected, done = simulate_resumable(a,s.clone(),llg,p,identity='layout_test')
            assert done
            _, done = simulate_resumable(b,s.clone(),llg,p,identity='layout_test',stop_after=8)
            assert not done
            actual, done = simulate_resumable(b,s.clone(),llg,p,identity='layout_test')
            assert done and torch.equal(actual,expected)
            with h5py.File(a) as x, h5py.File(b) as y:
                for name in x:
                    np.testing.assert_array_equal(x[name][:],y[name][:])
        print('PASS: frozen Bauer MT continuous/resumed trajectories and all datasets exactly equal')
        return
    if args.check:
        print(json.dumps([prepare(specification(i), specification(i).paper, check_only=True)
                          for i in range(7)], indent=2))
        return
    if args.index is None or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('production requires a Slurm task and explicit index')
    selected = specification(args.index)
    workflow.prepare = prepare
    workflow.arguments = lambda paper, modes: selected
    importlib.import_module('scripts.literature.'+selected.paper+'.run').main()


if __name__ == '__main__': main()
