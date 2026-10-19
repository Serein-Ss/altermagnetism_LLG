"""Position-resolved open-bond restart diagnostic; descriptive, not a gate."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.real_anchor_analysis import validate_report
from tests.experiments.restart_observables import matched_restart


def bond_profile(g, r):
    matched_restart(g,r)  # Check shape, unit spins and matched real anchors.
    def bonds(s):
        s=np.asarray(s,dtype=np.float64)
        return (s[:,:,:-1]*s[:,:,1:]).sum(-1)
    gb,rb=bonds(g),bonds(r)
    return dict(generated=gb,reference=rb,difference=gb-rb)


def main():
    cfg=json.loads(Path('tests/conf/real_anchor_probe.json').read_text())
    reports={seed:json.loads(Path(f'tests/results/real_anchor_probe/{seed}/summary.json').read_text())
             for seed in cfg['training_seeds']}
    for report in reports.values(): validate_report(report,cfg)
    out=Path('tests/results/restart_bond_profile'); out.mkdir(exist_ok=False)
    result=dict(status='RUNNING',rows=[],code_sha256=digest(__file__),
        scope='Exploratory signed open-bond dot residual, 128 real-anchor parents per group. Exclude anchor frame; retain all 24 bonds and ten future frames. No CI, independent-noise pointwise trajectory matching claim, causal boundary attribution or certification.')
    cache={}
    for seed,report in reports.items():
        for row in report['rows']:
            if digest(row['path'])!=row['sha256']: raise ValueError('generated paths changed')
            rp=row['reference_path']
            if rp not in cache:
                if digest(rp)!=row['reference_sha256']: raise ValueError('reference changed')
                with np.load(rp) as d: cache[rp]=d['spins'][:128]
            start=row['start_frame']; ref=cache[rp][:,start:start+11]
            with np.load(row['path']) as d:
                g=d['spins']
                if not np.array_equal(d['parent_index'],np.arange(128)): raise ValueError('parent mismatch')
                if not np.array_equal(d['reduced_time'],np.arange(11)*80): raise ValueError('time mismatch')
            obs=bond_profile(g,ref)
            path=out/f'seed{seed}_family{row["family"]}_T{row["theta"]}_start{start}.npz'
            np.savez_compressed(path,**obs)
            # No spatial pooling in saved arrays. Diagnostic summaries have fixed bond indices.
            bias=obs['difference'][:,1:].mean(axis=(0,1))
            result['rows'].append(dict(seed=seed,family=row['family'],theta=row['theta'],start_frame=start,
                path=str(path),sha256=digest(path),source=row,bond_bias=bias.tolist(),
                end_bonds_mean=float(bias[[0,23]].mean()),
                central_bonds_8_through_15_mean=float(bias[8:16].mean())))
            save(out/'summary.json',result)
    result['status']='COMPLETE_POSITION_DIAGNOSTIC_NOT_CERTIFIED'; save(out/'summary.json',result)


if __name__=='__main__': main()
