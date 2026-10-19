"""Prospective initial bank disjoint from specified, already viewed datasets."""
from pathlib import Path
import h5py
import numpy as np
from scripts.workflow.bauer_campaign import initial_states,save,digest


def initial_keys(states):
    a=np.asarray(states)
    if a.ndim!=3 or a.shape[-1]!=3 or not np.isfinite(a).all():
        raise ValueError('finite [initial,site,3] array required')
    return [np.asarray(s,dtype='<f4').tobytes() for s in a]


def assert_disjoint_initials(candidate,previous):
    keys=initial_keys(candidate)
    if len(set(keys))!=len(keys):raise ValueError('duplicate candidate initial at model precision')
    for name,states in previous.items():
        if np.asarray(states).shape[1:]!=np.asarray(candidate).shape[1:]:raise ValueError('initial shape mismatch')
        if set(keys)&set(initial_keys(states)):raise ValueError(f'initial overlap with {name}')
    return dict(candidate_initials=len(keys),previous_initial_counts={name:len(initial_keys(s)) for name,s in previous.items()},
        comparison='Exact equality after float32 conversion, matching model input precision; not a proof of statistical independence or distinct source chains.')


def main():
    destination=Path('tests/results/reference_initial_bank')
    raw=Path('tests/data/reference_initial_bank')
    if destination.exists() or raw.exists():raise FileExistsError('preserve registered bank')
    data=Path('data/research/runs/bauer_first_loop/20260913_v1')
    source=data/'development.npz'
    with np.load(source) as d:previous=dict(train=d['train'][:,0],development=d['development'][:,0])
    provenance={str(source):digest(source)}
    for theta in (.11,.13):
        path=data/f'reference_T{theta}.h5'
        with h5py.File(path) as h:previous[f'old_evaluation_T{theta}']=h['initial'][:]
        provenance[str(path)]=digest(path)
    previous['runtime_pilot']=initial_states(4,25,'reference_runtime_pilot_20260914')
    namespace='stable_independent_reference_v1_20260914'
    candidate=initial_states(4,25,namespace)
    audit=assert_disjoint_initials(candidate,previous)
    destination.mkdir(parents=True);raw.mkdir(parents=True)
    path=raw/'initials.npy';np.save(path,candidate)
    save(destination/'manifest.json',dict(status='INITIAL_BANK_REGISTERED_NO_TRAJECTORIES',
        scope='Four prospective initial families, shared across theta for matched-condition comparisons. No future trajectories viewed. Not a complete validation contract; no final noise sample size, physics tolerance or confirmation claim.',
        namespace=namespace,initial_sha256=digest(path),source_sha256=provenance,audit=audit,
        code_sha256={p:digest(p) for p in [__file__,'scripts/workflow/bauer_campaign.py']},
        pending=['noise namespace and sample count','reference timestep and cost','analysis contract and statistical precision']))


if __name__=='__main__':main()
