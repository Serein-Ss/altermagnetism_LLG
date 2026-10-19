"""Frozen-checkpoint diagnostics: shared-state transitions and free rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from tests.bauer_closed_loop_20260919.model import Net
from tests.bauer_closed_loop_20260919.metrics import compare
from tests.bauer_transition_20260920.run import config, rollout, temporal_errors
from tests.bauer_two_stage_20260919.common import allocation, dump
from tests.bauer_teacher_state_20260920.sample import checked_state
from tests.bauer_teacher_state_20260920.report import moments

HERE = Path(__file__).resolve().parent


def identity(index):
    if index not in range(6): raise ValueError('Six paired experiments only')
    return f'w{index//3}_{[11,22,33][index%3]}'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--index', type=int, required=True)
    p.add_argument('--device', choices=['cuda'], required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); allocation('cuda'); torch.set_num_threads(1); cfg = config(); name = identity(a.index)
    run = HERE/('train_smoke' if a.smoke else 'runs')/name
    assert json.loads((run/'complete.json').read_text())['smoke'] == a.smoke
    checkpoint = run/'best.pt'; ck = torch.load(checkpoint, map_location='cuda', weights_only=False)
    net = Net('C', cfg['width'], cfg['depth']).cuda(); net.load_state_dict(ck['model']); net.eval()
    out = HERE/('eval_smoke' if a.smoke else 'evaluation')/name; out.mkdir(parents=True, exist_ok=False)
    teacher = HERE.parent/'bauer_teacher_state_20260920'
    spec = json.loads((teacher/'data/specification.json').read_text())
    states = spec['rows'][:1] if a.smoke else spec['rows']; local = []; paths = []
    def initial(s, branches):
        return F.normalize(torch.tensor(s, dtype=torch.float32, device='cuda'), dim=-1)[None].repeat(branches, 1, 1)
    with h5py.File(teacher/'data/initial_states.h5', 'r') as inputs, h5py.File(teacher/'results/reference.h5', 'r') as ref, h5py.File(out/'teacher.h5', 'x') as dest:
        assert inputs.attrs['complete'] and ref.attrs['complete']; dest.attrs['complete'] = False
        for steps in [16, 64]:
            group = dest.create_group(f'steps_{steps}')
            for j, row in enumerate(states):
                s = checked_state(inputs, row)
                x = rollout(net, initial(s, 64), .1, 260920910+j, steps, 'transition', cfg).cpu().numpy()
                truth = ref[row['key']][:]; group.create_dataset(row['key'], data=x, compression='lzf')
                local.append(dict(L=row['L'], parent=row['parent'], time_ps=row['time_ps'], steps=steps,
                     model_ref_a=moments(x, truth[:64]), model_ref_b=moments(x, truth[64:]),
                     ref_b_ref_a=moments(truth[64:], truth[:64])))
        dest.attrs['complete'] = True
    for length in ([16] if a.smoke else cfg['evaluation_sizes']):
        with h5py.File(out/f'L{length}.h5', 'x') as dest:
            dest.attrs['complete'] = False
            for parent in range(1 if a.smoke else 6):
                source = HERE.parent/'bauer_two_stage_20260919/data'/f'L{length}'/f'test_future_{parent}.h5'
                with h5py.File(source, 'r') as ref:
                    assert ref.attrs['complete']; truth = ref['spins'][:, :101].astype(float)
                    s = ref['initial'][:]; times = ref['time_ps'][:101]
                torch.cuda.synchronize(); start = time.perf_counter()
                x = rollout(net, initial(s, 16), 10., 260920000+length*100+parent, 64, 'transition', cfg).cpu().numpy()
                seconds = time.perf_counter()-start
                assert x.shape[1] == len(times) and np.isclose(times[-1], 10.)
                dest.create_dataset(f'parent_{parent}', data=x, compression='lzf')
                for end in [11,101]:
                    paths.append(dict(L=length, parent=parent, horizon_ps=float(times[end-1]),
                        raw=compare(x[:,:end].astype(float),truth[:16,:end]),
                        reference_floor=compare(truth[16:,:end],truth[:16,:end]),
                        temporal=temporal_errors(x[:,:end].astype(float),truth[:,:end]),
                        generation_seconds_10ps=seconds))
                print(name, length, parent, flush=True)
            dest['time_ps'] = times; dest.attrs['complete'] = True
    dump(out/'metrics.json', dict(model=name, selected_step=ck['step'], teacher=local, paths=paths,
         checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
         scientific_certification=False, note='Previously inspected diagnostic parents; no final holdout or test selection.'))
    dump(out/'complete.json', dict(model=name, smoke=a.smoke, teacher_cases=len(local), path_cases=len(paths)))


if __name__ == '__main__':
    main()
