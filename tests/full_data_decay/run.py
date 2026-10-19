"""Paired full-data FM continuation: constant versus cosine decay."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from tests.curriculum_repair.run import load, checkpoint, fm_loss
from tests.moment_bias_repair.run import evaluate
from tests.empirical_transport_repair.run import rate
from tests.bauer_p2_20260923.common import cfg, dump
from tests.bauer_comprehensive_20260923.run import real_data, next_batch, require_gpu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def tasks():
    rows = [dict(domain='bauer', arm='hierarchical', seed=s, repair=a, updates=6000)
            for s in (11, 22, 33) for a in ('constant', 'decay')]
    np.random.default_rng(93001).shuffle(rows)
    return rows


def prepare():
    if (HERE/'source_lock.json').exists(): raise RuntimeError('Already frozen')
    lock = json.loads((ROOT/'tests/moment_bias_repair/source_lock.json').read_text())
    for name, digest in lock.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    train = json.loads((ROOT/'tests/bauer_p2_20260923/manifest_train.json').read_text())
    dev = json.loads((ROOT/'tests/bauer_p2_20260923/manifest_dev.json').read_text())
    assert not {r['parent_id'] for r in train} & {r['parent_id'] for r in dev}
    dump(HERE/'protocol.json', dict(tasks=tasks(),
        scope='Existing Bauer open L16/25/32 full training windows; 63.824849668K; 1ps, .1ps saved grid; unchanged Hamiltonian/backbone/data/loss.',
        design='Three training-seed blocks, paired original FM6000 checkpoint and optimizer/data/source/tau RNG. 6000 additional FM-only updates. Constant .001 versus cosine .001 to .00003. No auxiliary losses. Shuffled task order seed93001.',
        evaluation='Fixed final step12000; 18 train calibration and36 independent dev parents,64 generated per parent;128flow steps, first train/dev parent32/256 additionally. Whole-branch bootstrap100. Reused dev is not untouched final holdout. No auto pass threshold or composite score.',
        estimands='Full-data evaluation estimates independent physical population: ddof1 both ensembles. Historical eight-path empirical-fit reanalysis uses fixed reference ddof0, generated ddof1, bootstrap generated only. No correction of physical reference variance.',
        budget='6GPU jobs estimated10-25min each; CPU report5-15min. GPU smoke estimated1-3min. Stop monitoring formal jobs after submission.',
        skills='Karpathy surgical regression; experimental-design seed-blocked concurrent paired controls; Scientific Agent Skills DOI10.48550/arXiv.2609.00065',
        limitations='Optimization diagnostic, not size transfer or proof of full path-law fidelity; three seeds cannot establish universal effectiveness. Train calibration reuse is descriptive.'))
    files = list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json', ROOT/'tests/empirical_transport_repair/run.py']
    files += [checkpoint(s) for s in (11, 22, 33)]
    for file in files: lock[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json', lock)
    (HERE/'logs').mkdir(exist_ok=True)


def train(task, out, smoke):
    net, ck = load(task)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=1e-5)
    opt.load_state_dict(ck['optimizer'])
    rng = np.random.default_rng(); rng.bit_generator.state = ck['data_rng']
    gs = {k: torch.Generator(device='cuda').set_state(v) for k, v in ck['rng'].items()}
    config = cfg(); data = real_data(config); history = []; stream = hashlib.sha256()
    steps = 32 if smoke else task['updates']; start = time.perf_counter()
    for step in range(1, steps+1):
        lr = rate(task['repair'], step, steps)
        for group in opt.param_groups: group['lr'] = lr
        y, times = next_batch(task, config, data, rng, gs['target'], 'cuda')
        tau_state = gs['tau'].get_state()
        loss, z = fm_loss(net, y, times, gs)
        for value in (y, z, tau_state): stream.update(value.detach().cpu().numpy().tobytes())
        opt.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        if not torch.isfinite(loss) or not torch.isfinite(grad): raise RuntimeError('Nonfinite update')
        opt.step()
        if step % 100 == 0 or smoke:
            history.append(dict(step=6000+step, lr=lr, loss=float(loss.detach()), gradient=float(grad)))
        if step % 500 == 0: print(task, step, flush=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start
    torch.save(dict(model=net.state_dict(), optimizer=opt.state_dict(), task=task, step=6000+steps,
        rng={k:g.get_state() for k,g in gs.items()}, data_rng=rng.bit_generator.state,
        auxiliary_rng=ck['auxiliary_rng']), out/'checkpoint.pt')
    dump(out/'history.json', history); net.eval()
    evaluation = evaluate(net, out, [], smoke)
    return dict(task=task, train_seconds=elapsed, evaluation=evaluation,
                full_data_source_tau_hash=stream.hexdigest(), device=torch.cuda.get_device_name(),
                reference_kind='population', scientific_certification=False)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare','smoke','train'])
    p.add_argument('--index', type=int, default=0); a = p.parse_args()
    if a.action == 'prepare': prepare(); return
    require_gpu('cuda'); torch.set_num_threads(1)
    for name, digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    out = HERE/f'{a.action}/{a.index:02d}'; out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); result = train(tasks()[a.index], out, a.action=='smoke')
    result['total_seconds'] = time.perf_counter()-start; dump(out/'complete.json', result)


if __name__ == '__main__': main()
