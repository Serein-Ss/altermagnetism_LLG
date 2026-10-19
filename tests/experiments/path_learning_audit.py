"""Oracle sampler checks, fixed-pair velocity fitting, and physics-input ablation."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from scripts.model.gate_f import reference_path, path_interpolate, sample, build_model
from scripts.workflow.bauer_campaign import graph, save, digest


class Oracle:
    """Test-only target-aware field; target leakage is intentional, never a model."""
    def __init__(self, target):
        self.target = target

    def __call__(self, x, tau, *args, **kwargs):
        dot = (x*self.target).sum(-1, keepdim=True).clamp(-1, 1)
        q = self.target-dot*x
        length = q.norm(dim=-1, keepdim=True)
        v = torch.atan2(length, dot)*q/length.clamp_min(1e-12)
        v = v/(1-tau).reshape(-1, 1, 1, 1, 1, 1)
        v[:, 0] = 0
        return v


def oracle_check(dtype=torch.float64, device='cpu'):
    gen = torch.Generator(device=device).manual_seed(123)
    initial = torch.randn((2, 1, 5, 1, 3), generator=gen, device=device, dtype=dtype)
    initial /= initial.norm(dim=-1, keepdim=True)
    y = reference_path(initial, 7, gen); z = reference_path(initial, 7, gen)
    tau = y.new_full((2,), .4)
    x, v, _ = path_interpolate(z, y, tau, gen)
    eps = 1e-4 if dtype == torch.float32 else 1e-6
    xp = path_interpolate(z, y, tau+eps, gen)[0]
    xm = path_interpolate(z, y, tau-eps, gen)[0]
    t = torch.linspace(0, 1, 7, device=device, dtype=dtype)[None].expand(2, -1)
    rows = []
    for steps in (16, 32, 64, 128):
        pred = sample(Oracle(y), initial, y.new_tensor([[.13,.1]]).expand(2,-1), t,
                      graph(5, device, dtype), y.new_tensor([[0.,0.,1.]]).expand(2,-1), gen,
                      steps=steps, source=z)
        rows.append(dict(steps=steps, max_error=float((pred-y).abs().max())))
    return dict(dtype=str(dtype), sampler=rows,
        derivative_error=float(((xp-xm)/(2*eps)-v).abs().max()),
        tangent_error=float((x*v).sum(-1).abs().max()))


def fixed_pair(seed, iterations=6000, device='cuda', root='tests/results/path_learning_audit', variant='original', output_root='tests/output/path_learning_audit'):
    dest = Path(root)/f'fixed_{seed}'
    if dest.exists():
        raise FileExistsError(dest)
    torch.manual_seed(seed)
    with np.load('data/research/runs/bauer_first_loop/20260913_v1/development.npz') as d:
        y = torch.as_tensor(d['train'][97:98, :, None, :, None], device=device).expand(4,-1,-1,-1,-1,-1)
    gen = torch.Generator(device=device).manual_seed(916000)
    z = reference_path(y[:,0], y.shape[1], gen)
    tau = y.new_tensor([.1,.3,.6,.9])
    x, target, diag = path_interpolate(z, y, tau, gen)
    t = torch.linspace(0,1,y.shape[1],device=device)[None].expand(4,-1)
    c = y.new_tensor([[.13,.1]]).expand(4,-1); axis=y.new_tensor([[0.,0.,1.]]).expand(4,-1)
    g = graph(25,device,y.dtype)
    if variant == 'table':
        from tests.experiments.velocity_controls import VelocityTable
        net = VelocityTable(x.shape).to(device)
    elif variant == 'cartesian':
        from tests.experiments.velocity_controls import CartesianPathNet
        net = CartesianPathNet().to(device)
    elif variant in ('original', 'wide'):
        net = build_model('rfm',width=96 if variant=='wide' else 24,blocks=2,
                          condition_mean=(.12,.1),condition_std=(.01,1.)).to(device)
    else:
        raise ValueError('unknown capacity control')
    with torch.no_grad():
        if hasattr(net,'output'):
            net.output.weight.mul_(.01);net.output.bias.zero_()
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    history=[]; scale=float(target[:,1:].square().mean())
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        prediction=net(x,tau,y[:,0],c,t,g,axis)
        loss=(prediction[:,1:]-target[:,1:]).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite fixed-pair loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        if step in (0,599,2999,iterations-1):history.append(dict(step=step+1,pre_update_relative_mse=float(loss)/scale))
    with torch.no_grad():
        error=float((net(x,tau,y[:,0],c,t,g,axis)[:,1:]-target[:,1:]).square().mean())/scale
    output=Path(output_root)/f'fixed_{seed}.pt'
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=net.state_dict(),seed=seed,steps=iterations),output)
    save(dest/'summary.json',dict(status='COMPLETE_DIAGNOSTIC',relative_mse=error,
        fixed_pair_fit_pass=error<.01,history=history,interpolation=diag,variant=variant,
        parameters=sum(p.numel() for p in net.parameters()),
        oracle=oracle_check(torch.float32,device),checkpoint_sha256=digest(output),
        scope='fixed four interpolation states; neither random-source path learning nor generalization certified'))


def main(index):
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm CUDA required')
    if not 0 <= index < 6:raise ValueError('unknown index')
    seed=[11,22,33][index%3]
    if index < 3:
        fixed_pair(seed)
    else:
        from tests.experiments.bauer_budget_comparison import run
        cfg=json.loads(Path('tests/conf/bauer_training_budget_20260913.json').read_text())
        cfg.update(physics_inputs='geometry_only',compare_legacy=False,
            output='tests/output/geometry_only',raw='tests/data/geometry_only',analysis='tests/results/geometry_only')
        path=Path(f'tests/conf/information_runs/{seed}.json')
        if path.exists():raise FileExistsError(path)
        save(path,cfg);run(path,seed,'cuda')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(1);main(args.index)
