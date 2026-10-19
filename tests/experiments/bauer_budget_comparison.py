"""Budget-only RFM experiment; continuous optimization then paired evaluation."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from scripts.model.gate_f import build_model, matching_loss, sample
from scripts.workflow.bauer_campaign import graph, save, digest
from tests.experiments.bauer_checkpoint_diagnostics import initial_families, statistics
from tests.experiments.bauer_window import crop_training_windows, sample_chunks
from tests.experiments.conditional_source import conditional_source


def run(path, seed, device):
    cfg = json.loads(Path(path).read_text())
    base = json.loads(Path(cfg['base_config']).read_text())
    budgets = cfg['budgets']
    window_frames=cfg.get('window_frames',base['frames'])
    if not 2<=window_frames<=base['frames'] or (base['frames']-1)%(window_frames-1):
        raise ValueError('window must divide the full observation interval')
    if seed not in cfg['seeds'] or budgets != sorted(set(budgets)) or min(budgets) < 1:
        raise ValueError('invalid seed or budgets')
    if device == 'cuda' and (not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('Slurm GPU allocation required')
    out, raw, analysis = [Path(cfg[k])/str(seed) for k in ('output', 'raw', 'analysis')]
    for directory in (out, raw, analysis):
        if directory.exists():
            raise FileExistsError(f'preserve existing experiment: {directory}')
    for directory in (out, raw, analysis):
        directory.mkdir(parents=True)
    with np.load(cfg['data']) as d:
        arrays = {k: d[k] for k in ('train', 'condition', 'development', 'development_condition')}
    if 'train_indices' in cfg:
        arrays['train'] = arrays['train'][cfg['train_indices']]
        arrays['condition'] = arrays['condition'][cfg['train_indices']]
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    y = tensor(arrays['train'][:, :, None, :, None])
    c = tensor(arrays['condition'])
    times = torch.linspace(0, 1, base['frames'], device=device)[None,:window_frames]
    g = graph(base['L'], device, torch.float32)
    if cfg.get('physics_inputs', 'full') == 'geometry_only':
        from tests.experiments.physics_inputs import GeometryOnlyGraph
        g = GeometryOnlyGraph(g)
    elif cfg.get('physics_inputs', 'full') != 'full':
        raise ValueError('unknown information ablation')
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    if cfg.get('model', 'gate_f') == 'cartesian':
        from tests.experiments.velocity_controls import CartesianPathNet
        net = CartesianPathNet(temporal_dilations=cfg.get('temporal_dilations'),
                               global_context=cfg.get('global_context',False)).to(device)
    elif cfg.get('model', 'gate_f') == 'gate_f':
        net = build_model('rfm', width=base['width'], blocks=base['blocks'],
                          condition_mean=(.12, .1), condition_std=(.01, 1.)).to(device)
    else:
        raise ValueError('unknown experimental model')
    with torch.no_grad():
        net.output.weight.mul_(.01); net.output.bias.zero_()
    optimizer = torch.optim.Adam(net.parameters(), lr=base['learning_rate'])
    optimization=cfg.get('optimization')
    ema={k:v.detach().clone() for k,v in net.state_dict().items()} if optimization else None
    gen = torch.Generator(device=device).manual_seed(500000000+seed)
    report = dict(status='TRAINING', config=cfg, base_config=base, seed=seed,
                  data_sha256=digest(cfg['data']), code_sha256={p: digest(p) for p in
                  [__file__, 'scripts/model/gate_f.py', 'tests/experiments/bauer_checkpoint_diagnostics.py', 'tests/experiments/physics_inputs.py', 'tests/experiments/velocity_controls.py']},
                  job_id=os.getenv('SLURM_JOB_ID'), milestones=[], evaluation=[])
    save(analysis/'summary.json', report)
    if cfg.get('flow_weighting')=='quadratic' or cfg.get('source_scale') is not None:
        report['code_sha256']['tests/experiments/weighted_flow_loss.py']=digest('tests/experiments/weighted_flow_loss.py')
    report['code_sha256']['tests/experiments/bauer_window.py']=digest('tests/experiments/bauer_window.py')
    report['code_sha256']['tests/experiments/conditional_source.py']=digest('tests/experiments/conditional_source.py')
    if optimization:report['code_sha256']['tests/experiments/stable_optimization.py']=digest('tests/experiments/stable_optimization.py')
    losses = []; started = time.monotonic()
    for step in range(1, max(budgets)+1):
        if optimization:
            from tests.experiments.stable_optimization import cosine_lr,update_ema
            for group in optimizer.param_groups:
                group['lr']=cosine_lr(step,max(budgets),base['learning_rate'],optimization['min_lr'])
        index = torch.randint(len(y), (base['batch_size'],), device=device, generator=gen)
        batch = crop_training_windows(y[index],window_frames,gen); cond = c[index]; t = times.expand(len(batch), -1)
        axis = batch.new_tensor([0., 0., 1.]).expand(len(batch), -1)
        optimizer.zero_grad(set_to_none=True)
        if cfg.get('source_scale') is not None:
            if cfg.get('flow_weighting','uniform')!='uniform':raise ValueError('source ablation must keep uniform weighting')
            from tests.experiments.weighted_flow_loss import weighted_flow_loss
            loss, _ = weighted_flow_loss(net,batch,cond,t,g,axis,gen,power=0,source_scale=cfg['source_scale'])
        elif cfg.get('flow_weighting','uniform')=='quadratic':
            from tests.experiments.weighted_flow_loss import weighted_flow_loss
            loss, _ = weighted_flow_loss(net,batch,cond,t,g,axis,gen)
        elif cfg.get('flow_weighting','uniform')=='uniform':
            loss, _ = matching_loss(net, batch, cond, t, g, axis, gen, kind='rfm')
        else:
            raise ValueError('unknown flow weighting')
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        if not torch.isfinite(grad):
            raise RuntimeError('nonfinite gradient')
        optimizer.step(); losses.append(float(loss))
        if ema is not None:update_ema(ema,net.state_dict(),optimization['ema_decay'])
        if step in budgets:
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.monotonic()-started
            checkpoint = out/f'step_{step}.pt'
            torch.save(dict(model=ema if ema is not None else net.state_dict(),raw_model=net.state_dict(),
                inference_weights='ema' if ema is not None else 'raw_model',optimizer_weights='raw_model',optimizer=optimizer.state_dict(),
                generator_state=gen.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else [],
                step=step, seed=seed, config=base, experiment=cfg), checkpoint)
            row = dict(step=step, cumulative_wall_seconds=elapsed,
                       learning_rate=optimizer.param_groups[0]['lr'],inference_weights='ema' if ema is not None else 'raw_model',
                       last50_mean=float(np.mean(losses[-50:])), checkpoint_sha256=digest(checkpoint))
            if step == 600 and cfg.get('compare_legacy', True):
                old_path = Path(cfg['old_checkpoints'])/f'rfm_{seed}.pt'
                old = torch.load(old_path, map_location=device, weights_only=False)['model']
                row['old_checkpoint_sha256'] = digest(old_path)
                row['old_weight_max_abs_difference'] = max(float((v-old[k]).abs().max())
                                                            for k, v in net.state_dict().items())
            report['milestones'].append(row)
            save(analysis/'training_loss.json', losses)
            save(analysis/'summary.json', report)
            print(json.dumps(row), flush=True)
    report['status'] = 'EVALUATING'
    for budget in budgets:
        net.load_state_dict(torch.load(out/f'step_{budget}.pt', map_location=device,
                                       weights_only=False)['model'])
        net.eval()
        loss_summary = {}
        splits = [('train', 'condition')] if cfg.get('evaluate_training_subset') else [('train', 'condition'), ('development', 'development_condition')]
        for split, condition_key in splits:
            yy = tensor(arrays[split][:, :, None, :, None]); cc = tensor(arrays[condition_key])
            eg = torch.Generator(device=device).manual_seed(915000)
            values = []
            with torch.no_grad():
                for _ in range(cfg['loss_draws']):
                    idx = torch.randint(len(yy), (base['batch_size'],), device=device, generator=eg)
                    axis = yy.new_tensor([0., 0., 1.]).expand(len(idx), -1)
                    eval_batch=crop_training_windows(yy[idx],window_frames,eg)
                    if cfg.get('source_scale') is not None:
                        from tests.experiments.weighted_flow_loss import weighted_flow_loss
                        value,_=weighted_flow_loss(net,eval_batch,cc[idx],times.expand(len(idx),-1),g,axis,eg,power=0,source_scale=cfg['source_scale'])
                    else:
                        value, _ = matching_loss(net, eval_batch, cc[idx], times.expand(len(idx), -1), g, axis, eg)
                    values.append(float(value))
            loss_summary[split] = dict(mean=float(np.mean(values)), draws=values,window_frames=window_frames)
        for theta in cfg.get('evaluation_theta', base['theta']):
            split, condition_key = ('train', 'condition') if cfg.get('evaluate_training_subset') else ('development', 'development_condition')
            pool = arrays[split][np.isclose(arrays[condition_key][:, 0], theta)]
            selected, groups = initial_families(pool, cfg['initials'], cfg.get('reference_latents', cfg['latents']))
            x = tensor(np.repeat(pool[selected, 0], cfg['latents'], axis=0)[:, None, :, None])
            condition = x.new_tensor([theta, .1]).expand(len(x), -1)
            axis = x.new_tensor([0., 0., 1.]).expand(len(x), -1)
            if window_frames<base['frames']:
                pred=sample_chunks(net,x,condition,times.expand(len(x),-1),g,axis,
                    torch.Generator(device=device).manual_seed(913000),
                    chunks=(base['frames']-1)//(window_frames-1),steps=cfg['sampling_steps'],source_scale=cfg.get('source_scale'))
            else:
                source = torch.cat([conditional_source(x[j:j+1], base['frames'],
                    torch.Generator(device=device).manual_seed(913000+j),cfg.get('source_scale')) for j in range(len(x))])
                pred = sample(net, x, condition, times.expand(len(x), -1), g, axis,
                              torch.Generator(device=device).manual_seed(913000),
                              steps=cfg['sampling_steps'], source=source)
            if not torch.isfinite(pred).all() or not torch.equal(pred[:, 0], x):
                raise RuntimeError('invalid generated trajectory')
            a = pred[:, :, 0, :, 0].cpu().numpy()
            np.save(raw/f'step_{budget}_T{theta}.npy', a)
            stats = lambda paths: statistics(paths, base['save_dt'], cfg['threshold'], cfg['dwell'])
            row = dict(step=budget, theta=theta, loss=loss_summary,
                initial_indices=selected.tolist(), reference_indices=groups.tolist(),
                generated=stats(a), reference=stats(pool[groups.reshape(-1)]),
                per_initial_generated=[stats(part) for part in a.reshape(cfg['initials'], cfg['latents'], *a.shape[1:])],
                per_initial_reference=[stats(pool[group]) for group in groups])
            if cfg.get('evaluate_training_subset'):
                distances = np.sqrt(np.square(a[:, None]-pool[None]).mean(axis=(2, 3, 4)))
                row['nearest_training_path_rmse'] = distances.min(axis=1).tolist()
                row['nearest_training_path_counts'] = np.bincount(distances.argmin(axis=1), minlength=len(pool)).tolist()
                row['scope'] = 'training memorization only; nearest-path counts are not physical mechanism probabilities'
            report['evaluation'].append(row)
            save(analysis/'summary.json', report)
    report['status'] = 'COMPLETE_EXPLORATORY_NOT_CERTIFIED'
    save(analysis/'summary.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='tests/conf/bauer_training_budget_20260913.json')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    a = p.parse_args()
    torch.set_num_threads(1)
    run(a.config, a.seed, a.device)
