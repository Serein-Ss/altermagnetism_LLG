"""Descriptive PNGs and path-level sign events; no basin certification."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.descriptive_path_events import sign_events


def figure(profiles, time, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ['black', 'gray', '#0072B2', '#D55E00', '#009E73']
    styles = ['-', '--', '-', '-.', ':']
    events = {}
    for (name, p), color, style in zip(profiles.items(), colors, styles):
        mz = p['magnetization_per_path'][..., 2]
        events[name] = sign_events(mz, time)
        axes[0, 0].plot(time, p['spatial_dot_lag1'], color=color, ls=style, label=name)
        axes[0, 1].plot(time, p['spatial_dot_lag1']-profiles['R1']['spatial_dot_lag1'],
                        color=color, ls=style, label=name)
        axes[1, 0].hist(mz[:, -1], bins=np.linspace(-1, 1, 26), density=True,
                        histtype='step', color=color, ls=style, label=name)
        cumulative = np.maximum.accumulate(mz < 0, axis=1).mean(axis=0)
        axes[1, 1].plot(time, cumulative, color=color, ls=style, label=name)
    axes[0, 0].set(xlabel='Reduced time', ylabel='Nearest-neighbor spin dot product')
    axes[0, 1].set(xlabel='Reduced time', ylabel='Dot product difference from R1')
    axes[0, 1].axhline(0, color='black', lw=.5)
    axes[1, 0].set(xlabel='Endpoint mean magnetization z', ylabel='Probability density', xlim=(-1, 1))
    axes[1, 1].set(xlabel='Reduced time', ylabel='Fraction ever observed below zero', ylim=(0, 1))
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(title+'\n128 paths/group; descriptive curves, no confidence bands or committed-switch claim')
    return fig, events


def main():
    root = Path('tests/results/independent_moment_analysis')
    out = Path('tests/results/independent_diagnostic_figures')
    if out.exists():
        raise FileExistsError('preserve existing diagnostic figures')
    reports = []
    for family in range(4):
        for theta in (.11, .13):
            p = root/f'family_{family}_T{theta}'/'summary.json'
            r = json.loads(p.read_text())
            if r['status'] != 'COMPLETE_CONDITIONAL_MOMENTS_NOT_CERTIFIED':
                raise ValueError('complete batch required')
            if (r['family'], r['theta']) != (family, theta):
                raise ValueError('mismatched conditional report')
            reports.append((p, r))
    rows = []
    out.mkdir()
    for p, r in reports:
        profiles = {}
        sources = [p, p.parent/'reference_profiles.npz']
        with np.load(sources[-1]) as d:
            time = d['reduced_time']
            for name in ('R1', 'R2'):
                prefix = name.lower()+'_'
                profiles[name] = {k[len(prefix):]: d[k] for k in d.files if k.startswith(prefix)}
        for seed in (11, 22, 33):
            source = p.parent/f'model_{seed}_profiles.npz'
            sources.append(source)
            with np.load(source) as d:
                profiles[f'Model {seed}'] = dict(d)
        if any(v['magnetization_per_path'].shape != (128, 201, 3) for v in profiles.values()):
            raise ValueError('unexpected ensemble shape')
        fig, events = figure(profiles, time, f"Initial {r['family']}, reduced temperature {r['theta']}")
        stem = p.parent.name
        fig.savefig(out/f'{stem}.png', dpi=180)
        plt.close(fig)
        np.savez_compressed(out/f'{stem}_events.npz',
                            **{f'{name}/{k}': v for name, e in events.items() for k, v in e.items()})
        rows.append(dict(family=r['family'], theta=r['theta'],
            source_sha256={str(s): digest(s) for s in sources},
            counts={name: {k: int(e[k].sum()) for k in ('negative_endpoint', 'crossed_zero')}
                    for name, e in events.items()}))
    save(out/'summary.json', dict(status='COMPLETE_DESCRIPTIVE_FIGURES_NOT_CERTIFIED', rows=rows,
         code_sha256=digest(__file__), scope='Saved-frame sign events only; not basin switching. No inference or model selection.'))


if __name__ == '__main__':
    main()
