"""PNG-only, offline HTML report. Missing/failed runs remain visible."""
import base64
import html
import json
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.model.sphere import sphere_exp, tangent_project
from tests.bauer_closed_loop_20260919.run import HERE, config, dump, records, reference
from tests.bauer_closed_loop_20260919.metrics import compare, observables, summary


def random_paths(initial, frames, sigma, seed):
    # No exchange/drift: intentionally weak train-fitted stochastic baseline.
    s = torch.tensor(initial, dtype=torch.float64)[None].repeat(16, 1, 1)
    s = s/s.norm(dim=-1, keepdim=True)
    result = [s]
    rng = torch.Generator().manual_seed(seed)
    for _ in range(frames-1):
        noise = torch.randn(s.shape, dtype=s.dtype, generator=rng)*sigma
        s = sphere_exp(s, tangent_project(s, noise))
        result.append(s)
    return torch.stack(result, 1).numpy()


def main():
    torch.set_num_threads(1)
    cfg = config()
    out = Path(__file__).resolve().parent/'styled'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.linewidth':.8, 'legend.frameon':False,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>一维闭环</title>',
             '<style>body{max-width:1200px;margin:30px auto;font:18px/1.65 sans-serif;padding:15px}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}pre{white-space:pre-wrap}</style>',
             '<h1>一维小体系闭环：1 ps 先导</h1><p>单温、开放链 L16/25/32。不是大尺寸或长时间认证。所有运行均展示，不按测试结果挑选。</p>',
             '<p>下列容差为训练前冻结的工程筛查线，不是文献标准；误差棒为按尺寸与准备族分层的父状态bootstrap 95%区间。每条件只有16条参考未来，区间不能替代充分的分支统计功效。</p>']
    def figure(name, fig, caption):
        path = out/name
        fig.savefig(path, dpi=300, facecolor='white')
        plt.close(fig)
        encoded = base64.b64encode(path.read_bytes()).decode()
        parts.append(f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    if not (HERE/'audit.json').exists():
        parts.append('<p>参考审计未完成。下游训练未被放行。</p>')
        (out/'index.html').write_text('\n'.join(parts)+'</html>')
        return
    audit = json.loads((HERE/'audit.json').read_text())
    parts.append('<h2>参考数据核验</h2><pre>'+html.escape(json.dumps({k:audit[k] for k in ['integrity_pass','files','parents','max_stored_norm_error','reference_status']}, indent=2))+'</pre>')
    parts.append('<p>基础/细步长的对照仍是有限样本筛查，完整性通过不等于参考分布已认证。</p>')
    runs, missing = [], []
    for kind in cfg['models']:
        for seed in cfg['seeds']:
            p = HERE/'runs'/f'{kind}_{seed}'
            if not (p/'complete.json').exists():
                missing.append(f'{kind}_{seed}')
                continue
            selection = json.loads((p/'selection.json').read_text())
            report = json.loads((p/'test_128.json').read_text())
            runs.append(dict(name=p.name, path=p, selection=selection, report=report))
    parts.append('<p>缺失/未完成运行：'+(', '.join(missing) or '无')+'</p>')
    # Style-only adaptation: frozen statistics, no new baseline draws or selection.
    baseline = json.loads((HERE/'assets/baseline_statistics.json').read_text())
    bs, rs = baseline['random'], baseline['reference_8_vs_8']
    parts.append('<h2>所有模型与基线</h2><p>A局部单步；B局部联合路径；C多尺度联合路径；D同B骨干确定性预测。随机基线为仅由训练增量拟合幅度的球面随机游走，不包含相互作用。R1/R2为8对8参考，其样本量不同，不能当作16对16同精度基线。</p>')
    keys = list(cfg['margins'])
    fig, axes = plt.subplots(2, 5, figsize=(19, 8), layout='constrained')
    labels = [r['name'] for r in runs]+['random', 'R1/R2']
    stats = [r['report']['statistics'] for r in runs]+[bs, rs]
    for key, ax in zip(keys, axes.flat):
        for i, s in enumerate(stats):
            v = s[key]; lo, hi = v['parent_bootstrap_95']
            color={'A':'#8BB6D6','B':'#E7B47D','C':'#75B8AD','D':'#B5ABC8'}.get(labels[i][0],'#C4C7CB')
            ax.bar(i, v['mean'], width=.72, color=color, edgecolor='#333333', linewidth=.55,
                   hatch='//' if labels[i] in ['random','R1/R2'] else None)
            ax.errorbar(i,v['mean'],yerr=[[v['mean']-lo],[hi-v['mean']]],fmt='none',
                        ecolor='#222222',elinewidth=1.,capsize=2)
        ax.axhline(cfg['margins'][key], color='black', ls='--', label='Pilot margin')
        ax.set(title=key, ylabel='Per-parent error', xticks=range(len(labels)), xticklabels=labels, ylim=(0, None))
        ax.tick_params(axis='x', rotation=90)
        for tick in ax.get_xticklabels():tick.set_rotation_mode('anchor');tick.set_ha('right');tick.set_va('center')
    figure('metrics.png', fig, '每个训练seed单独报告；所有统计均基于固定测试父状态。虚线为工程筛查容差，不是已证明的自然误差界。')
    parts.append('<table><tr><th>运行</th><th>开发分数</th><th>选择步数</th><th>参数数</th><th>统计筛查</th></tr>')
    for r in runs:
        s = r['selection']; st = r['report']['statistics']
        count = sum(v['status'] == 'WITHIN_PILOT_MARGIN' for v in st.values())
        parts.append(f'<tr><td>{r["name"]}</td><td>{s["dev_score"]:.4g}</td><td>{s["selected_step"]}</td><td>{s["parameters"]}</td><td>{count}/{len(st)} 项在先导容差内</td></tr>')
    parts.append('</table>')
    # Example selected solely by dev among generative models, never the test score.
    generative = [r for r in runs if not r['name'].startswith('D')]
    chosen = min(generative, key=lambda r:r['selection']['dev_score']) if generative else None
    if chosen:
        parts.append('<h2>开发集选择的展示模型：'+chosen['name']+'</h2><p>每个尺寸展示测试集首个父状态，不根据是否翻转或预测效果选图。显示0–1 ps全部保存帧；1 ps内没有翻转不代表模型已学会翻转寿命。</p>')
        for length in cfg['sizes']:
            row = next(r for r in records(cfg, 'test') if r['L'] == length)
            with np.load(chosen['path']/'generated'/'steps_128'/f'{row["parent_id"]}.npz') as data:
                ref, gen, t = data['reference'], data['generated'], data['time_ps']
            fig, ax = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
            for panel, s, name in [(ax[0,0], ref, 'LLG reference, branch 0'), (ax[0,1], gen, 'Generated, branch 0')]:
                im = panel.imshow(s[0, ..., 2].T, origin='lower', aspect='auto', interpolation='nearest',
                                  extent=[t[0], t[-1], -.5, length-.5], vmin=-1, vmax=1, cmap='RdBu_r')
                panel.set(title=name, xlabel='Time (ps)', ylabel='Site'); fig.colorbar(im, ax=panel, label='sz')
            for s, color, style, label in [(ref, '#0072B2', '-', 'LLG'), (gen, '#D55E00', '--', 'Generated')]:
                o = observables(s.astype('f8'))
                for curve in o['mz']:
                    ax[0,2].plot(t, curve, color=color, ls=style, alpha=.25, lw=.7)
                ax[0,2].plot(t, o['mz'].mean(0), color=color, ls=style, label=label)
                bins=np.linspace(-1,1,21)
                counts,_=np.histogram(o['mz'][:, -1],bins=bins,density=True)
                center=(bins[:-1]+bins[1:])/2
                offset=-.023 if label=='LLG' else .023
                ax[1,0].bar(center+offset,counts,width=.044,color=color,alpha=.8,
                            edgecolor='#333333',linewidth=.5,hatch=None if label=='LLG' else '//',
                            label=f'{label} (n={len(s)})')
                ax[1,1].plot(t, o['temporal'].mean(0), color=color, ls=style, label=label)
                ax[1,2].plot(t, o['walls'].mean(0), color=color, ls=style, label=label)
            ax[0,2].set(title='All 16 branches and mean', xlabel='Time (ps)', ylabel='Mean sz', ylim=(-1,1))
            ax[1,0].set(title='Endpoint distribution (bin width 0.1)', xlabel='Mean sz at 1 ps', ylabel='Density', xlim=(-1,1),ylim=(0,None))
            ax[1,1].set(title='Spin overlap with initial', xlabel='Time (ps)', ylabel='Mean dot product', ylim=(-1,1))
            ax[1,2].set(title='Unsmoothed sign-change wall proxy', xlabel='Time (ps)', ylabel='Fraction of bonds', ylim=(0,1))
            for panel in [ax[0,2], *ax[1]]: panel.legend()
            figure(f'paths_L{length}.png', fig, f'L={length}，父状态{row["parent_id"]}，同初态参考与生成分支。两条单路径不应逐像素匹配，应结合分布和时间关联比较。')
        fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
        for r in generative:
            scores = [json.loads((r['path']/f'test_{n}.json').read_text())['score'] for n in cfg['test_flow_steps']]
            ax.plot(cfg['test_flow_steps'], scores, marker='o', label=r['name'])
        ax.set(xlabel='Flow integration steps', ylabel='Frozen composite test score', title='Sampler sensitivity: no checkpoint reselection')
        ax.legend(ncol=3)
        figure('sampling_steps.png', fig, '固定checkpoint和latent seed比较32/64/128步采样。统计分数稳定不替代逐路径数值误差分析。')
    report = dict(status='COMPLETED_PILOT' if not missing else 'INCOMPLETE_RUNS', missing=missing,
                  selected_for_display=chosen['name'] if chosen else None,
                  reference_certified=False, large_size_certified=False, long_time_certified=False,
                  runs=[dict(name=r['name'], selection=r['selection'], test=r['report']) for r in runs])
    dump(out/'report.json', report)
    encoded = base64.b64encode((out/'report.json').read_bytes()).decode()
    parts.append(f'<p><a download="report.json" href="data:application/json;base64,{encoded}">下载完整数值报告</a></p>')
    parts.append('<h2>结论边界与后续</h2><p>本页自动报告闭环运行和先导统计，不自动宣称学会完整路径分布。若误差超线，优先定位局部增量、相关性、数据覆盖或采样问题；若均在线内，仍需核验参考误差与统计功效，再决定更长窗口。未自动启动大体系、10 ns生成或新数据生产。</p></html>')
    (out/'index.html').write_text('\n'.join(parts))


if __name__ == '__main__':
    main()
