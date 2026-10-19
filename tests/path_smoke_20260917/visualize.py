"""PNG-only pilot diagnostics and an offline, self-contained HTML gallery."""
import base64
import html
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / 'literature_preflight_20260917' / 'results'


def main():
    pieces = ['<!doctype html><meta charset="utf-8"><title>三体系路径模型冒烟测试</title>',
              '<style>body{max-width:1200px;margin:30px auto;font-family:sans-serif;line-height:1.7}img{width:100%}pre{white-space:pre-wrap;background:#eee;padding:16px}</style>',
              '<h1>数据检查与训练—生成链路测试</h1>',
              '<p>原任务 670073 的三个子任务均 COMPLETED，并非消失。此页只报告小规模数值试验和软件约束；参考数据尚未通过完整文献复现认证。</p>',
              '<p>训练采用完整空间域、3 帧窗口（初态＋2 帧后继）、80 次更新，每个体系单独训练。按轨迹编号划分训练/开发数据；不是按窗口随机混分。二维和三维各只有 1 条开发轨迹，不能用于严格分布检验。</p>',
              '<p>旋转测试同时旋转磁矩和晶体参考轴；不是在固定晶体各向异性下任意旋转磁矩。周期性指连接跨越边界，不要求首尾磁矩相等。磁矩模长指单个自旋，不要求总磁化守恒。</p>',
              '<p>原型使用两跳局域信息和相邻时间帧，不包含完整远程多尺度模块、初态生成器或长时间滚动认证。固定单温度数据也不能检验温度条件的有效性。</p>']
    for name in ('bauer', 'miltat', 'hirst'):
        out = BASE / 'results' / name
        out.mkdir(parents=True, exist_ok=True)
        report = json.loads((SOURCE / name / 'report.json').read_text())
        with h5py.File(SOURCE / name / 'trajectory.h5') as f:
            t = f['time_ps'][:]
            obs = f['observables_dt_div_4']
            mag = obs['magnetization'][:]
            energy = obs['energy_per_spin'][:]
            extra = obs['neel'][:] if name == 'hirst' else (obs['core_sites'][:] if name == 'miltat' else None)
        fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
            for b in range(mag.shape[1]):
                ax[0].plot(t, mag[:, b, k], color=color, alpha=.6, label='xyz'[k] if b == 0 else None)
        ax[0].set(title='All pilot trajectories: magnetization', xlabel='Time (ps)', ylabel='Mean spin'); ax[0].legend()
        ax[1].plot(t, energy); ax[1].set(title='Energy', xlabel='Time (ps)', ylabel='Energy / (spin E0)')
        if name == 'hirst':
            ax[2].plot(t, np.linalg.norm(extra, axis=-1)); ax[2].set(title='Neel order magnitude', xlabel='Time (ps)', ylabel='|n|')
        elif name == 'miltat':
            ax[2].plot(t, extra); ax[2].set(title='Core-site diagnostic (not topology)', xlabel='Time (ps)', ylabel='Count of sz >= 0.5')
        else:
            ax[2].bar(['coarse/fine', 'medium/fine'], report['coupled_final_spin_rms_vs_finest']); ax[2].set(title='Coupled dt refinement', ylabel='Final spin RMS difference')
        fig.suptitle(name + ': full numerical pilot; physical certification pending')
        fig.savefig(out / 'reference_audit.png', dpi=160); plt.close(fig)
        pieces.append('<h2>' + name + '</h2>')
        pieces.append('<p>原始数据认证未完成项：</p><pre>' + html.escape(json.dumps(report['certification_holds'], indent=2)) + '</pre>')
        for filename, explanation in [('reference_audit.png', '参考数据全时段：磁化、能量及体系特定诊断。平滑或有限值并不等同于数值收敛或文献复现成功。'),
                                      ('overview.png', '模型冒烟测试：左上为训练损失；右上为开发轨迹与生成轨迹平均磁化；下方为同一短窗口终点的空间对比。虚实线差异是当前模型误差，不代表样本一一对应。')]:
            path = out / filename
            if path.exists():
                pieces.append('<p>' + explanation + '</p><img src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '">')
            else:
                pieces.append('<p>尚无生成结果：' + filename + '；任务可能尚未完成或失败。</p>')
        if (out / 'summary.json').exists():
            summary = json.loads((out / 'summary.json').read_text())
            summary.pop('loss', None)
            pieces.append('<pre>' + html.escape(json.dumps(summary, indent=2, ensure_ascii=False)) + '</pre>')
    if all((BASE / 'results' / n / 'summary.json').exists() for n in ('bauer', 'miltat', 'hirst')):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for i, name in enumerate(('bauer', 'miltat', 'hirst')):
            checks = json.loads((BASE / 'results' / name / 'summary.json').read_text())['checks']
            values = [checks['unit_norm_error'], checks['rotation_generated_path'],
                      max(checks['periodic_translation']), checks['open_boundary_aggregation']]
            ax.bar(np.arange(4) + (i - 1) * .24, np.maximum(values, 1e-9), width=.24, label=name)
        ax.set_xticks(np.arange(4), ['Unit norm', 'Joint SO(3)', 'PBC translation', 'Open adjacency'])
        ax.set(yscale='log', ylabel='Maximum absolute error', title='Software constraints (exact zeros shown at 1e-9 for display)')
        ax.legend(); fig.savefig(BASE / 'results' / 'constraints.png', dpi=160); plt.close(fig)
        pieces.append('<h2>几何约束检查</h2><p>精确为零的误差仅为画对数图显示在 1e-9；原始数值见各体系 JSON。测试通过不代表能量、涨落或路径分布正确。</p><img src="data:image/png;base64,' + base64.b64encode((BASE / 'results' / 'constraints.png').read_bytes()).decode() + '">')
    (BASE / 'results' / 'index.html').write_text('\n'.join(pieces))
    print(BASE / 'results' / 'index.html')


if __name__ == '__main__':
    main()
