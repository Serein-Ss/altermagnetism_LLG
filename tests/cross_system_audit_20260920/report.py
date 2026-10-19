"""Offline development report; missing/failed experiments never imply success."""
import base64
import html
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tests.bauer_closed_loop_20260919.run import records, config
from tests.bauer_two_stage_20260919.common import dump

HERE = Path(__file__).resolve().parent
OUT = HERE/'assets'


def image(path, caption):
    kind = 'gif' if path.suffix == '.gif' else 'png'
    return '<figure><img alt="'+html.escape(caption)+'" src="data:image/'+kind+';base64,'+base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'


def local_metrics(x, r):
    dg, dr = x[:, 1]-x[:, 0], r[:, 1]-r[:, 0]
    mg, mr = dg.mean(0), dr.mean(0)
    s = r[0, 0]
    tg, tr = [v-(v*s).sum(-1, keepdims=True)*s for v in [mg, mr]]
    return dict(raw_mse=float(((mg-mr)**2).mean()), persistence_mse=float((mr**2).mean()),
                mean_tangent_cosine=float((tg*tr).sum()/max(np.linalg.norm(tg)*np.linalg.norm(tr), 1e-12)),
                tangent_amplitude_ratio=float(np.linalg.norm(tg)/max(np.linalg.norm(tr), 1e-12)),
                generated_variance=float(dg.var(0, ddof=1).mean()), reference_variance=float(dr.var(0, ddof=1).mean()))


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    sections = ['<h1>一维原因排查与二维新数据测试 · 2026-09-20</h1>',
        '<p>这是开发性结果快照，不是实时监控，不是发表级认证。图像内嵌，单独下载本页也可看图与GIF。文献部分基于本轮核对的摘要和出版记录，不冒充逐篇全文审查。</p>',
        '<h2>创新定位与相近工作</h2><p>组合可以成为方法贡献，但须证明其必要性和收益。本轮为有界检索，未确认有完全覆盖本项目目标的工作；这不是“全球首次”的证明。</p>',
        '<ul><li><a href="https://arxiv.org/abs/1903.09499">Learning magnetization dynamics (2019)</a>：磁化状态压缩与潜空间动力学预测。</li>',
        '<li><a href="https://orbit.dtu.dk/en/publications/transformer-neural-networks-for-predicting-magnetization-dynamics-2/">Transformer Neural Networks for Predicting Magnetization Dynamics (2022)</a>：会议摘要，非完整期刊证据。</li>',
        '<li><a href="https://arxiv.org/abs/2607.28537">GNN Force Fields for Spin Dynamics (2026)</a>：预印本；学习磁性能量／力矩，用于动力学，不等同于直接学习条件路径分布。</li>',
        '<li><a href="https://arxiv.org/abs/2305.03804">Equivariant Neural Networks for Spin Dynamics Simulations of Itinerant Magnets (2023)</a>：已有二维方格／三角格上的等变自旋动力学工作，不能把旋转等变本身写成首次。</li>',
        '<li><a href="https://arxiv.org/abs/2508.15318">Flow Matching-based Monte Carlo initialization (v5, 2026)</a>：XY小体系到大体系；最新版明确是MC初态生成，不是动力学路径替代。</li></ul>',
        '<p>本项目应检验：小体系条件随机演化→大体系时空统计与机制概率，以及同精度计算成本。世界模型是任务定位，不是额外一个独立网络模块。</p>',
        '<h2>一维：分层排查</h2><p>所有数值是复用训练／开发参考的描述性诊断，不给独立显著性或认证结论。不同自旋不是独立实验重复。</p>']
    root = HERE.parent/'bauer_moment_training_20260920/fit_audit'
    forensic = []; families = ['positive','negative','tilted','edge_domain','interior_domain','single_wall']
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for wi, color in [(0, '#0072B2'), (1, '#D55E00')]:
        rows = []
        for p in sorted(root.glob(f'w{wi}_*')):
            rows += json.loads((p/'metrics.json').read_text())['rows']
        for ax, split in zip(axs, ['train','dev']):
            values = [[r['mean_increment']['corrected_mse'] for r in rows if r['split']==split and r['family']==f] for f in families]
            ax.bar(np.arange(6)+(wi-.5)*.36, [np.mean(v) for v in values], width=.36, color=color, label=f'Mean weight {wi}')
            ax.set_xticks(range(6), families, rotation=35, ha='right')
            ax.set(title=split+' (3 seeds pooled descriptively)', ylabel='Sampling-corrected local mean MSE')
            ax.legend()
    fig.savefig(OUT/'bauer_families.png', dpi=180); plt.close(fig)
    sections += [image(OUT/'bauer_families.png', '按父状态族分层：训练与开发均存在偏差，正向／负向类别最突出。柱为三seed与父状态的描述性平均，无独立置信区间。')]
    # Check all six saved models, using exactly the corresponding real parent.
    ref_rows = {r['parent_id']:r for split in ['train','dev'] for r in records(config(), split)}
    for p in sorted(root.glob('w*')):
        with h5py.File(p/'generated.h5') as h:
            for split in h:
                for pid in h[split]:
                    row = ref_rows[pid]
                    with h5py.File(row['path']) as ref:
                        assert np.allclose(ref['time_ps'][:2], [0., .1])
                        r = ref['spins'][:, :2].astype(float)
                        assert len(np.unique(ref['seeds'][:])) == len(r)
                    x = h[split][pid][:]
                    assert np.allclose(x[:, 0], r[0, 0], atol=2e-6)
                    forensic.append(dict(model=p.name, split=split, family=row['family'], parent_id=pid, **local_metrics(x, r)))
    dump(OUT/'bauer_forensic.json', dict(rows=forensic, scientific_certification=False,
         limitation='Empirical branch means; no finite-sample correction of alignment or persistence ratios; reused fit/selection data.'))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, key, title in zip(axs, ['mean_tangent_cosine','tangent_amplitude_ratio','variance_ratio'],
                             ['Mean tangent direction cosine','Mean tangent amplitude / reference','Generated variance / reference']):
        values=[]
        for family in families:
            group=[r for r in forensic if r['split']=='dev' and r['family']==family]
            values.append(np.mean([r['generated_variance']/r['reference_variance'] if key=='variance_ratio' else r[key] for r in group]))
        ax.bar(range(6),values,color='#0072B2',edgecolor='black',linewidth=.4)
        ax.axhline(1,color='black',ls='--',lw=1);ax.set_xticks(range(6),families,rotation=35,ha='right')
        ax.set_title(title)
    fig.savefig(OUT/'bauer_direction_variance.png',dpi=180);plt.close(fig)
    sections.append(image(OUT/'bauer_direction_variance.png','开发集六模型描述性汇总：方向余弦与切向幅度、随机方差。理想对应为1；参考均值有限样本有噪声，不能当作无偏精度估计。'))
    sections.append('<p>正负态绝对误差较大，但其真实运动幅度也更大；因此不能只按绝对MSE排序难度。当前正负态同时呈现切向平均增量偏小、随机方差偏大的迹象；并非简单的所有运动都太快，也不是所有分布都模式塌缩。</p>')
    sections += ['<p>追加核验：保存轨迹与对应父状态匹配、0.1 ps时间间隔和参考seed唯一性；新增切向平均增量方向、幅度、随机方差及不演化基线诊断。</p>',
        '<p>0.1 ps约对应7.6个内部时间单位（hbar/J约0.01316 ps），不是无限小时间漂移；局部传播范围、频率混叠与网络上下文仍需专门实验。</p>',
        '<p>仍未排除：参考积分／物理时间换算的共同系统误差、局部结构表达不足、优化冲突、多时间尺度信息损失。模长守恒不能排除这些问题。</p>',
        '<h3>固定四父状态拟合与损失权重敏感性</h3><p>L16，正向／负向／倾斜／单畴壁各一个训练父状态；三seed，权重0和10；相同起点checkpoint与FM样本流。权重10是诊断干预，不是定稿。报告方差，避免均值改善掩盖模式塌缩。</p>']
    if (HERE/'symmetry.json').exists():
        symmetry=json.loads((HERE/'symmetry.json').read_text())['rows']
        sections.append('<h3>已发现的结构问题：非二次幂尺寸的链反射偏差</h3><p>当前逐层二分粗粒化在遇到奇数长度时补末位，打破开放链左右反射对称性。下面是固定随机状态的网络输出相对RMS误差，不是轨迹物理统计误差。</p><table><tr><th>L</th><th>三seed平均反射相对误差</th></tr>')
        for n in [16,25,32,50,100]:
            value=np.mean([r['reflection_relative_rms'] for r in symmetry if r['L']==n])
            sections.append(f'<tr><td>{n}</td><td>{value:.4g}</td></tr>')
        sections.append('</table><p>双向平均粗粒化候选算子已补充，尚未用于训练或替换checkpoint。此缺陷不能单独解释L16偏差。后续必须比较修正前后的条件统计，而非仅使对称性测试变绿。</p>')
    for name in [f'w{w}_{s}' for w in [0,10] for s in [11,22,33]]:
        p = HERE/'bauer'/name/'report.json'
        if not p.exists():
            sections.append('<p>'+name+'：尚无完成报告，不推断运行或通过。</p>'); continue
        result = json.loads(p.read_text())
        before, after = [np.mean([r['mean_increment']['raw_mse'] for r in result[k]]) for k in ['before','after']]
        vg=np.mean([r['generated_variance'] for r in result['after']]);vr=np.mean([r['reference_variance'] for r in result['after']])
        sections.append(f'<p>{name}：同训练父状态局部均值原始MSE {before:.5g} → {after:.5g}；最终平均随机方差比 {vg/vr:.3f}。仅拟合诊断，不能证明新父状态泛化。</p>')
    sections += ['<p>固定少量状态能拟合，支持进一步检查全库曝光分配、容量与目标权重；不证明已解决泛化。禁止将这些特化checkpoint直接视为正式10 ns模型。</p>',
        '<h2>二维：新训练库上的实际训练与生成</h2>',
        '<p>12.5 K，开放三角晶格单层；2是原胞内两个共面基元，不是两层。每尺寸12训练父状态、4开发父状态；保留4测试父状态不读取。完整空间、1 ps路径块、1000次更新；每开发父状态8条生成，与8条参考比较。</p>',
        '<p>复用冻结的局部两跳PathNet；尚非最终多尺度＋时间注意力模型。三个尺寸分别训练，不是尺寸泛化实验。单训练seed先导，不代表稳定性结论。</p>']
    summary = []
    for i in range(3):
        path = HERE/'miltat'/str(i)
        if not (path/'report.json').exists():
            sections.append(f'<p>尺寸索引{i}：尚无完成报告，不以旧烟测替代。</p>'); continue
        report = json.loads((path/'report.json').read_text()); shape = report['shape']
        sections.append(f'<h3>{shape[0]} × {shape[1]} 原胞</h3><p>结构检查：{report["checks"]["passed"]}；模长误差 {report["norm_error"]:.3g}；耗时 {report["seconds"]:.1f}秒。通过几何约束不代表分布准确。</p>')
        with h5py.File(path/'generated.h5') as h:
            ids = sorted(h)
            for pid in ids:
                g = h[pid]; x, r = g['generated'][:], g['reference'][:]
                summary.append(dict(shape=shape, parent_id=pid,
                    mean_field_mse=float(((x.mean(0)-r.mean(0))**2).mean()),
                    generated_increment_power=float(np.mean(np.sum(np.diff(x,axis=1)**2,axis=-1))),
                    reference_increment_power=float(np.mean(np.sum(np.diff(r,axis=1)**2,axis=-1)))))
            g = h[ids[0]]; x, r, pos, t = [g[k][:] for k in ['generated','reference','positions_nm','time_ps']]
        fig, axs = plt.subplots(1,3, figsize=(13,3.6), constrained_layout=True)
        axs[0].plot(report['loss']); axs[0].set(xlabel='Update',ylabel='Flow loss',title='Training')
        for data,label,color in [(r,'LLG','#0072B2'),(x,'Generated','#D55E00')]:
            power = np.sum(np.diff(data,axis=1)**2,axis=-1).reshape(8,10,-1).mean((0,2))
            axs[1].plot(t[1:],power,label=label,color=color)
            norm = abs(np.linalg.norm(data,axis=-1)-1).reshape(8,11,-1).max((0,2))
            axs[2].plot(t,norm,label=label,color=color)
        axs[1].set(xlabel='Time (ps)',ylabel='Mean squared spin increment',title='Local motion amplitude');axs[1].legend()
        axs[2].set(xlabel='Time (ps)',ylabel='Max norm error',title='Unit-spin constraint');axs[2].legend()
        png = OUT/f'miltat_{i}.png';fig.savefig(png,dpi=180);plt.close(fig)
        sections.append(image(png,'固定顺序的首个开发父状态：loss下降与模长保持不等于局部运动幅度正确。'))
        fig, axs = plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
        xy = pos.reshape(-1,3); artists=[]
        for row, (data,label) in enumerate([(r,'LLG reference'),(x,'Generated')]):
            for c in range(3):
                ax=axs[row,c]
                artist=ax.scatter(xy[:,0],xy[:,1],c=data[0,0,...,c].ravel(),s=3,cmap='coolwarm',vmin=-1,vmax=1,linewidths=0)
                ax.set_aspect('equal');ax.set(xlabel='x (nm)',ylabel='y (nm)',title=label+' / s'+'xyz'[c])
                fig.colorbar(artist,ax=ax,shrink=.8);artists.append((artist,data,c))
        title=fig.suptitle('Illustrative branch 0; independent noise, not paired prediction')
        def animate(k):
            for artist,data,c in artists:artist.set_array(data[0,k,...,c].ravel())
            title.set_text(f't = {t[k]:.1f} ps; independent illustrative branches (not paired targets)')
        animate(10);fig.savefig(OUT/f'miltat_components_{i}.png',dpi=160)
        animation=FuncAnimation(fig,animate,frames=11,interval=180)
        gif=OUT/f'miltat_{i}.gif';animation.save(gif,writer=PillowWriter(fps=5),dpi=100);plt.close(fig)
        sections.append(image(gif,'上：真实LLG；下：生成模型。三列是sx、sy、sz，统一[-1,1]颜色。展示独立随机分支，不要求逐像素同轨迹。'))
    dump(OUT/'miltat_metrics.json',dict(rows=summary,scientific_certification=False))
    sections.append('<h3>二维开发父状态汇总</h3><table><tr><th>原胞尺寸</th><th>生成/参考局部运动功率比</th></tr>')
    for shape in sorted({tuple(r['shape']) for r in summary}):
        rows=[r for r in summary if tuple(r['shape'])==shape]
        ratio=np.mean([r['generated_increment_power'] for r in rows])/np.mean([r['reference_increment_power'] for r in rows])
        sections.append(f'<tr><td>{shape}</td><td>{ratio:.3f}</td></tr>')
    sections.append('</table><p>每尺寸4个开发父状态，生成与参考各8条；比值大于1表示相邻保存帧之间局部运动幅度偏大。不是速度加速比。二维先导尚未学准路径统计。</p>')
    sections.append('<h2>排查边界与下一步</h2><ul><li>已核查：时间/父状态配对、独立seed、流插值导数、切向更新、模长、开放边界、旋转协变、采样一致性、已有积分步数对照。</li><li>本轮发现：固定少量状态可明显改善拟合；均值与方差有不同偏差；非二次幂尺寸的粗粒化反射缺陷。</li><li>尚待：修正粗粒化后的受控重训、丰富状态训练曝光与优化消融、更细物理保存时间、独立参考数值误差、多时间尺度/远程信息、全时空协方差和事件概率、最终独立测试。</li><li>二维本轮测试不含最终多尺度/时间注意力架构、尺寸迁移或长时扩散认证；周期平移是网络软件测试，不代表已在周期边界数据上学会动力学。</li></ul>')
    sections.append('<p>下一步：根据误差定位决定模型修改。此页不声称已复现文献概率、长期扩散或大尺寸迁移。</p>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最新一二维开发诊断</title><style>body{max-width:1200px;margin:30px auto;padding:20px;font:17px/1.7 sans-serif;color:#203040}img{width:100%;height:auto}figure{margin:25px 0}h2{border-bottom:2px solid #ddd}a{color:#006b92}</style><body>'+''.join(sections)+'</body></html>'
    (OUT/'index.html').write_text(page)
    print('Report:',OUT/'index.html','forensic rows',len(forensic),'2D parents',len(summary))


if __name__ == '__main__': main()
