"""Build an honest current-result section; pending jobs never become results."""
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
from PIL import Image
from tests.bauer_two_stage_20260919.common import dump

HERE = Path(__file__).resolve().parent
OUT = HERE/'assets'


def image(path, caption):
    with Image.open(path) as im: frames = getattr(im,'n_frames',1)
    mime = 'gif' if path.suffix=='.gif' else 'png'
    return f'<figure><img loading="lazy" data-frames="{frames}" alt="{html.escape(caption)}" src="data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def table(headers, rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'


def animate_2d(ref, gen, times, positions, path):
    # One branch, actual saved frames; identical component color scales.
    xy = positions.reshape(-1,3); fig, axes = plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    artists = []
    for row,(s,label) in enumerate([(ref,'LLG reference'),(gen,'Generated')]):
        for c in range(3):
            ax = axes[row,c]
            artist = ax.scatter(xy[:,0],xy[:,1],c=s[0,...,c].ravel(),s=3,cmap='coolwarm',vmin=-1,vmax=1,linewidths=0)
            ax.set_aspect('equal'); ax.set(xlabel='x (nm)',ylabel='y (nm)',title=label+' / s'+'xyz'[c])
            fig.colorbar(artist,ax=ax,shrink=.8); artists.append((artist,s,c))
    title = fig.suptitle('Independent noise; not pixelwise reconstruction')
    def update(k):
        for artist,s,c in artists: artist.set_array(s[k,...,c].ravel())
        title.set_text(f't = {times[k]:g} ps; independent illustrative branches')
    animation = FuncAnimation(fig,update,frames=len(times),interval=180)
    animation.save(path,writer=PillowWriter(fps=5),dpi=90); plt.close(fig)


def short_results(parts):
    rows = []
    for name in ['existing_0','existing_1','existing_2','local_dev','local_test','full_dev','full_test']:
        path = HERE/'evaluation'/f'{name}.json'
        if not path.exists():
            if not name.startswith('existing'): rows.append([name,'等待任务完成','—','—','—','—'])
            continue
        report = json.loads(path.read_text()); values = report['rows']
        mean = lambda key: float(np.mean([r[key] for r in values]))
        rows.append([name,len(values),f'{mean("increment_power_ratio"):.3f}',f'{mean("increment_cov_rmse"):.4g}',
                     f'{mean("energy_mean_rmse"):.4g}',f'{mean("Q_mean_rmse"):.4g}'])
        first = report['series'][0]; t = np.array(first['time_ps'])
        fig, axs = plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        for ax,key,label in zip(axs.flat,['mz','energy','Q','overlap','nearest_corr','increment_power'],
                               ['Mean magnetization','Energy / (N J)','Open-boundary Q','Initial overlap','Nearest-neighbor correlation','Increment power']):
            for source,color,title in [('reference','#0072B2','LLG'),('generated','#D55E00','Generated')]:
                v = np.asarray(first[source][key]); tt = t[1:] if key=='increment_power' else t
                if v.ndim==2:
                    ax.plot(tt,v.mean(0),color=color,label=title)
                    lo,hi = np.quantile(v,[.1,.9],axis=0); ax.fill_between(tt,lo,hi,color=color,alpha=.2)
                else: ax.plot(tt,v,color=color,label=title)
            ax.set(xlabel='Time (ps)',ylabel=label); ax.legend()
        png = OUT/f'{name}.png'; fig.savefig(png,dpi=160); plt.close(fig)
        parts.append(image(png,f'{name}：固定首个父状态，曲线为分支均值，色带为经验10–90%分位而非置信区间。完整父状态指标另存JSON；单图不代表所有状态。'))
        if name in ['full_test','local_test']:
            gif = OUT/f'{name}.gif'
            with h5py.File(HERE/'evaluation'/f'{name}.h5') as h:
                g = h[first['parent_id']]
                animate_2d(g['reference'][0],g['generated'][0],g['time_ps'][:],g['positions_nm'][:],gif)
            parts.append(image(gif,f'{name}：真实与生成的三个磁矩分量，展示独立随机轨迹；不是逐像素预测。'))
    parts.insert(0, table(['二维实验','父状态数','增量功率比≈1','增量协方差RMSE↓','能量均值RMSE↓','拓扑荷均值RMSE↓'],rows))
    parts.append('<p>能量含交换、各向异性、DMI和开放边界全偶极项，只用于评价。Q采用定向三角形固角求和，开放边界下不强制整数；完整JSON保留核消失、触边和背景失效状态。空间谱为矩形原胞倒空间坐标，不冒充各向同性径向谱。</p>')


def literature_results(parts):
    from tests.bauer_fig2_10ns_20260917.diagnose import events
    for name in ['bauer','miltat_local','miltat_full']:
        root = HERE/'literature'/name; path = root/'generated.h5'
        parts.append(f'<h3>文献条件模型生成：{name}</h3>')
        if not (root/'complete.json').exists():
            parts.append('<p class="warning">尚无完整生成结果；不能用LLG参考图冒充模型输出。任务安排见上表。</p>'); continue
        if name=='bauer':
            with h5py.File(path) as h:
                if not h.attrs['complete']: raise RuntimeError('Completion marker conflicts with data')
                x = h['spins'][:]; t = h['time_ps'][:]/1000
            source = HERE.parent/'bauer_fig2_10ns_20260917/results/trajectory.h5'
            with h5py.File(source) as h: ref = h['spins'][:]
            cut_report = json.loads((source.parent/'diagnostics_events/summary.json').read_text())
            threshold = cut_report['basin_threshold']; event_rows = []
            for branch in range(3):
                fig, axs = plt.subplots(2,2,figsize=(14,7),constrained_layout=True)
                for col,(s,label) in enumerate([(ref,'LLG reference'),(x,'Generated')]):
                    mz = s[branch,...,2].mean(-1)
                    axs[0,col].plot(t,mz,lw=.6); axs[0,col].axhline(threshold,ls='--',color='grey'); axs[0,col].axhline(-threshold,ls='--',color='grey')
                    axs[0,col].set(title=label+f' / branch {branch}',xlabel='Time (ns)',ylabel='Mean mz')
                    im = axs[1,col].imshow(s[branch,::10,:,2].T,aspect='auto',origin='lower',extent=(0,10,0,100),cmap='coolwarm',vmin=-1,vmax=1)
                    axs[1,col].set(xlabel='Time (ns)',ylabel='Site'); fig.colorbar(im,ax=axs[1,col])
                    ev = events(t,mz,threshold,20)
                    event_rows.append(dict(source=label,branch=branch,negative_endpoint=bool(mz[-1]<0),crossed_zero=bool(np.any(mz<0)),
                        committed_switches=sum(e['kind']=='committed_switch' for e in ev),events=ev))
                png = OUT/f'bauer_10ns_{branch}.png'; fig.savefig(png,dpi=170); plt.close(fig)
                parts.append(image(png,'左：LLG；右：修正版模型。整段10 ns不切断，空间图每1 ps显示一帧，完整0.1 ps数据保存在H5。相同初态、独立噪声，不要求事件时刻逐一相同。'))
            dump(OUT/'bauer_literature_events.json',dict(rows=event_rows,threshold=threshold,hold_ps=20,
                 note='Exploratory reference-derived threshold, not certified basin distribution. Three paths do not certify probabilities.'))
            fig, axs = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
            for s,label,color in [(ref,'LLG','#0072B2'),(x,'Generated','#D55E00')]:
                mz = s[...,2].mean(-1)
                axs[0].hist(mz[:,::10].ravel(),bins=np.linspace(-1,1,61),density=True,histtype='step',label=label,color=color)
                error = abs(np.linalg.norm(s,axis=-1)-1).max((0,2)); axs[1].plot(t,error,label=label,color=color)
            axs[0].set(xlabel='mz',ylabel='Time-occupancy density (correlated frames)'); axs[1].set(xlabel='Time (ns)',ylabel='Max norm error')
            for ax in axs: ax.legend()
            png = OUT/'bauer_literature_statistics.png'; fig.savefig(png,dpi=170); plt.close(fig)
            parts.append(image(png,'占据分布是时间统计，不是独立终态样本分布；仅三条轨迹的终态不足以作稳定直方图。模长保持不是动力学准确性的证明。'))
        else:
            from tests.miltat_library_20260920.geometry import CoreGeometry
            source = HERE.parent/'miltat_1ns_20260919/results/thermal.h5'
            with h5py.File(path) as h: x = h['spins'][0,::100]; t = h['time_ps'][::100]; positions = h['positions_nm'][:]
            with h5py.File(source) as h: ref = h['spins'][::10,0]
            geom = CoreGeometry(positions); measured = {}
            for s,label in [(ref,'reference'),(x,'generated')]: measured[label] = [geom.measure(frame) for frame in s]
            dump(OUT/f'{name}_core.json',dict(time_ps=t.tolist(),**measured,
                 note='One trajectory, finite open window, no diffusion coefficient or probability certification. Invalid cores retained.'))
            fig, axs = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
            for s,label,color in [(ref,'reference','#0072B2'),(x,'generated','#D55E00')]:
                axs[0].plot(t,s.reshape(len(s),-1,3)[...,2].mean(-1),label=label,color=color)
                centers = np.array([r['isolated_core_center_nm'] if r['isolated_core_center_nm'] is not None else [np.nan,np.nan] for r in measured[label]])
                axs[1].plot(centers[:,0],centers[:,1],label=label,color=color)
                valid = np.isfinite(centers).all(1); displacement = np.full(len(centers),np.nan)
                if valid[0]: displacement[valid] = ((centers[valid]-centers[0])**2).sum(-1)
                axs[2].plot(t,displacement,label=label,color=color)
            axs[0].set(xlabel='Time (ps)',ylabel='Mean mz'); axs[1].set(xlabel='Core x (nm)',ylabel='Core y (nm)')
            axs[2].set(xlabel='Time (ps)',ylabel='Single-path displacement squared (nm²)')
            for ax in axs: ax.legend()
            png = OUT/f'{name}_1ns.png'; fig.savefig(png,dpi=170); plt.close(fig)
            parts.append(image(png,'单条模型路径与单条参考路径：磁化、有效孤立核的位置和单路径位移平方。断线表示核诊断失效；不是多轨迹MSD，不据此拟合扩散系数。'))
            gif = OUT/f'{name}_1ns.gif'; animate_2d(ref[::2],x[::2],t[::2],positions,gif)
            parts.append(image(gif,'完整1 ns范围取51个真实保存帧：上LLG，下生成；sx/sy/sz使用统一颜色。完整生成数据每0.1 ps保存。'))


def main():
    OUT.mkdir(exist_ok=True); plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    parts = ['<section id="latest-campaign"><h2>当前任务：一维修正、二维完整骨干、文献条件生成</h2>',
      '<p>以下结果直接在主网页展示。历史报告在后；未生成的结果明确标记为等待，不能认为提交即成功。</p>']
    submission = HERE/'submission.json'
    if submission.exists():
        jobs = json.loads(submission.read_text())['jobs']
        parts.append(table(['任务','Slurm ID','预计计算时间','提交时状态/说明'],[[j['name'],j['id'],j['estimate'],j['note']] for j in jobs]))
    parts.append('<p>一维使用整个向量场的左右反转群平均修正，保留旧checkpoint兼容性，计算量增加；不是已经证明统计修复。二维完整候选含局部图聚合、有向原胞差分（保留手性辨识信息）、共享多尺度位移聚合、全局均值上下文、逐格点时间注意力、FiLM和球面切向流；它不是解析长程场求解器，长波准确性仍需验证。没有输入J/D/K或有效场。该候选目前针对二维实现，不声称已完成三维通用训练。</p>')
    parts.append('<p>对照：一维old/fixed × 3 seed，同起点、相同FM样本、1500更新、均值损失权重10；该权重是诊断候选而非已证最优。二维local/full × 3 seed，同三尺寸全空间数据和1500更新；不同参数量和计算成本单列。均按开发集选checkpoint，既有测试集作为诊断，未来正式结论需要新冻结holdout。</p>')
    for system in ['bauer','miltat']:
        rows = []
        for path in sorted((HERE/system).glob('*/complete.json')):
            result = json.loads(path.read_text()); rows.append([path.parent.name,f'{result["best_dev_score"]:.4g}',f'{result["seconds"]/60:.1f}'])
        parts.append('<h3>'+system+' 三seed训练结果</h3>'+table(['运行','最佳开发分数↓（选模启发式）','总分钟'],rows))
    metrics = HERE/'bauer_evaluation/metrics.json'
    if metrics.exists():
        rows = json.loads(metrics.read_text())['rows']; fig,axs = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for ax,key,title in zip(axs,['increment_mean','increment_cov','temporal_corr'],['Increment mean RMSE','Increment covariance RMSE','Initial-overlap RMSE']):
            lengths = [16,25,32,50,100]
            for source,offset,color,label in [('raw',-.18,'#D55E00','Model/reference'),('reference_floor',.18,'#0072B2','Reference/reference')]:
                values = [np.mean([r[source][key] for r in rows if r['L']==n and r['horizon_ps']==10]) for n in lengths]
                ax.bar(np.arange(5)+offset,values,.36,color=color,label=label,edgecolor='black',linewidth=.4)
            ax.set_xticks(range(5),lengths); ax.set(xlabel='Chain length',ylabel=title); ax.legend()
        png = OUT/'bauer_size_ladder.png'; fig.savefig(png,dpi=170); plt.close(fig)
        parts.append(image(png,'修正版10 ps五级尺寸诊断；参考/参考为独立噪声分支的有限样本基线。柱为六父状态描述性均值，不是自动认证。'))
    short_results(parts); literature_results(parts)
    parts.append('''<h2>借鉴MDGen：损失、评价与轨迹补全</h2>
<p><a href="https://arxiv.org/html/2409.17808v1">MDGen原文</a>支持前向、插值、时间上采样及分子部分补全等任务；其报告中不同实验设置使用分别训练的模型，不能理解成任意前向checkpoint天然具有所有能力。MDGen的分子部分inpainting也不完全等同于本项目的连续时间缺口补全。</p>
<h3>当前能力边界</h3><p>当前训练模型只支持首帧条件的前向生成；一维转移模型仅两帧，没有中间未知帧，不能直接用于桥接。二维候选虽联合生成11帧，也尚未进行观测掩码多任务训练。本轮只实现并测试了无泄漏观测掩码与未知token损失接口，不声称已训练成功桥接模型。</p>
<p>要实现：采用具有多个未知中间帧的联合路径模型，把观测值和观测mask分别编码并注入各残差块/时间交互；训练混合首帧、首末帧、稀疏中间帧、连续缺口、空间遮挡任务。未知值必须清零、不可偷看目标；每次球面更新后恢复已知单位磁矩；只在未知token上计算流匹配损失。对当前首帧专用骨干还需修改条件编码，不能只在采样器里clamp末帧。</p>
<h3>推荐损失</h3>
<p>主损失 L_FM = E[Σ未知token ‖vθ − u球面插值‖² / 未知token数]。模长与已知帧用硬约束。辅助项首选同初态独立分支的条件均值匹配，方差/协方差、多滞后相关、低波数结构谱可作为后续单独受控消融；权重只用训练/开发集确定，不把所有指标盲目相加。</p>
<p>禁止把不同噪声的生成与真实路径逐帧MSE作为主要物理目标：它会偏向均值预测。有限温且有阻尼的体系不要求能量守恒，也不应把无噪声LLG残差压到零。若使用有效场或涨落耗散关系作教师损失，须单列“物理教师监督”消融，不能仍声称完全只从样本学习。</p>
''')
    parts.append(table(['层级','优先指标','解释/限制'],[
      ['条件局部随机演化','条件增量均值、切空间协方差、角位移分布、多分支energy score','同一父状态比较；有限保存间隔不是瞬时漂移/扩散系数'],
      ['时间联合规律','C(t,t′)、增量多滞后相关、条件路径特征energy distance','非平衡过程不预设平稳；终态正确不能替代时间结构'],
      ['空间与机制','连通相关、低q结构谱、畴壁/核位置、Q及失效状态','看大体系新增长波、传播和机制；不能只看平均磁化'],
      ['事件概率','首次到达/生存曲线、承诺翻转/返回、成核位置及机制频率','未发生事件保留右删失；需要足够独立路径，三条只是演示'],
      ['分布与校准','W1/JSD、路径特征距离、覆盖率与条件多样性','JSD分箱预先冻结；投影指标接近不证明完整分布相等'],
      ['成本','固定精度与时长的端到端生成耗时、有效独立样本/秒、训练摊销','不能以更大统计偏差交换速度后仍宣称同精度加速'],
      ['桥接/补全','观测帧误差、缺口条件分布、桥接机制权重和不确定性校准','不能用无条件前向参考作末态条件金标准；精确连续终态需专门桥接参照或预定义容差协议']]))
    parts.append('<p>桥接模型学的是 P[缺失路径 | 所有观测]，不是恢复唯一真实热噪声历史。它可用于稀疏磁成像的概率补全、在指定起末态之间分析可能的成核/畴壁机制、对缺失测量量化不确定性。指定终态会改变路径权重，因此桥接样本不能直接给出无条件翻转率；还需终态发生概率或正确重加权。</p><p>验收应报告独立父状态/准备批次与训练seed的稳定性、参考对参考的采样底噪和数值积分误差。开发综合分数仅用于选模，不是一个具有独立物理意义的“认证分数”。</p></section>')
    fragment = ''.join(parts); (OUT/'fragment.html').write_text(fragment)
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>文献条件路径实验</title><style>body{max-width:1200px;margin:auto;padding:20px;font:16px/1.7 sans-serif}img{width:100%}td,th{padding:10px;border-bottom:1px solid #ddd}.scroll{overflow:auto}.warning{background:#fff4d6;padding:12px}</style><body>'+fragment+'</body></html>')
    print('Updated',OUT/'index.html')


if __name__ == '__main__': main()
