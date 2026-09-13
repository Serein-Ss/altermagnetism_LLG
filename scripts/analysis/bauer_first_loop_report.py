"""Reviewable first-loop figures, complete comparison CSV, and Chinese decision report."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.workflow.bauer_campaign import observables


def read(path):return json.loads(Path(path).read_text())


def run(root):
    root=Path(root);cfg=read(root/'config.json');decision=read(root/'decision.json')
    metrics=read(root/'distribution.json');events=read(root/'events_and_paths.json')
    training=read(root/'training.json');sampling=read(root/'sampling.json');completed=read(root/'completed.json')
    ed={}
    for label,estimate,ci in zip(metrics['labels'],metrics['estimate'],metrics['simultaneous_ci']):
        if label.endswith('/ed_excess'):ed[label.rsplit('/',1)[0]]=(estimate,*ci)
    rows=[]
    for e in events:
        theta=e['theta'];name=e['model'];key=f'T{theta}/{name}'
        if key not in ed:continue
        k,s=name.split(':');sample=next((r for r in sampling['models'] if r['kind']==k and r['seed']==int(s) and r['theta']==theta),{})
        r=dict(theta=theta,model=name,ed_excess=ed[key][0],ci_low=ed[key][1],ci_high=ed[key][2],
            reference_reversal=e['reference_reversal'],generated_reversal=e['generated_reversal'],
            zero_generated_event_upper95=1-.05**(1/(cfg['test_initials']*cfg['noise_per_half'])) if e['generated_reversal']==0 else '',
            rmst_difference_reduced=e['event_difference'][2]*16000,
            reference_nn_correlation=e['reference_nn_correlation'],generated_nn_correlation=e['generated_nn_correlation'],
            reference_increment_variance=e['reference_increment2'],generated_increment_variance=e['generated_increment2'],
            norm_max=sample.get('norm_max',''),sampling_seconds=sample.get('seconds',''))
        rows.append(r)
    with open(root/'all_comparisons.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    out=root/'figures';out.mkdir(exist_ok=True)
    colors=dict(reference='black',rfm='#0072B2',deterministic='#D55E00',euclidean='#CC79A7',autoregressive='#009E73',retrieval='#E69F00')
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for j,theta in enumerate(cfg['theta']):
        with h5py.File(root/f'reference_T{theta}.h5') as h:p=h['spins'][:,:cfg['noise_per_half']]
        series={'reference':p.reshape(-1,201,25,3)}
        for k in cfg['kinds']+['retrieval']:
            s=11 if k!='retrieval' else 0;p=np.load(root/f'{k}_{s}_T{theta}.npy')
            if np.isfinite(p).all():series[k]=p.reshape(-1,201,25,3)
        for name,paths in series.items():
            v=observables(paths);t=np.arange(201)*80
            axes[j,0].plot(t,v[...,3].mean(0),label=name,color=colors[name])
            axes[j,1].plot(t,v[...,0].mean(0),label=name,color=colors[name])
            axes[j,2].plot(t,v[...,4].mean(0),label=name,color=colors[name])
        for ax in axes[j]:ax.set_xlabel('Time (hbar/J)');ax.set_title(f'theta={theta}; fixed seed11');ax.grid(alpha=.2)
        axes[j,0].set_ylabel('Mean Mz/N');axes[j,1].set_ylabel('Mean energy/spin');axes[j,2].set_ylabel('Nearest-neighbor correlation')
        axes[j,0].set_ylim(-1.05,1.05)
    axes[0,0].legend(ncol=2,fontsize=8)
    fig.suptitle('Independent reference vs generated paths; seed11 fixed before evaluation')
    for ext in ('png','pdf'):fig.savefig(out/f'physical_observables.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,theta in zip(axes,cfg['theta']):
        subset=[r for r in rows if r['theta']==theta]
        for j,r in enumerate(subset):
            k=r['model'].split(':')[0]
            ax.errorbar(j,r['ed_excess'],yerr=[[max(0,r['ed_excess']-r['ci_low'])],[max(0,r['ci_high']-r['ed_excess'])]],
                        fmt='o',color=colors[k],capsize=2)
        ax.set_xticks(range(len(subset)),[r['model'] for r in subset],rotation=70,ha='right',fontsize=8)
        ax.axhline(.1,color='black',ls='--',label='Screening threshold0.10');ax.axhline(0,color='grey',lw=.5)
        ax.set_yscale('symlog',linthresh=.1);ax.set_title(f'theta={theta}');ax.set_ylabel('Excess energy distance (44 features)');ax.legend(fontsize=8)
    fig.suptitle('All seeds and baselines; simultaneous bootstrap intervals, exploratory small sample')
    for ext in ('png','pdf'):fig.savefig(out/f'distribution_comparison.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,theta in zip(axes,cfg['theta']):
        subset=[r for r in rows if r['theta']==theta]
        ax.bar(range(len(subset)),[r['generated_reversal'] for r in subset],color=[colors[r['model'].split(':')[0]] for r in subset])
        ax.axhline(subset[0]['reference_reversal'],color='black',ls='--',label='Independent R1')
        ax.set_xticks(range(len(subset)),[r['model'] for r in subset],rotation=70,ha='right',fontsize=8)
        ax.set(ylim=(0,1),ylabel='Completed reversal fraction',title=f'theta={theta}');ax.legend()
    fig.suptitle('Descriptive event fractions; zero events do not prove zero probability')
    for ext in ('png','pdf'):fig.savefig(out/f'reversal_fractions.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,theta in zip(axes,cfg['theta']):
        with h5py.File(root/f'reference_T{theta}.h5') as h:real=h['spins'][:,:cfg['noise_per_half']]
        for name,paths in [('reference',real),('rfm',np.load(root/f'rfm_11_T{theta}.npy')),
                           ('retrieval',np.load(root/f'retrieval_0_T{theta}.npy'))]:
            mz=paths[:,:,-1,:,2].mean(-1).ravel()
            ax.hist(mz,bins=np.linspace(-1,1,21),density=True,histtype='step',lw=2,color=colors[name],label=name)
        ax.set(xlabel='Final Mz/N',ylabel='Empirical density',title=f'theta={theta}; pre-fixed seed11')
        ax.legend()
    fig.suptitle('Post-hoc diagnosis: a plausible mean does not establish a correct mixture')
    for ext in ('png','pdf'):fig.savefig(out/f'final_magnetization_distribution.{ext}',dpi=220)
    plt.close(fig)
    failures=[r for r in decision['accuracy_checks'] if r['status']=='FAIL']
    lines=['# Bauer 铁磁链第一研究闭环', '',
        f'**第一研究闭环已完成。Bauer参考物理支持开展实验；当前RFM配置未通过可行性筛查，停止扩算（{decision["status"]}）；强方法创新尚未建立。** 这是固定预算的探索性实验，不是正式P3/P5认证。', '',
        '## 本轮真正得到的结论', '',
        '1. 可以用很小的计算预算完成真实训练和独立检验；不需要先等待全部寿命扫描。',
        '2. RFM的模长误差约1.2e-7、初态误差为零，但低温三个seed均为0/64反转，对照为16/64；高温三个seed为1/64、15/64、17/64，对照为39/64。几何正确不等于反转统计正确。',
        '3. RFM的excess ED点估计约1.13–4.15，轨迹重采样约0.001–0.031。重采样明显更有竞争力，但其宽区间和反转率偏差也不支持把它直接认证为真值替代。',
        '4. 事后检查发现，RFM末态磁化方差约0.05–0.11，而独立参考约0.50。结合较弱空间关联，这与没有保留集体反转/双盆混合相符；尚未做单因素消融，不能断言唯一原因是网络结构或训练步数。',
        '5. 同一潜变量从32步增加到64步没有恢复正确反转率；高温主特征均值差的最大值约0.080–0.158，仍超过0.05的点估计筛查线。这不能用“再多走一点运输步”直接解释完。',
        '6. 下一项值得检验的假设是如何表达链级集体反转与微观涨落的联合变化；在小规模开发集上辨别表达能力、训练不足与源分布问题，再决定是否开新的独立测试。现在不扩大同一配置的数据或AM任务。', '',
        '## 研究问题与实际执行', '',
        '固定完整初态后，RFM 能否生成既保持自旋几何约束、又保留真实反转概率及路径统计的样本，并提供超过简单基线的价值？', '',
        '- L=25，K/J=0.1，lambda=0.1，theta=0.11/0.13，窗口16000 hbar/J。',
        '- 192条既有路径训练，64条分离初态的开发数据归档；模型固定600步，不通过独立评价集选模型。',
        '- 每温度4个新完整初态，每初态R1/R2各16条独立未来噪声，总计256条新参考路径；G等量。',
        '- RFM、确定性、欧氏flow、随机AR分别运行seed11/22/33；同骨干、学习率、batch和更新次数，附加温度匹配的训练轨迹重采样。',
        '- 11时点能量/三分量磁化组成44维特征；训练数据定尺度；初态—噪声两级bootstrap2000次、全族近似95%同时区间。',
        '- 保存201帧；采样初态硬约束。物理时间输入为t/16000，不是将LLG物理积分步长改为1/200。', '',
        '## 独立参考上的主要比较', '',
        '| 温度 | 模型/seed | excess ED [同时95% CI] | 真反转率 | 生成反转率 | 最近邻关联 真/生成 |',
        '|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f'| {r["theta"]} | {r["model"]} | {r["ed_excess"]:.3f} [{r["ci_low"]:.3f}, {r["ci_high"]:.3f}] | {r["reference_reversal"]:.3f} | {r["generated_reversal"]:.3f} | {r["reference_nn_correlation"]:.3f}/{r["generated_nn_correlation"]:.3f} |')
    lines += ['',f'预先定义的RFM分布/均值/基线检查中，{len(failures)}项区间明确越过筛查容差。宽区间仍记证据不足，不能用不显著证明等价。', '',
        '原条件参考已有64条完成750000约化时间的轨迹，共165次完成反转；边缘成核诊断117/165，父轨迹bootstrap95%区间约[0.634,0.781]。这是操作性机制证据，非完整文献寿命认证。', '',
        '## 创新性判断', '',
        '现成RFM、球面约束、神经网络加速自旋动力学、生成过渡路径均已有先例。仅把这些组合到Bauer链，尚不足以建立强方法创新。',
        '可继续检验的差异是：不规定终态、不筛选成功反转，学习完整初态条件下的真实路径系综，同时保留反转/返回概率和时空统计。这个研究目标是否能成为贡献，还需要方法与跨条件实证支撑。',
        '逐篇来源与边界见 [NOVELTY_REVIEW.md](NOVELTY_REVIEW.md)。没有把检索未发现同名工作写成优先权证明。', '',
        '## 资源与结论边界', '',
        f'- 本次作业{completed["job_id"]}实际运行{completed["wall_seconds"]/60:.1f}分钟；单RTX3090预留90分钟。CPU Numba生成参考，GPU训练和推理。',
        '- 不宣称科学加速：数据生产和训练成本尚未按相同目标误差摊销；RFM采样计时还包含32/64步两次运行，不能直接拿来声称公平速度倍数。',
        '- 4个新初态和每半集16条噪声是筛查规模，不能认证小误差等价、稀有事件尾部、温度泛化或全路径分布相等。',
        '- 自回归基线在部分长路径上的模长误差超过1e-5，欧氏flow也有明显模长偏离；二者不被认证为合格物理模型。短路径软件测试不能替代训练后的长路径数值检查。',
        '- bootstrap区间没有在本轮做覆盖率校准；共享全族的宽误差带会降低筛查功效，因此不把某些点估计优势写成已证实的统计优势。',
        '- L=25生产dt=.02尚未完成该条件全部步长认证；结论首先针对这套离散参考。既有解析/软件测试通过不替代这一限制。',
        '- 本轮事件定义为阈值±0.6、驻留160。先导细保存间隔5与本轮间隔80对完成反转标签一致，但不由此宣称快速成核、连续首达时间或频谱已收敛。',
        '- 事件CI为探索性逐项区间；生成零事件时，在本次固定初态、64个独立噪声样本下单侧95%上界约4.57%，不写成概率严格为零。',
        '- 小预算失败只否定当前配置在当前预算下的可行性，不证明整个生成建模方向不可能。', '',
        '## 核查文件', '',
        '`config.json`、`submission.json`：冻结设计、代码哈希与Slurm编号；`training.json`、12个.pt文件：全部训练记录与检查点；',
        '`reference_T*.h5`、生成.npy文件：所有独立评价样本；`distribution.json`：全部同时区间；`events_and_paths.json`、`step_doubling.json`：事件、关联与运输步数诊断；',
        '`all_comparisons.csv`和`figures/`：直接核查的表格与图。', '',
        '新增长扫描已撤下，原数据和检查点保留。后续不得仅为了得到通过结果增加训练或修改阈值；新版本应登记改变的假设和固定预算。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();run(args.root)
