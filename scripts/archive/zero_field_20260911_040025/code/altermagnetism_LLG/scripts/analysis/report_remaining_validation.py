"""Campaign comparisons and figures from derived arrays only; no new simulations."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analysis.analyze_remaining_validation import write,clean
from scripts.core.literature_config import read_document,sha256

PAPERS=('bauer_2011','hirst_mn2au_2022','gomonay_2024','laliena_crnb3s6_2020')
LIMITATIONS={
 PAPERS[0]:'观测窗仅 200，而计划要求 750000；固定 L=100、theta=0.11，没有文献长度/温度扫描。翻转阈值尚未由磁盆分布标定，事件数只作操作性敏感性检查；不能拟合寿命标度、Arrhenius 能垒或认证机制概率。',
 PAPERS[1]:'ASD 平衡为 8³/12³/16³，而目标是 30³；只有一个平衡温度及每条件一条轨迹，不能确定 T_N 或独立重复置信区间。AFMR 仅 8³；LLB 使用文献拟合输入而非本批 ASD 拟合。畴壁尚缺尺寸/长时扫描及逐键外部核验。',
 PAPERS[2]:'色散为 32²/64²/96² 模拟晶胞，不是论文原尺寸。频谱与同代码解析式比较属于内部一致性，不是独立文献认证。仅 110 晶向、512×8 畴壁；一个非零初速度不能证明 Walker breakdown，不能将自由运动标为受驱动稳态。',
 PAPERS[3]:'分支扫描固定一个 h_y，不能复现整条临界曲线；电流轨迹固定 dx，时间步收敛不代表空间收敛。螺旋需检查局部波矢及能量的长期稳定，而非仅看整数 winding。尚缺勘误图数字化点及误差。'}


def arrays(row,aid):
    path=ROOT/'data/literature_reproduction'/row['paper']/'derived'/row['run_id']/aid/'series.npz'
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def relative(path,base):return os.path.relpath(path,base)


def compare(rows,aid):
    comparisons=[]
    groups={}
    for row in rows:
        protocol=row['protocol'];p=dict(row['parameters'])
        if protocol not in ('llb_afmr','current','wall','moving_wall'):continue
        for key in ('dt','steps','save_every'):p.pop(key,None)
        groups.setdefault((row['paper'],protocol,json.dumps(p,sort_keys=True)),[]).append(row)
    for (paper,protocol,_),group in groups.items():
        if len(group)<2:continue
        group=sorted(group,key=lambda r:r['parameters']['dt'],reverse=True)
        fine=arrays(group[-1],aid)
        key='m_a' if protocol=='llb_afmr' else ('unwrapped_center' if protocol=='current' else
            ('neel_width_over_a' if paper==PAPERS[1] else 'wall_observables'))
        for row in group[:-1]:
            a=arrays(row,aid);shape=a[key].shape
            y=fine[key].reshape(len(fine['time']),-1)
            interpolated=np.stack([np.interp(a['time'],fine['time'],y[:,i]) for i in range(y.shape[1])],axis=1).reshape(shape)
            difference=a[key]-interpolated
            comparisons.append(dict(paper=paper,protocol=protocol,run_id=row['run_id'],reference_run_id=group[-1]['run_id'],
                dt=row['parameters']['dt'],reference_dt=group[-1]['parameters']['dt'],observable=key,
                rms_difference=float(np.sqrt(np.mean(difference**2))),max_abs_difference=float(np.max(abs(difference))),
                status='descriptive_no_preregistered_tolerance',reference='finest numerical run, not ground truth'))
    return comparisons


def figure(paper,rows,aid,dest):
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);ax=ax.ravel()
    styles=('-', '--', ':')
    if paper==PAPERS[0]:
        for j,dt in enumerate(sorted({r['parameters']['dt'] for r in rows})):
            group=[r for r in rows if r['parameters']['dt']==dt];zs=[arrays(r,aid) for r in group]
            mz=np.concatenate([z['mean'][...,2] for z in zs],axis=1)
            ax[0].plot(zs[0]['time'],mz.mean(1),styles[j],label=f'dt={dt}, n={mz.shape[1]}')
            ax[1].hist(mz[-1],bins=np.linspace(-1,1,41),histtype='step',linestyle=styles[j],label=f'dt={dt}')
            for r in group:ax[2].plot(dt,r['endpoint']['mean'],'o',color=f'C{j}')
        z=arrays(rows[0],aid)
        ax[3].plot(z['time'],z['mean'][...,2],alpha=.4)
        for a in ax[:2]:a.legend()
        labels=[('Reduced time','Ensemble mean m_z'),('Endpoint m_z','Trajectory count'),('dt','Mean endpoint per seed'),('Reduced time','All 32 paths of task 0')]
    elif paper==PAPERS[1]:
        for r in rows:
            z=arrays(r,aid);p=r['parameters'];dt=p['dt'];protocol=r['protocol']
            if protocol=='equilibrium':ax[0].plot(z['time'],z['sublattice_length'][:,0],label=f'L={p["shape"][0]}, dt={dt}')
            elif protocol=='wall':ax[3].plot(z['time'],z['neel_width_over_a'][:,0],label=f'dt={dt}')
            elif protocol=='afmr':ax[1].plot(z['time'],z['m_a'][:,0,2],label=p['theta_key'])
            elif protocol=='llb_afmr' and dt==.0005:ax[2].plot(z['time'],z['m_a'][:,0,2],label=p['theta_key'])
        ax[3].axhline(31.2/.333,ls='--',color='k',label='GUIDE target')
        for a in ax:a.legend(fontsize=7)
        labels=[('Reduced time','ASD sublattice length'),('Reduced time','ASD m_a,z'),('Reduced time','LLB m_a,z (finest dt)'),('Reduced time','Neel width / a')]
    elif paper==PAPERS[2]:
        for r in rows:
            p=r['parameters']
            if r['protocol'] in ('wall','moving_wall'):
                z=arrays(r,aid);a=ax[2 if r['protocol']=='wall' else 3]
                a.plot(z['time'],z['wall_observables'][:,0,1 if r['protocol']=='wall' else 0],label=f'dt={p["dt"]}')
            else:
                for peak in r['spectral_peaks']:
                    if peak['power_fraction']<1e-5:continue
                    measured=sorted([peak['positive_peak'],-peak['negative_peak']]);pred=sorted(peak['predicted_branches'])
                    a=ax[0 if r['protocol']=='spinwave' else 1]
                    a.plot(pred,measured,'o',ms=3,alpha=.5)
        for a in ax[:2]:a.plot([0,6],[0,6],'k--')
        for a in ax[2:]:a.legend(fontsize=7)
        labels=[('Internal analytic omega','FFT omega (excited modes only)'),('Control analytic omega','Control FFT omega'),('Reduced time','Static wall tanh width / a0'),('Reduced time','Moving wall position / a0')]
    else:
        for r in rows:
            z=arrays(r,aid);p=r['parameters']
            if r['protocol']=='branch':ax[0].plot(z['center_theta'],z['gamma'],label=f'X={p["extent"]}, n={p["points"]}, ds={p["ds"]}')
            elif r['protocol']=='helix':ax[1].plot(z['time'],z['mean_wavevector'],label=f'N={p["sites"]}')
            elif r['protocol']=='current':
                ax[2].plot(p['u'],r['velocity_late'][0],'o',label=f'dt={p["dt"]}')
                ax[3].plot(z['time'],z['phase_gradient_rms_width'][:,0],label=f'u={p["u"]}, dt={p["dt"]}')
        ax[0].axhline(1.2405,color='k',ls='--',label='GUIDE target');ax[2].plot([0,1.1],[0,2.2],'k--',label='v=2u')
        ax[1].axhline(1,color='k',ls='--')
        for a in ax:a.legend(fontsize=6)
        labels=[('Center theta (rad)','Gamma'),('Reduced time','Mean q / q0'),('Signed torque coefficient u','Late fitted velocity'),('Reduced time','Phase-gradient RMS width')]
    for a,(x,y) in zip(ax,labels):a.set(xlabel=x,ylabel=y)
    fig.suptitle(paper+' — numerical diagnostics, not production certification')
    dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=160,facecolor='white');plt.close(fig)


def table(rows):
    text='| task | 协议 | 主要数值结果 |\n|---:|---|---|\n'
    for r in rows:
        if 'sampled_gamma_max' in r:value=f'Gamma max={r["sampled_gamma_max"]:.9f}; 偏差={100*r["reference_relative_error"]:.3f}%'
        elif 'endpoint' in r:value=f'mz_end={r["endpoint"]["mean"]:.6f}; crossed={r["crossed_zero_count"]}; events={r["survival_diagnostic"]["completed_events"]}'
        elif 'wall_final' in r:value=f'width/a={r["wall_final"][0]["neel_width_over_a"]:.6f}'
        elif 'late_length_by_replica' in r:value=f'late |ma|={r["late_length_by_replica"][0]:.6f}'
        elif 'afmr' in r:
            fit=r['afmr'][0];value=f'omega={fit["parameters"][1]:.7f}; RMSE={fit["residual_rmse"]:.6g}' if fit['resolved'] else 'fit unresolved'
        elif 'spectral_peaks' in r:
            selected=[p for p in r['spectral_peaks'] if p['power_fraction']>=1e-5]
            value=f'展示 {len(selected)}/4 已激发诊断点; FFT bin={r["frequency_bin"]:.5f}'
        elif 'wall' in r:value=f'v_fit={r["wall"]["velocity"][0]:.7f}; width_final={r["wall"]["observables"][-1][0][1]:.6f}'
        elif 'velocity_late' in r:value=f'v_late={r["velocity_late"][0]:.7f}; v_expected={r["velocity_expected"]:.5f}'
        else:value='q_final='+', '.join(f'{v:.5f}' for v in r['final_mean_wavevector'])
        text+=f'| {r["index"]} | {r["protocol"]} | {value} |\n'
    return text


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--analysis-id',required=True);args=parser.parse_args();aid=args.analysis_id
    out=ROOT/'output/literature_reproduction'/aid;rows=json.loads((out/'metrics.json').read_text());comparisons=compare(rows,aid)
    write(out/'comparisons.json',comparisons)
    for paper in PAPERS:
        group=[r for r in rows if r['paper']==paper];doc=read_document(ROOT/'conf/literature'/f'{paper}.yaml')
        report=ROOT/'output/literature_reproduction'/paper/aid;report.mkdir(exist_ok=False)
        dest=ROOT/'assets/literature_reproduction'/paper/aid/'figures/overview.png';figure(paper,group,aid,dest)
        notes=LIMITATIONS[paper]
        text=f'# {paper} 数值验证分析\n\n_本批 {len(group)} 项；全部量按配置约化；不是正式复现认证。_\n\n---\n\n## 📋 结论与范围\n\n状态：`inconclusive`，生产开关不变。{notes}\n\n'
        text+='## 📚 论文、公式与参数\n\n'
        text+=f'目标及来源以 [GUIDE]({relative(ROOT/"GUIDE/STRICT_LITERATURE_REPRODUCTION_PLAN.md",report)}) 和每个 run 的冻结配置为准。DOI：`{doc["provenance"]["doi"]}`。旧 reference_manifest 中部分实现状态已过期，本报告不把它当最新代码状态。\n\n'
        text+='| 原参数 | 原值 | 单位 |\n|---|---:|---|\n'
        for k,v in doc['source_parameters'].items():
            if isinstance(v,dict) and 'value' in v:text+=f'| {k} | {v["value"]} | {v["unit"]} |\n'
        text+='\n约化定义：\n\n```json\n'+json.dumps(doc['reduction'],ensure_ascii=False,indent=2)+'\n```\n\n约化参数：\n\n```json\n'+json.dumps(doc['reduced'],indent=2)+'\n```\n\n'
        text+='## 📊 数值结果与可视化\n\n'+table(group)+'\n'
        text+=f'![{paper} 的轨迹及数值参数比较]({relative(dest,report)})\n\n_图 1：各条件均保留；曲线是实际保存帧。频谱概览仅展示 power_fraction≥1e-5 的点，该值为展示规则而非验收阈值；所有点均在 JSON。_\n\n'
        text+='## 🔍 统计、收敛与验收门\n\n单位自旋模长按 GUIDE 的 1e-10 检查，LLB 磁化长度不适用。仅保存了归一化后误差，缺少归一化前误差，不能凭模长通过来认证积分器。\n\n'
        text+='随机轨迹按整条保留，Bauer 每条件 3 个种子×32 条轨迹。生存 bootstrap 为1000次、固定种子20260910、整轨迹重采样、点态95%区间；操作阈值0.7、驻留时间5仅作诊断，同时报告0.6/0.7/0.8与1/5/10敏感性。零事件的退化区间不证明概率为零。没有排除异常轨迹。\n\n'
        text+='确定性比较以最细 dt 为数值参考，保存完整 RMS/max 差异；不是解析真值。随机时间帧相关，窗口均值差仅为描述，未做独立帧 t 检验。AFMR 拟合协方差不是独立重复CI，周期不足时不得强行解释为稳定频率。没有预注册的收敛容差，因此不新增 pass/fail 阈值，也不以不显著推断等价。\n\n'
        text+=f'本体系的确定性比较见 [comparisons.json]({relative(out/"comparisons.json",report)})。\n\n'
        text+='## ⚠️ 文献对照与未完成项\n\n'+notes+'\n\n没有完整的审定数字化参考曲线及误差，本批不能生成可信的四篇论文逐图数值误差。GUIDE 的标量目标及内部解析关系只作清楚标记的诊断对照；不伪造文献点或文献/模拟并排图。\n\n'
        text+='## 🔗 数据、代码和重现\n\n'
        for r in group:
            rd=ROOT/'data/literature_reproduction'/paper/'derived'/r['run_id']/aid
            text+=f'- task {r["index"]}：[{r["run_id"]}]({relative(rd,report)})，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`\n'
        text+=f'\n[分析清单与代码哈希]({relative(out/"manifest.json",report)})；全部原始文件哈希在逐 run 的 metrics.json 中重新核对。分析工具采用 Karpathy Guidelines 和 Scientific Agent Skills 的原始数据保留、相关性与可视化原则。[^1]\n\n[^1]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065\n'
        with (report/'report.md').open('x') as f:f.write(text)
        write(report/'report.json',dict(paper=paper,status='inconclusive',limitations=notes,tasks=group))
        for r in group:
            rdir=ROOT/'output/literature_reproduction'/paper/r['run_id']/aid
            with (rdir/'report.md').open('x') as f:
                f.write(f'# {paper} task {r["index"]}\n\n_数值诊断，不是文献认证。_\n\n---\n\n## 📊 结果\n\n'+table([r])+f'\n## 🔗 完整分析\n\n[体系报告]({relative(report/"report.md",rdir)})；[机器可读指标](report.json)。\n')
    write(out/'report_artifacts.json',dict(code_sha256=sha256(Path(__file__)),files={str(p.relative_to(ROOT)):sha256(p)
        for folder in (ROOT/'output/literature_reproduction',ROOT/'assets/literature_reproduction')
        for p in folder.rglob('*') if p.is_file() and aid in p.parts and p.name!='report_artifacts.json'}))
    print('Reports and comparisons complete',flush=True)


if __name__=='__main__':main()
