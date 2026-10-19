"""Self-contained PNG/HTML report; missing runs are never labelled successful."""
import base64
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import HERE, cfg, dump, observable_samples, statistics

LABELS = dict(mz_mean='Mean magnetization',mz_var='Magnetization variance',
    energy_mean='Mean H/(NJ)',energy_var='Variance H/(NJ)',temporal='Initial-state overlap',temporal_lag='Lagged spin dot correlation',
    spatial='Spatial dot correlation',increment_mean='Conditional increment mean',
    increment_cov='Conditional increment covariance')


def image(fig,name,caption,out):
    path=out/f'{name}.png'
    fig.savefig(path,dpi=180,bbox_inches='tight'); plt.close(fig)
    content=base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{content}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def curves_and_paths(path,out,name):
    with np.load(path) as z:
        generated,reference,t=z['generated'].astype('f8'),z['reference'].astype('f8'),z['time_ps']
    sections=[]
    fig,axes=plt.subplots(2,3,figsize=(13,6),constrained_layout=True)
    for row,(s,label) in enumerate(((reference,'LLG'),(generated,'Generated'))):
        for col in range(3):
            im=axes[row,col].imshow(s[col,:,:,2].T,origin='lower',aspect='auto',cmap='coolwarm',vmin=-1,vmax=1,extent=[t[0],t[-1],0,s.shape[2]])
            axes[row,col].set(title=f'{label}: branch {col+1}',xlabel='Time (ps)',ylabel='Site')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Spin z')
    sections.append(image(fig,name+'_paths','同父状态的前三条独立路径，完整 1 ps。上下不是同一噪声的配对真值，不能用逐像素误差判断随机路径是否正确。',out))
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    rng=np.random.default_rng(237)
    for s,label,color in ((reference,'LLG','#0072B2'),(generated,'Model','#D55E00')):
        v=observable_samples(s)
        for ax,key in zip(axes[0],('mz','energy','temporal')):
            a=v[key]; boot=np.array([a[rng.integers(len(a),size=len(a))].mean(0) for _ in range(400)])
            lo,hi=np.quantile(boot,[.025,.975],axis=0)
            ax.plot(t,a.mean(0),color=color,label=label); ax.fill_between(t,lo,hi,color=color,alpha=.2)
            ax.set(xlabel='Time (ps)',ylabel=key); ax.legend()
        axes[1,0].hist(v['energy'][:,-1],bins=np.linspace(-1.1,1.1,35),alpha=.55,color=color,label=label,density=True)
        axes[1,1].plot([1,2,4],v['spatial'][:,-1].mean(0),'o-',color=color,label=label)
        var=statistics(s)['increment_cov']
        axes[1,2].plot(t[1:],np.trace(var,axis1=-2,axis2=-1).mean(-1),color=color,label=label)
    axes[1,0].set(xlabel='Endpoint H/(NJ)',ylabel='Density'); axes[1,0].legend()
    axes[1,1].set(xlabel='Site separation',ylabel='Endpoint spin dot correlation'); axes[1,1].legend()
    axes[1,2].set(xlabel='Time (ps)',ylabel='Mean trace conditional increment covariance'); axes[1,2].legend()
    sections.append(image(fig,name+'_statistics','上排阴影是固定父状态均值的点态 95% bootstrap 区间，不是轨迹涨落带；下排比较终态能量、空间相关和增量涨落。所有重采样以完整分支为单位。',out))
    return ''.join(sections)


def main():
    config=cfg(); out=HERE/'assets'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    parts=['<h1>P0–P2：Bauer 局域图模型首轮闭环</h1>',
        '<p>范围：开放链易轴模型，L16/L25/L32，T=63.82485 K，1 ps / 0.1 ps。单温度、局域图网络、FM-only；不是完整多尺度或文献长时复现。</p>',
        '<p>模型：标量—向量图消息传递 → 逐格点通道归一化 → 块内时间注意力与 FiLM → 切向速度 → 球面流积分。实际初帧固定，晶体轴与磁矩联合 SO(3) 协变。温度标准化为常量零，不宣称学会温度依赖。</p>',
        '<p>数据：144 个训练父状态 × 8 分支；36 个开发父状态 × 16 分支；36 个历史诊断父状态 × 16 分支。另有开发集细步长参考。先按父状态划分再截窗；准备族均衡不代表自然占据概率；历史 test 不作为未触碰最终测试。</p>',
        '<p>选模：开发集条件增量协方差 RMSE 优先，能量均值 RMSE 只作同分排序。物理容差尚未确认，报告偏差及不确定性，不自动盖章通过。</p>']
    a=json.loads((HERE/'audit.json').read_text())
    parts.append(f'<h2>P0 审计</h2><p>文件、来源哈希与能量复算通过。最大模长偏差 {a["max_norm_error"]:.3g}；最大能量差 {a["max_stored_energy_error"]:.3g}。</p>')
    numerical=json.loads((HERE/'numerical_reference.json').read_text())['comparisons']
    numeric={k:float(np.mean([r['metrics'][k]['rmse'] for r in numerical])) for k in LABELS}
    parts.append('<p>已有 0.25 / 0.125 fs 独立参考对比。差异同时包含数值离散化与有限样本误差；区间包含零不证明等价，参考差异也不是通用物理阈值。</p>')
    summary=[]; completed=[]
    for seed in config['seeds']:
        run=HERE/'runs'/f'seed_{seed}'
        if not (run/'complete.json').exists():
            parts.append(f'<h2>Seed {seed}</h2><p>尚无完整报告（未开始／未完成／失败均可能，不推断成功）。</p>'); continue
        selection=json.loads((run/'selection.json').read_text())
        dev=json.loads((run/'dev_128.json').read_text()); diag=json.loads((run/'diagnostic_128.json').read_text())
        completed.append((seed,run,selection,dev,diag))
        summary.append(dict(seed=seed,selection=selection,dev=dev['mean_parent_RMSE'],diagnostic=diag['mean_parent_RMSE']))
        parts.append(f'<h2>Seed {seed}</h2><p>选中更新步 {selection["selected_step"]}；GPU {html.escape(selection["gpu"])}；训练含读取 {selection["train_seconds"]/60:.1f} 分钟。种子差异单独报告，不当作分支误差。</p>')
        fig,axes=plt.subplots(3,3,figsize=(13,9),constrained_layout=True)
        for ax,key in zip(axes.flat,LABELS):
            ax.bar(['Dev','Diagnostic','dt ref'],[dev['mean_parent_RMSE'][key],diag['mean_parent_RMSE'][key],numeric[key]],color=['#0072B2','#D55E00','#888888'],edgecolor='black',linewidth=.5)
            ax.set(title=LABELS[key],ylabel='Mean parent RMSE'); ax.tick_params(axis='x',rotation=20)
        parts.append(image(fig,f'seed_{seed}_metrics','逐项统计误差，越低越接近参考。灰色是开发集粗细步长间的有限样本差异，不是验收线，也不能直接用于诊断集显著性检验。',out))
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        loss=np.array([x['loss'] for x in json.loads((run/'history.json').read_text())]); width=min(100,len(loss))
        axes[0].plot(np.arange(width,len(loss)+1),np.convolve(loss,np.ones(width)/width,mode='valid'))
        axes[0].set(xlabel='Update',ylabel=f'FM loss ({width}-update average)')
        for key in ('increment_cov','temporal','energy_mean'):
            vals=[json.loads((run/f'dev_{s}.json').read_text())['mean_parent_RMSE'][key] for s in config['flow_steps']]
            axes[1].plot(config['flow_steps'],vals,'o-',label=key)
        axes[1].set(xlabel='Flow integration steps',ylabel='Dev mean-parent RMSE'); axes[1].legend()
        parts.append(image(fig,f'seed_{seed}_training_sampler','左：损失只是优化诊断。右：同 latent seed 比较 16/32/64/128 步；采样收敛不等于物理正确。',out))
    if completed:
        seed,run,selection,dev,diag=min(completed,key=lambda item:tuple(item[2]['criterion']))
        parts.append(f'<h2>展示模型 Seed {seed}（仅按开发集规则选取）</h2><p>每个尺寸使用 manifest 中第一个父状态，不挑选最好看的路径。</p>')
        manifest=json.loads((HERE/'manifest_dev.json').read_text())
        for size in config['sizes']:
            row=next(r for r in manifest if r['L']==size)
            parts.append(f'<h3>L{size}：{row["parent_id"]} / {row["family"]}</h3>')
            parts.append(curves_and_paths(run/'generated'/'dev_128'/f'{row["parent_id"]}.npz',out,f'L{size}'))
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for ax,key in zip(axes.flat,('mz_mean','energy_mean','temporal','increment_cov')):
            for i,r in enumerate(dev['parents']):
                m=r['metrics'][key]; lo,hi=m['mean_difference_ci95']
                ax.plot([lo,hi],[i,i],color='#0072B2'); ax.plot(m['mean_difference'],i,'o',color='#D55E00',ms=3)
            ax.axvline(0,color='black',lw=.7); ax.set(title=LABELS[key],xlabel='Mean signed difference (model - LLG)',ylabel='Fixed parent index')
        parts.append(image(fig,'differences','固定父状态，完整独立分支重采样，包含参考和生成两侧误差。标量差值点态 95% CI，不是同时带。带符号平均可能抵消，须结合全张量 RMSE。',out))
        seconds=np.mean([r['generation_seconds'] for r in dev['parents']])/config['branches']
        parts.append(f'<p>批量采样吞吐折算每条 {seconds:.5f} 秒（batch=16），不是单条延迟或同精度 LLG 加速比。仅含采样、不含写盘，未扣除首次初始化。</p>')
    else:
        parts.append('<h2>模型结果待计算</h2><p>正式任务完成后依赖任务自动填充，不使用冒烟冒充训练结果。</p>')
    parts.append('<h2>结果边界</h2><p>不认证全部路径分布、罕见事件、温度／尺寸外推或长时接续。物理容差和新增独立最终留出尚待确认。checkpoint、全部生成路径和 JSON 保存于 tests/bauer_p2_20260923/runs/。</p>')
    parts.append('<p>流程与绘图工具归属：Kassis, T., Agarwal, V., He, Y., Patel, D., &amp; Brueckner, A. M. (2026). <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents</a>。仅作为 experimental-design / matplotlib 工作流来源，不作为模型性能证据。</p>')
    body='\n'.join(parts)
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Bauer P2</title><style>body{max-width:1250px;margin:30px auto;padding:20px;font:16px/1.7 sans-serif;color:#20313e}img{width:100%}figure{margin:20px 0}figcaption{color:#526575}h2{border-bottom:1px solid #ccc}p{overflow-wrap:anywhere}</style>'+body+'</html>'
    (out/'index.html').write_text(document)
    (out/'fragment.html').write_text('<section id="bauer-p2-20260923">'+body+'</section>')
    dump(out/'report.json',dict(completed_seeds=[x['seed'] for x in summary],runs=summary,
        status='ALL_COMPUTATIONS_REPORTED_NOT_PHYSICS_CERTIFIED' if len(summary)==3 else 'INCOMPLETE'))
    from tests.progress_dashboard.build import main as dashboard
    dashboard()
    print(str(out/'index.html'))


if __name__=='__main__':
    main()
