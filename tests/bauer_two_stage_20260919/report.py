"""Self-contained diagnostic report; absent outputs are missing, never a pass."""
import base64
import html
import json
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import HERE,ROOT,cfg,dump


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def figure(parts,name,caption):
    path=HERE/'assets'/name
    parts.append('<figure><img style="width:100%" alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')


def table(parts,headers,rows):
    parts.append('<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>')
    parts.extend('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows);parts.append('</table>')


def main():
    c=cfg();out=HERE/'assets';out.mkdir(exist_ok=True)
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Bauer 两阶段实验</title><style>body{max-width:1200px;margin:30px auto;padding:20px;font:16px/1.7 sans-serif;color:#182b3a}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}pre{white-space:pre-wrap}figure{margin:20px 0}</style><h1>Bauer：初态生成、条件路径与尺寸推广</h1><p><a href="../../index.html">返回测试主网页</a></p><p>状态来自本目录实际输出；非实时队列监控。单温开放链、项目指定正向饱和后20 ps制备。初态与路径噪声独立，未向网络输入解析有效场。没有结果的阶段不记为成功。</p>']
    submission=read(HERE/'submission.json');assessment=read(HERE/'assessment.json');long=read(HERE/'long_status.json')
    parts.append('<h2>任务与数据状态</h2>')
    if submission:parts.append('<pre>'+html.escape(json.dumps(submission,ensure_ascii=False,indent=2))+'</pre>')
    table(parts,['L','初态数据完成','测试参考时长'],[(n,'是' if (HERE/'data'/f'L{n}'/'complete.json').exists() else '尚无完成记录','100 ps') for n in c['train_sizes']+c['transfer_sizes']])
    parts.append('<h2>参考校准与训练稳定性</h2>')
    cal=read(HERE/'calibration.json')
    if cal:
        table(parts,['指标','真实16对16误差','原工程线'],[(k,round(v['mean'],5),v['margin']) for k,v in cal['statistics'].items()])
        parts.append('<p>原工程线未修改。另报告模型误差减去匹配样本量参考误差的配对差值；这不是分布等价证明。</p>')
    runs=[]
    for k in ['initial','uniform','anchored']:
        for s in c['seeds']:
            r=read(HERE/'runs'/f'{k}_{s}'/'complete.json')
            runs.append((f'{k}_{s}',r['dev_score'] if r else '缺失',r['seconds'] if r else '缺失'))
    table(parts,['运行','开发分数（初态与路径不可互比）','训练和开发评估秒数'],runs)
    selection=read(HERE/'selection.json')
    if selection:parts.append('<p>仅开发集选定：'+html.escape(str(selection['chosen']))+'。最终测试及目标尺寸未参与选择。</p>')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    if assessment:
        parts.append('<h2>四组闭环与尺寸/时间推广</h2><p>R/R是真实初态＋LLG；R/G是真实初态＋模型；G/R是生成初态＋LLG；G/G是生成初态＋模型。每尺寸6个独立父构型，每组16条后继；1/10/100 ps为同条路径前缀，不是独立实验。</p>')
        table(parts,['条件','先导筛查'],[(k,'在线内（非认证）' if v['pass_screen'] else '未通过/证据不足') for k,v in assessment['gates'].items()])
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        a=assessment['initial'];ax.bar([str(r['L']) for r in a],[r['max_w1'] for r in a],color='#0072B2',edgecolor='black')
        ax.axhline(c['initial_margin'],color='black',ls='--',label='Frozen pilot margin');ax.set(xlabel='Chain length L',ylabel='Maximum descriptor W1',ylim=(0,None));ax.legend()
        fig.savefig(out/'initial.png',dpi=300);plt.close(fig)
        figure(parts,'initial.png','初态描述符最大W1距离；64个生成初态对64个独立真实初态，不显示虚构的置信区间。横线仅为预声明先导线。')
        for n in [25,100]:
            with h5py.File(HERE/'generated'/f'L{n}.h5') as h,h5py.File(HERE/'data'/f'L{n}'/'test_future_0.h5') as ref:
                fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
                for ax,s,title in zip(axes,[ref['spins'][0],h['RG_0'][0],h['GG_0'][0]],['R/R','R/G','G/G']):
                    im=ax.imshow(s[:,:,2].T,origin='lower',aspect='auto',extent=(0,100,0,n),vmin=-1,vmax=1,cmap='RdBu_r',interpolation='nearest')
                    ax.set(xlabel='Time (ps)',ylabel='Site',title=f'L{n} {title}')
                fig.colorbar(im,ax=axes,label='Spin z');fig.savefig(out/f'paths_L{n}.png',dpi=300);plt.close(fig)
            figure(parts,f'paths_L{n}.png','预先指定父状态0、分支0的完整100 ps路径，不按视觉效果挑选。R/G与R/R初态相同但未来噪声不同；G/G初态不同。仅展示，不代表概率认证。')
    else:parts.append('<p>四组检验尚无完整评估结果，不判断尺寸推广成功。</p>')
    parts.append('<h2>速度：先看同CPU、再看GPU实际吞吐</h2>')
    speeds=[]
    for n in c['train_sizes']+c['transfer_sizes']:
        r=read(HERE/'generated'/f'cpu_L{n}.json')
        if r:
            samples=r['same_cpu_one_thread_1ps'];speeds.append((n,float(np.median([v['llg_seconds'] for v in samples])),float(np.median([v['model_seconds'] for v in samples]))))
    if speeds:
        table(parts,['L','LLG秒/条/1ps','模型秒/条/1ps','原始时间比LLG/模型'],[(n,a,b,a/b) for n,a,b in speeds])
        initial_speed=[]
        for n,_,_ in speeds:
            values=read(HERE/'generated'/f'cpu_L{n}.json')['same_cpu_one_thread_1ps']
            initial_speed.append((n,float(np.median([r['llg_preparation_seconds'] for r in values])),float(np.median([r['model_initial_seconds'] for r in values])),float(np.median([r['end_to_end_raw_ratio'] for r in values]))))
        table(parts,['L','LLG初态20ps准备秒数','模型初态采样秒数','初态＋1ps未来原始时间比'],initial_speed)
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained');x=np.arange(len(speeds))
        ax.bar(x-.2,[r[1] for r in speeds],.4,color='#0072B2',edgecolor='black',label='LLG CPU')
        ax.bar(x+.2,[r[2] for r in speeds],.4,color='#E69F00',edgecolor='black',hatch='//',label='Model CPU')
        ax.set(xticks=x,xticklabels=[str(r[0]) for r in speeds],xlabel='L',ylabel='Seconds / path / 1 ps');ax.legend()
        fig.savefig(out/'cpu_speed.png',dpi=300);plt.close(fig);figure(parts,'cpu_speed.png','同节点、单CPU线程、同初态、同1 ps时长与0.1 ps输出网格，预热后3次计时的中位数。无磁盘写入；完整原始计时见JSON。未匹配精度前不称有效加速。')
    gpu=read(HERE/'generated'/'runtime.json')
    if gpu:
        table(parts,['GPU阶段','L','批次数量','总秒数'],[(r['stage'],r['L'],r.get('count',r.get('branches')),round(r['seconds'],4)) for r in gpu['measurements']])
    parts.append('<p>GPU模型与CPU模拟是不同硬件的实用吞吐比较，不是纯算法加速。初态生成单独计时；训练和开发耗时在上表，数据生成耗时在data/L*/complete.json。若目标统计不通过，不能报告“同精度加速比”或摊销盈亏平衡。</p>')
    parts.append('<h2>文献10 ns条件结果</h2><p>J=50 meV，K/J=0.1，λ=0.1，kBT/J=0.11，零外场，开放链L100。步长、准备协议和输出网格为项目选择。<a href="https://arxiv.org/html/1010.4730v2#S2.F2">原文图2与方法</a>。单温三条轨迹不能复现整篇论文的Arrhenius规律或精确寿命。</p>')
    parts.append('<pre>'+html.escape(json.dumps(long,ensure_ascii=False,indent=2) if long else '尚无10 ns阶段结果')+'</pre>')
    if long and long['status']=='GENERATED_DIAGNOSTIC_NOT_LITERATURE_CERTIFIED':
        from tests.bauer_fig2_10ns_20260917.diagnose import events
        baseline=ROOT/'tests/bauer_fig2_10ns_20260917/results'
        threshold=read(baseline/'diagnostics_events/summary.json')['basin_threshold']
        event_rows=[]
        with h5py.File(HERE/'literature_model.h5') as g,h5py.File(baseline/'trajectory.h5') as r:
            for j in range(c['long_paths']):
                fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
                for col,(tag,h) in enumerate([('LLG',r),('Model',g)]):
                    t=h['time_ps'][:]/1000;s=h['spins'][j];m=s[:,:,2].mean(-1)
                    # Display minutely subsampled time map, full-resolution curves and event calculation.
                    im=axes[0,col].imshow(s[::50,:,2].T,origin='lower',aspect='auto',extent=(0,10,0,100),vmin=-1,vmax=1,cmap='RdBu_r',interpolation='nearest')
                    axes[0,col].set(title=f'{tag} path {j}',xlabel='Time (ns)',ylabel='Site')
                    axes[1,col].plot(t,m,lw=.5);axes[1,col].set(xlabel='Time (ns)',ylabel='Mean spin z',ylim=(-1,1))
                    event_rows.append(dict(source=tag,path=j,events=events(t,m,threshold,20),negative_endpoint=bool(m[-1]<0),crossed_zero=bool(np.any(m<0))))
                fig.colorbar(im,ax=axes[0],label='Spin z');fig.savefig(out/f'literature_{j}.png',dpi=300);plt.close(fig)
                figure(parts,f'literature_{j}.png','左为现有LLG参考，右为模型，不是相同噪声配对。时空图每5 ps显示一帧；曲线与事件按完整0.1 ps数据计算。三条路径全部保留，不剪切、不按翻转筛选。')
        dump(out/'literature_events.json',dict(threshold=threshold,hold_ps=20,paths=event_rows,warning='Exploratory inherited threshold from reference coherent frames, not certified terminal-basin or lifetime result'))
    parts.append('<h2>可追溯文件</h2>')
    for p in [HERE/'config.yaml',HERE/'calibration.json',HERE/'selection.json',HERE/'assessment.json',HERE/'long_status.json']:
        if p.exists():parts.append('<p><a href="../'+p.name+'">'+p.name+'</a></p>')
    parts.append('</html>');(out/'index.html').write_text('\n'.join(parts))
    dump(out/'status.json',dict(assessment_available=assessment is not None,long_status=long,scientific_certification=False))


if __name__=='__main__':main()
