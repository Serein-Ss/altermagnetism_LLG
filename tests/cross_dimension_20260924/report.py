"""Automatic after-any report; missing runs never count as success."""
import argparse
import base64
import io
import json
import re
import html
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tests.bauer_p2_20260923.common import dump
from tests.literature_preflight_20260917.models import build
from tests.literature_path_campaign_20260920.train_miltat import physical_config
from .run import HERE,ROOT,tasks
from .analysis import observables,compare


def image_tag(path):
    mime='image/gif' if path.suffix=='.gif' else 'image/png'
    return '<img style="max-width:100%" src="data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()+'">'


def animate(group,dest):
    # Fixed first branch, not selected for favorable agreement.
    raw=[group[k][0] for k in ('reference','generated')]
    pos=group['positions_nm'][:].reshape(-1,3);frames=[]
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    plots=[]
    for row in range(2):
        for col in range(3):
            ax=axes[row,col];sc=ax.scatter(pos[:,0],pos[:,1],c=raw[row][0,...,col].ravel(),
                s=7,marker='s',cmap='coolwarm',vmin=-1,vmax=1,linewidths=0)
            ax.set_aspect('equal');ax.set_title(('LLG','Model')[row]+' / '+('sx','sy','sz')[col])
            ax.set_xlabel('x (nm)');ax.set_ylabel('y (nm)');plots.append(sc)
    fig.colorbar(plots[0],ax=axes.ravel().tolist(),label='spin component [-1,1]',shrink=.7)
    for t in range(len(raw[0])):
        for row in range(2):
            for col in range(3):plots[row*3+col].set_array(raw[row][t,...,col].ravel())
        fig.suptitle(f'{t*.1:.1f} ps; independent futures, same initial state')
        stream=io.BytesIO();fig.savefig(stream,format='png',dpi=90);stream.seek(0)
        frames.append(Image.open(stream).convert('RGB').copy())
    frames[0].save(dest,save_all=True,append_images=frames[1:],duration=300,loop=0)
    plt.close(fig)


def plot_series(result,dest):
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,key in zip(axes.flat,('mz','energy','overlap','Q','increment_variance','adjacent_increment_dot')):
        s=result['series'][key];n=len(s['generated_mean'])
        t=np.arange(n)*.1+(0 if n==11 else .1 if n==10 else .2)
        for label,color in [('reference','#0072B2'),('generated','#D55E00')]:
            mean=np.asarray(s[label+'_mean']);ax.plot(t,mean,label=label,color=color)
            if label+'_sd' in s:
                sd=np.asarray(s[label+'_sd']);ax.fill_between(t,mean-sd,mean+sd,color=color,alpha=.18)
        ax.set_title(key+(' (H / NJ)' if key=='energy' else ''));ax.set_xlabel('physical time (ps)')
        ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Fixed parent: lines = means; shaded bands = branch SD, NOT confidence intervals')
    fig.savefig(dest,dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);assets=HERE/'assets';assets.mkdir(exist_ok=True)
    results=[];missing=[];body=['<h1>二维图模型与解析流诊断（2026-09-24）</h1>',
        '<p>本页区分运行成功、结构合法与物理统计正确。尚无自动物理通过阈值；不是文献长时扩散认证。</p>',
        '<h2>为什么暂时不能归因于一维体系</h2><p>上一轮一维模型首个0.1 ps增量方差约0.216，参考约0.143；'
        '增加flow步数未修复。合成双模式中单个训练seed严重偏置，跨seed平均接近50%会掩盖失败。'
        '因此体系难度可能参与，但模型/训练链路问题仍未排除。</p>',
        '<h2>本轮比较什么</h2><p>二维：64×38×2自旋，开放边界、12.5 K、阻尼0.5，'
        '完整空间1 ps / 0.1 ps网格；9次训练比较均匀/初态偏移潜变量和全局向量通信。'
        '不是把一维checkpoint直接用于二维。两种更大尺寸仅作探索性诊断。</p>',
        '<p>解析诊断：18次训练比较单/双整段模式、普通FM、全局FM和精确条件速度监督，三个seed分别显示。'
        '使用无物理单位合成时间；运输终点0.95，对照同一0.95处精确分布，不能与完整物理路径混淆。</p>',
        '<p>二维数据来自已登记的Miltat工程参考库；完整能量含交换、各向异性、DMI、开放偶极。'
        '原始文献来源和时间换算的未解决注释保留在data_audit.json；本轮不宣称重新认证原文全部物理。</p>']
    if args.preview:
        body.append('<h2>当前状态：新正式训练尚待完成</h2><p>软件测试已运行。下列正式结果缺失不是失败判定；此页在汇总任务结束后自动替换。</p>')
    for index,task in enumerate(tasks()):
        out=HERE/f'runs/{index:02d}';complete=out/'complete.json'
        if not complete.exists():missing.append(dict(index=index,task=task));continue
        info=json.loads(complete.read_text());section=f'任务 {index}: {task}'
        body.append('<h2>'+html.escape(section)+'</h2>')
        if task['domain']=='toy':
            results.append(info)
            body.append('<pre>'+html.escape(json.dumps(info['rows'],indent=2,ensure_ascii=False))+'</pre>');continue
        cache=out/'diagnostics.json'
        if cache.exists():diagnostics=json.loads(cache.read_text())
        else:
            diagnostics=[]
            with h5py.File(out/'paths.h5') as h:
                if not h.attrs['complete']:raise RuntimeError('incomplete path file')
                for key in h:
                    group=h[key];raw=group['reference'][:];generated=group['generated'][:]
                    cell,_,_=build(physical_config(),'miltat',shape=raw.shape[2:4])
                    a,b=observables(generated,cell),observables(raw,cell)
                    comp=compare(a,b,index+924);comp.update(group=key,split=group.attrs['split'],scope=group.attrs['scope'])
                    comp['metrics']['anchor_error']=float(abs(generated[:,0]-raw[0,0]).max())
                    # Descriptive 4+4 finite-reference floor, not an acceptance tolerance.
                    r1={k:v[:4] if isinstance(v,np.ndarray) else v for k,v in b.items()}
                    r2={k:v[4:] if isinstance(v,np.ndarray) else v for k,v in b.items()}
                    comp['reference_split_floor']=compare(r1,r2,index+928)['metrics']
                    diagnostics.append(comp)
            dump(cache,diagnostics)
        primary=[v for v in diagnostics if v['group']=='parent_000_steps128'][0]
        plot=assets/f'run{index:02d}.png';plot_series(primary,plot)
        body.append('<p>固定展示parent_000、第一条路径，未经挑选。阴影是分支标准差；置信区间另存JSON。增量方差比接近1、时间/空间误差较小只是诊断，须连同参考误差判断。</p>')
        body.append(image_tag(plot))
        gif=assets/f'run{index:02d}.gif'
        if not gif.exists():
            with h5py.File(out/'paths.h5') as h:animate(h['parent_000_steps128'],gif)
        body.append(image_tag(gif));body.append('<pre>'+html.escape(json.dumps(primary['metrics'],indent=2))+'</pre>')
        results.append(dict(task=task,diagnostics=diagnostics,total_seconds=info['total_seconds']))
    oracle=HERE/'oracle_reference/complete.json'
    if oracle.exists():
        info=json.loads(oracle.read_text());body.append('<h2>无需神经网络的精确流场积分</h2><p>支持域外调用必须接近零且步数收敛，'
            '才能作为可靠数值对照；否则只说明此解析积分尚未收敛。</p><pre>'+html.escape(json.dumps(info,indent=2))+'</pre>')
    else:body.append('<h2>精确流场积分：结果尚缺失</h2>')
    body.append('<h2>未完成或缺失任务</h2><pre>'+html.escape(json.dumps(missing,indent=2))+'</pre>')
    body.append('<h2>如何定位</h2><ul><li>精确流场也不收敛：先查数值积分，不能直接怪网络。</li>'
        '<li>精确流场成功但精确监督网络失败：表达/优化仍有问题。</li>'
        '<li>精确监督成功而普通FM失败：支持监督方差/训练耦合困难。</li>'
        '<li>二维图像合理但增量、能量、两时相关不对：仍不能认为学会动力学。</li>'
        '<li>二维好于一维不能单独证明维数因果，因为Hamiltonian、初态与温度也变了。</li></ul>')
    content='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Cross-dimensional diagnosis</title><body style="max-width:1250px;margin:auto;font-family:sans-serif"><main>'+''.join(body)+'</main></body></html>'
    (assets/'index.html').write_text(content)
    dump(assets/'summary.json',dict(results=results,missing=missing,scientific_certification=False))
    main=ROOT/'tests/index.html'
    if main.exists():
        text=main.read_text();start='<!-- cross-dimension-20260924-start -->';end='<!-- cross-dimension-20260924-end -->'
        text=re.sub(re.escape(start)+'.*?'+re.escape(end),'',text,flags=re.S)
        section=start+'<details open><summary>新增：二维图模型与精确流诊断</summary>'+''.join(body)+'</details>'+end
        text=re.sub(r'(<main\b[^>]*>)',lambda m:m.group(1)+section,text,count=1)
        main.write_text(text)
    print('report',assets/'index.html','complete',len(results),'missing',len(missing))


if __name__=='__main__':main()
