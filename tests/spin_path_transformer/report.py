"""Self-contained test dashboard; after-any failures remain explicit."""
import argparse
import html
import json
import re
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tests.bauer_p2_20260923.common import dump, comparison, observable_samples
from tests.bauer_comprehensive_20260923.run import diagnostic_comparison
from tests.bauer_comprehensive_20260923.report import finite_sample
from tests.bauer_repair_20260924.run import mixture_samples, mixture_comparison
from tests.cross_dimension_20260924.analysis import observables, compare
from tests.cross_dimension_20260924.report import image_tag, animate, plot_series
from tests.literature_preflight_20260917.models import build
from tests.literature_path_campaign_20260920.train_miltat import physical_config
from .run import HERE, ROOT, tasks


def simple_plot(g,r,domain,path):
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    if domain=='toy':
        gs,rs=mixture_samples(g),mixture_samples(r)
        for ax,key in zip(axes.flat,('signed_rotation','direction_coherence','rotation_amplitude','direction_persistence')):
            low=min(gs[key].min(),rs[key].min());high=max(gs[key].max(),rs[key].max())
            edges=np.linspace(low-.01,high+.01,21)
            for value,label,color in [(rs[key],'Analytic reference','#0072B2'),(gs[key],'Model','#D55E00')]:
                ax.hist(value,bins=edges,density=True,alpha=.5,label=label,color=color)
            ax.set_title(key);ax.set_ylabel('Density')
    else:
        gs,rs=observable_samples(g),observable_samples(r)
        for ax,key in zip(axes.flat,('mz','energy','temporal')):
            for val,label,color in [(rs[key],'LLG','#0072B2'),(gs[key],'Model','#D55E00')]:
                mean=val.mean(0);sd=val.std(0,ddof=1);t=np.arange(11)*.1
                ax.plot(t,mean,label=label,color=color);ax.fill_between(t,mean-sd,mean+sd,color=color,alpha=.2)
            ax.set_title(key+(' (H/NJ)' if key=='energy' else ''));ax.set_xlabel('Time (ps)')
        for val,label,color in [(r,'LLG','#0072B2'),(g,'Model','#D55E00')]:
            variance=np.diff(val,axis=1).var(0,ddof=1).sum(-1).mean(-1)
            axes.flat[3].plot(np.arange(1,11)*.1,variance,label=label,color=color)
        axes.flat[3].set_title('Conditional increment variance');axes.flat[3].set_xlabel('Interval end (ps)')
    for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('First fixed development parent; shaded band = branch SD, not uncertainty')
    fig.savefig(path,dpi=170);plt.close(fig)


def analyze(index,task,out,assets):
    rows=[];images=[]
    with h5py.File(out/'paths.h5') as h:
        if not h.attrs['complete']:raise RuntimeError('incomplete HDF5')
        plotted=False
        for key in h:
            group=h[key];g=group['generated'][:];r=group['reference'][:]
            if task['domain']=='miltat':
                cell,_,_=build(physical_config(),'miltat',shape=r.shape[2:4])
                result=compare(observables(g,cell),observables(r,cell),924+index)
                result.update(split=group.attrs['split'],scope=group.attrs['scope'])
            elif task['domain']=='bauer':
                result=dict(physical=comparison(g,r,100,924+index),
                    diagnostic=diagnostic_comparison(g,r,100,925+index),finite_sample=finite_sample(g,r,926+index))
            else:result=dict(mixture=mixture_comparison(g,r,100))
            result.update(group=key,generated_branches=len(g),reference_branches=len(r),
                norm_error=float(abs(np.linalg.norm(g,axis=-1)-1).max()),
                anchor_error=float(abs(g[:,0]-r[:1,0]).max()))
            rows.append(result)
            if key.endswith('steps128') and not plotted:
                png=assets/f'run{index:02d}.png'
                if task['domain']=='miltat':
                    plot_series(result,png);gif=assets/f'run{index:02d}.gif';animate(group,gif);images.append(gif.name)
                else:simple_plot(g,r,task['domain'],png)
                images.insert(0,png.name);plotted=True
    return dict(parents=rows,images=images)


def main():
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);assets=HERE/'assets';assets.mkdir(exist_ok=True)
    literature=json.loads((HERE/'literature.json').read_text())
    body=['<h1>磁性路径 Transformer：文献指导的修复验证</h1>',
        '<p>这是开发实验，不是物理认证。三个训练种子分别报告，不能用种子平均掩盖模式偏置。</p>',
        '<h2>修复什么</h2><p>旧网络在解析双模式路径中不能保持整段共同模式，增加采样步数未解决；二维也有增量方差和时间相关偏差。'
        '本轮检验表示与信息交互，而不是宣称体系本身有问题。</p>',
        '<h2>实际架构</h2><p>带方向的局域图更新 → 每格点双向时间注意力 → 可选全局 latent 读写或区域细粗交互 → MLP。'
        '四个残差块、宽度64、逐格点 LayerNorm、adaLN-zero 条件门控。输出切向速度并在球面上采样；首帧固定为实际输入初态。</p>',
        '<p>标量 token 来自磁矩相对于登记晶体坐标系的分量，输出重建为向量；保证磁矩与结构一起旋转的 SO(3) 协变，'
        '不是磁矩单独旋转或反射对称性。网络不输入 J、D、K 或真实有效场。当前单温度条件归一化为0，不支持温度外推。</p>',
        '<p>local 为同期局域时空对照；latent 为8个学习查询读写；hierarchical 在此基础上用4格点区域均值和粗图通信，保留细特征及磁矩抵消幅度。'
        '只有一个细粗层级，不是完整递归多尺度算法。dense 仅用于小型合成问题。</p>',
        '<h2>证据与适用范围</h2><p>沿用现有独立父状态划分和所有物理参数。Bauer 1 ps/0.1 ps、开放链L16/25/32；二维开放64×38×2、12.5 K。'
        '三维目前只有软件约束测试。二维更大尺寸仅作探索，不是已经认证的尺寸推广。合成数据无ps单位，运输止于0.95，对照同一位置的解析分布。</p>',
        '<p>主要比较最终固定预算checkpoint。每父状态独立整条轨迹bootstrap100次；区间是逐点估计误差，不是物理涨落带或等价性证明。'
        '二维参考每父状态仅8条，统计分辨率有限。训练seed、数值步数差异单独检查。暂不新增事件通过阈值。</p>',
        '<h2>参考文献与迁移边界</h2>']
    for item in literature['sources']:
        body.append('<p><a href="'+html.escape(item['url'])+'">'+html.escape(item['title'])+'</a> ('+item['venue']+')：'+html.escape(item['adaptation'])+'</p>')
    body.append('<p>开发遵循 Karpathy 的隔离修改、实验设计的配对种子与同期对照、Matplotlib 的可读绘图规范。</p>')
    complete=[];missing=[];errors=[]
    for index,task in enumerate(tasks()):
        out=HERE/f'runs/{index:02d}';file=out/'complete.json'
        if not file.exists():missing.append(dict(index=index,task=task));continue
        info=json.loads(file.read_text());body.append('<h2>'+html.escape(str(dict(index=index,**task)))+'</h2>')
        try:
            cache=out/'diagnostics.json'
            if cache.exists():result=json.loads(cache.read_text())
            else:result=analyze(index,task,out,assets);dump(cache,result)
            complete.append(dict(index=index,run=info,diagnostics=result))
            body.append('<p>训练秒数 '+str(round(info['train_seconds'],1))+'；GPU '+html.escape(info['device'])+'。这不是通过精度门槛的加速比。</p>')
            for image in result['images']:body.append(image_tag(assets/image))
            body.append('<details><summary>所有父状态、采样步数及完整指标</summary><pre>'+html.escape(json.dumps(result['parents'],ensure_ascii=False,indent=2))+'</pre></details>')
        except Exception as exc:
            errors.append(dict(index=index,error=repr(exc)));body.append('<p>评估错误：'+html.escape(repr(exc))+'</p>')
    body.insert(1,f'<p>可分析完成 {len(complete)}/36；未完成 {len(missing)}；分析错误 {len(errors)}。'+('预览页面，尚无正式训练结论。' if args.preview else '')+'</p>')
    body.append('<details><summary>未完成及错误</summary><pre>'+html.escape(json.dumps(dict(missing=missing,errors=errors),ensure_ascii=False,indent=2))+'</pre></details>')
    section='\n'.join(body)
    document='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Spin path Transformer</title><body style="max-width:1280px;margin:30px auto;font:17px/1.6 sans-serif">'+section+'</body></html>'
    tmp=assets/'index.html.tmp';tmp.write_text(document);tmp.replace(assets/'index.html')
    dump(assets/'summary.json',dict(complete=complete,missing=missing,errors=errors,scientific_certification=False))
    dashboard=ROOT/'tests/index.html'
    if dashboard.exists():
        text=dashboard.read_text();start='<!-- spin-path-transformer:start -->';end='<!-- spin-path-transformer:end -->'
        block=start+'<section id="spin-path-transformer">'+section+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>')
        tmp=dashboard.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(dashboard)
    if errors:raise RuntimeError('report contains analysis errors; see summary.json')


if __name__=='__main__':main()
