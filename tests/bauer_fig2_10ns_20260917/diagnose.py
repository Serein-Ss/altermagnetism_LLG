"""Exploratory events, wall tracks, paper raster assembly and bounded dt check."""
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .comparison import OUT, MEDIA, PAPER_END_NS, image_tag

D=OUT/'diagnostics_events'


def table(path,rows,fields):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def events(t,m,threshold,hold_ps):
    # Basin exit to first continuously occupied basin; all times refer to saved frames.
    labels=np.where(m>=threshold,1,np.where(m<=-threshold,-1,0))
    hold=int(round(hold_ps/((t[1]-t[0])*1000)))+1
    state=int(labels[0]); start=None; run=0; previous=0; rows=[]
    for i,label in enumerate(labels):
        run=run+1 if label==previous else 1; previous=label
        if start is None and label!=state: start=i
        if start is not None and label and run>=hold:
            segment=m[start:i+1]
            crossed=bool(np.any(segment*state<0)) if state else False
            kind='committed_switch' if state and label==-state else ('crossing_return' if crossed else 'basin_return')
            rows.append(dict(start_ns=float(t[start]),entry_ns=float(t[i-hold+1]),confirmation_ns=float(t[i]),
                             kind=kind,crossed_zero=crossed,from_sign=state,to_sign=int(label)))
            state=int(label); start=None
    if start is not None:
        rows.append(dict(start_ns=float(t[start]),entry_ns='',confirmation_ns=float(t[-1]),
                         kind='unresolved_transition' if labels[-1]==0 else 'unconfirmed_dwell',
                         crossed_zero=bool(np.any(m[start:]*state<0)),from_sign=state,to_sign=int(labels[-1])))
    return rows


def assemble_paper():
    source=np.asarray(Image.open(MEDIA/'paper_figure2.png').convert('RGB'))
    fig,ax=plt.subplots(2,1,figsize=(12,3.3),constrained_layout=True)
    # Inner plot rectangles in the unchanged 1280x670 source crop; no curve digitization.
    bands=[[(22,92),(232,302),(442,512)],[(127,197),(337,407),(547,617)]]
    for a,ys,ylim in zip(ax,bands,[(-100,100),(0,100)]):
        for j,(y0,y1) in enumerate(ys):
            a.imshow(source[y0:y1,98:1248],extent=(j*PAPER_END_NS/3,(j+1)*PAPER_END_NS/3,*ylim),aspect='auto')
        for x in [PAPER_END_NS/3,2*PAPER_END_NS/3]: a.axvline(x,color='grey',ls='--',lw=.7)
        a.set_xlim(0,10)
        a.axvspan(PAPER_END_NS,10,color='lightgrey')
    ax[0].set(ylabel='Total Mz',title='Paper Fig. 2: original raster panels concatenated (not raw data)')
    ax[1].set(ylabel='Chain site',xlabel='Time (ns), using J = 50 meV')
    fig.savefig(D/'paper_full.png',dpi=170); plt.close(fig)
    (D/'paper_crop.json').write_text(json.dumps({'source':str(MEDIA/'paper_figure2.png'),'x':[98,1248],
       'bands':bands,'seams_ns':[PAPER_END_NS/3,2*PAPER_END_NS/3],
       'note':'Raster assembly; original arrows retained where inside plot; grey tail has no paper data.'},indent=2))


def analyze():
    D.mkdir(exist_ok=True); assemble_paper()
    all_events=[]; sensitivity=[]; summaries=[]
    with h5py.File(OUT/'trajectory.h5') as h:
        t=h['time_ps'][:]/1000; ms=h['M_total'][:,:,2]/100
        pools=[]
        for b in range(3):
            z=uniform_filter1d(h['spins'][b,::10,:,2],5,axis=1,mode='nearest')
            coherent=np.all(z>0,axis=1)|np.all(z<0,axis=1)
            pools.extend(np.abs(ms[b,::10][coherent]).tolist())
        threshold=float(np.quantile(pools,.01))
        fig,ax=plt.subplots(figsize=(8,3),constrained_layout=True)
        ax.hist(ms.ravel(),bins=160,density=True); ax.axvline(threshold,color='red'); ax.axvline(-threshold,color='red')
        ax.set(xlabel='Mean mz',ylabel='Time occupancy density',title='Exploratory basin cut from coherent reference configurations')
        fig.savefig(D/'basins.png',dpi=170); plt.close(fig)
        for b in range(3):
            for q in [.005,.01,.025]:
                cut=float(np.quantile(pools,q))
                for hold in [5,20,50]:
                    ev=events(t,ms[b],cut,hold)
                    sensitivity.append(dict(path=b,quantile=q,threshold=cut,hold_ps=hold,
                         switches=sum(e['kind']=='committed_switch' for e in ev),returns=sum(e['kind']=='crossing_return' for e in ev)))
            ev=events(t,ms[b],threshold,20)
            all_events.extend(dict(path=b,**e) for e in ev)
            z=uniform_filter1d(h['spins'][b,::10,:,2],5,axis=1,mode='nearest'); ts=t[::10]
            rows=[]; previous=[]; ids=[]; next_id=0
            for ti,zz in zip(ts,z):
                indices=np.flatnonzero(zz[:-1]*zz[1:]<0)
                positions=indices-zz[indices]/(zz[indices+1]-zz[indices])
                current_ids=[-1]*len(positions)
                if len(previous) and len(positions):
                    cost=np.abs(np.asarray(previous)[:,None]-positions[None,:])
                    aa,bb=linear_sum_assignment(cost)
                    for a,c in zip(aa,bb):
                        # Ambiguous births/mergers reset identity instead of forcing a continuous path.
                        if cost[a,c]<=5 and (cost[a]<=5).sum()==1 and (cost[:,c]<=5).sum()==1:
                            current_ids[c]=ids[a]
                for j,x in enumerate(positions):
                    if current_ids[j]<0: current_ids[j]=next_id; next_id+=1
                    rows.append(dict(time_ns=float(ti),wall_id=current_ids[j],position=float(x)))
                previous=positions; ids=current_ids
            table(D/f'path_{b}_walls.csv',rows,['time_ns','wall_id','position'])
            fig,ax=plt.subplots(2,1,figsize=(12,5),constrained_layout=True)
            ax[0].plot(t,ms[b],lw=.5); ax[0].axhline(threshold,color='grey',ls='--'); ax[0].axhline(-threshold,color='grey',ls='--')
            for e in ev:
                if e['kind'] in ['committed_switch','crossing_return']:
                    ax[0].axvspan(e['start_ns'],e['confirmation_ns'],alpha=.18,color='red' if e['kind']=='committed_switch' else 'orange')
            ax[0].set(ylabel='Mean mz',title=f'Path {b}: red = switch candidate, orange = crossing-return')
            ax[1].imshow(z.T,origin='lower',aspect='auto',extent=(0,10,-.5,99.5),cmap='coolwarm',vmin=-1,vmax=1)
            if rows: ax[1].scatter([r['time_ns'] for r in rows],[r['position'] for r in rows],s=.15,c='black')
            ax[1].set(xlabel='Time (ns)',ylabel='Site; black = wall zero crossings')
            fig.savefig(D/f'path_{b}_events.png',dpi=170); plt.close(fig)
            summaries.append(dict(path=b,negative_endpoint=bool(ms[b,-1]<0),crossed_zero=bool(np.any(ms[b]<0)),
                switches=sum(e['kind']=='committed_switch' for e in ev),crossing_returns=sum(e['kind']=='crossing_return' for e in ev)))
    table(D/'events.csv',all_events,['path','start_ns','entry_ns','confirmation_ns','kind','crossed_zero','from_sign','to_sign'])
    table(D/'event_sensitivity.csv',sensitivity,['path','quantile','threshold','hold_ps','switches','returns'])
    (D/'summary.json').write_text(json.dumps({'basin_threshold':threshold,'hold_ps':20,'paths':summaries,
      'caveat':'Exploratory threshold from coherent time frames, not independently certified terminal basin distribution. Correlated frames are not independent samples.',
      'wall_method':'5-site spatial average, every 1 ps, linear zero crossing; <=5 sites matching only when unambiguous; IDs reset at ambiguities. Not certified nucleation or diffusion statistics.'},indent=2))
    update_page()


def update_page():
    page=OUT/'index.html'; s=page.read_text(); start='<!-- DIAGNOSTICS_START -->'; end='<!-- DIAGNOSTICS_END -->'
    if start in s: s=s[:s.index(start)]+s[s.index(end)+len(end):]
    s=s.replace('<h2 id="full">',start+'<h3>原文连续轨迹：三个原图绘图区按时间顺序拼接</h3>'+image_tag(D/'paper_full.png')+
       '<p>这是原图像素拼接，不是原始数值数据。虚线标出接缝，原图内箭头仍保留；右端灰区为原文未覆盖的约0.127 ns。不得从拼接像素计算精确寿命或畴壁速度。</p>'+end+'<h2 id="full">')
    a='<!-- EVENTS_START -->'; z='<!-- EVENTS_END -->'
    if a in s: s=s[:s.index(a)]+s[s.index(z)+len(z):]
    summary=json.loads((D/'summary.json').read_text())
    block=a+'<section><h2>现有数据：事件与畴壁诊断</h2><p>探索性判据，不是独立认证：用空间平滑后全链同号的参考构型估计磁盆幅度，取 |m_z| 的1%分位数作为边界；反向磁盆连续保持20 ps才记为完成翻转候选。另检查0.5/1/2.5%分位数与5/20/50 ps驻留的敏感性。时间占据分布不是独立终态分布。</p>'
    block+=image_tag(D/'basins.png')+'<pre>'+json.dumps(summary,ensure_ascii=False,indent=2)+'</pre>'
    with (D/'event_sensitivity.csv').open() as f: checks=list(csv.DictReader(f))
    block+='<details><summary>展开阈值与驻留时间敏感性：不可只看默认计数</summary><table><tr><th>轨迹</th><th>磁盆边界</th><th>保持 ps</th><th>翻转候选</th><th>过零返回</th></tr>'
    for r in checks:
        block+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in [r['path'],round(float(r['threshold']),4),r['hold_ps'],r['switches'],r['returns']])+'</tr>'
    block+='</table><p>严格连续驻留可能被热涨落打断；计数变化反映判据敏感性，不证明事件消失。当前不以此拟合寿命。</p></details>'
    block+='<p>红色时间段：完成翻转候选；橙色：越过零点后回到原磁盆。下图为5格空间平均的连续 s_z（蓝负红正，与上方二值绿蓝不同），黑点为插值得到的零交叉畴壁。每1 ps提取，热涨落也可形成短暂零交叉，不能自动认定成核；多壁匹配不明确时轨迹编号重置，不强行连接。</p>'
    block+=''.join(image_tag(D/f'path_{b}_events.png') for b in range(3))
    if (D/'dt_summary.json').exists():
        block+='<h3>短时步长敏感性结果</h3>'+image_tag(D/'dt_check.png')+'<p>误差棒为8个独立分支终点均值的标准误差；不是等效性或长时寿命认证。请同时检查JSON中的逐状态指标和原始分支数据。</p>'
    else: block+='<h3>短时步长敏感性：尚无正式结果</h3><p>0.5/0.25/0.125 fs；正磁盆、负磁盆、过渡构型各8个独立分支，每条100 ps；相同起点、不同未来噪声。只检查短时统计敏感性，不认证10 ns稀有事件收敛。</p>'
    block+='</section>'+z
    s=s.replace('<dialog id="zoom">',block+'<dialog id="zoom">'); page.write_text(s)


def dt_check(smoke=False):
    from .kernel import ensemble
    duration=.1 if smoke else 100.; count=2 if smoke else 8
    target=D/('dt_smoke' if smoke else 'dt'); target.mkdir(exist_ok=True)
    if (target/'summary.json').exists(): raise FileExistsError('refusing to overwrite completed dt check')
    with h5py.File(OUT/'trajectory.h5') as h:
        source=[(0,10000),(0,68000),(1,41500)]
        initial=np.array([h['spins'][b,k] for b,k in source],dtype=float)
    initial/=np.linalg.norm(initial,axis=-1,keepdims=True)
    initial=np.repeat(initial,count,axis=0); metrics=[]
    for j,dt_fs in enumerate([.5,.25,.125]):
        seeds=np.arange(3*count,dtype=np.int64)+202609180+j*1000
        dt_ps=dt_fs/1000; stride=round(.1/dt_ps)
        paths,err=ensemble(initial,seeds,dt_ps/(.6582119569/50),round(duration/dt_ps),stride,.1,.1,.11,0,True)
        assert np.isfinite(paths).all() and np.max(abs(np.linalg.norm(paths,axis=-1)-1))<1e-10
        np.savez_compressed(target/f'dt_{dt_fs}.npz',spins=paths,time_ps=np.arange(paths.shape[1])*.1,seeds=seeds,source=np.array(source),preprojection_error=err)
        for group in range(3):
            p=paths[group*count:(group+1)*count,-1]; m=p[:,:,2].mean(1)
            energy=-(p[:,:-1]*p[:,1:]).sum((1,2))/100-.1*(p[:,:,2]**2).mean(1)
            metrics.append(dict(dt_fs=dt_fs,group=group,n=count,mean_mz=float(m.mean()),se_mz=float(m.std(ddof=1)/np.sqrt(count)),
                                mean_energy=float(energy.mean()),se_energy=float(energy.std(ddof=1)/np.sqrt(count)),negative_endpoints=int((m<0).sum())))
    (target/'summary.json').write_text(json.dumps(metrics,indent=2))
    if not smoke:
        (D/'dt_summary.json').write_text(json.dumps(metrics,indent=2))
        fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for g in range(3):
            rs=[r for r in metrics if r['group']==g]
            for a,key,err in [(ax[0],'mean_mz','se_mz'),(ax[1],'mean_energy','se_energy')]:
                a.errorbar([r['dt_fs'] for r in rs],[r[key] for r in rs],yerr=[r[err] for r in rs],marker='o',label=['positive','negative','transition'][g]); a.set(xlabel='dt (fs)',ylabel=key); a.legend()
        fig.savefig(D/'dt_check.png',dpi=170); plt.close(fig); update_page()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dt',action='store_true'); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if a.dt: dt_check(a.smoke)
    else: analyze()
