"""PNG evidence supplement; numerical completion is never scientific certification."""
import base64
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tests.paper_work.completion.run import ROOT, load, energy_distance
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.committor_pilot_analysis import first_hit, wilson


def read(name):
    report = json.loads((ROOT/'results'/f'{name}.json').read_text())
    assert report['status'] == 'COMPLETE_NUMERICAL_ONLY'
    assert report['manifest_sha256'] == digest(ROOT/'manifest.json')
    assert digest(report['path']) == report['sha256']
    with np.load(report['path']) as z:
        return z['spins'].astype(float), z['reduced_time'], report


def table(name, rows):
    with (ROOT/'tables'/f'{name}.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def events(p, dt, a, b):
    mz = p[...,2].mean(-1); result = []
    for dwell in (0., .1, .5):
        hit = first_hit(mz <= a, round(dwell/dt)+1)
        committed = hit >= 0
        for label, mask in [('negative_endpoint', mz[:,-1]<0),
                            ('crossed_zero', (mz<0).any(1)),
                            ('reverse_core_residence', committed),
                            ('crossing_return', ((mz<0).cumsum(1)>0).__and__(mz>=b).any(1)),
                            ('unresolved_endpoint', (mz[:,-1]>a)&(mz[:,-1]<b))]:
            n = len(mask); k = int(mask.sum()); lo, hi = wilson(k,n)
            result.append(dict(event=label, dwell=dwell, count=k, n=n, probability=k/n, lo95=lo, hi95=hi))
    return result


def run():
    m = load(); c = m['config']; files = []; captions = {}
    def export(fig, name, caption):
        fig.savefig(ROOT/'figures'/f'{name}.png', dpi=180, facecolor='white')
        plt.close(fig); files.append(name); captions[name] = caption
    # Full paths remain available; descriptive summaries never flatten initial groups.
    metrics = []; event_rows = []; cores = {}; records = {}
    for i, task in enumerate(m['reference']):
        if task['arm'] != 'calibration':
            continue
        p,t,_ = read(f'reference_{i}'); r=m['states'][task['state']]
        tail=p[:,t>=150,:,2].mean(-1).ravel()
        positive=tail[tail>0]; negative=tail[tail<0]
        if len(positive)<2 or len(negative)<2:
            raise ValueError('Insufficient positive/negative reference core samples')
        cores[(r['sites'],r['theta'])]=(float(np.quantile(negative,.9)),float(np.quantile(positive,.1)))
    table('core_proposals',[dict(sites=k[0],theta=k[1],B_max=v[0],A_min=v[1]) for k,v in cores.items()])
    for i,task in enumerate(m['reference']):
        if task['arm'] != 'calibration': records[('reference',i)] = task
    for i,task in enumerate(m['generation']): records[('generation',i)] = task
    for i,task in enumerate(t for t in m['generation'] if t['arm']=='frozen'):
        records[('baseline',i)]=dict(task,arm='gaussian')
    for (prefix,i),task in records.items():
        p,t,report=read(f'{prefix}_{i}'); state=task['state']; r=m['states'][state]
        matches=[j for j,q in enumerate(m['reference']) if q['arm']=='mt' and q['state']==state and q['duration']>=task['duration']]
        j=min(matches,key=lambda j:m['reference'][j]['duration'])
        ref,rt,_=read(f'reference_{j}'); ref=ref[:,:len(t)]
        np.testing.assert_allclose(rt[:len(t)],t,atol=1e-10)
        for horizon in (.01,.08,.8,8.,32.,200.):
            if horizon>t[-1]+1e-9: continue
            frame=round(horizon/c['save_dt']); x=p[:,frame]; y=ref[:,frame]
            metrics.append(dict(arm=task['arm'],state=state,sites=r['sites'],theta=r['theta'],kind=r['kind'],
                duration=task['duration'],time=horizon,mean_rmse=float(np.sqrt(np.mean((x.mean(0)-y.mean(0))**2))),
                energy_distance=energy_distance(x,y),mz_w1=float(wasserstein_distance(x[...,2].mean(-1),y[...,2].mean(-1))),
                bond_error=float(abs((x[:,1:]*x[:,:-1]).sum(-1).mean()-(y[:,1:]*y[:,:-1]).sum(-1).mean())),
                variance_ratio=float(x.var(0).mean()/max(y.var(0).mean(),1e-20)),
                seconds=report['seconds_excluding_save'],n=len(p)))
        a,b=cores[(r['sites'],r['theta'])]
        for e in events(p,c['save_dt'],a,b):
            if task['arm']=='deterministic':
                e['lo95']=e['hi95']=e['probability']
            event_rows.append(dict(arm=task['arm'],state=state,sites=r['sites'],theta=r['theta'],kind=r['kind'],
                                   duration=task['duration'],**e))
    table('path_metrics',metrics); table('events',event_rows)
    training=[]; validation=[]; efficiency=[]
    for i,spec in enumerate(m['training']):
        report=json.loads((ROOT/'results'/f'train{i}.json').read_text())
        training.append(dict(index=i,**spec,pairs=report['pair_count'],paths=report['path_count'],
             selected_step=report['selected_step'],selected_score=report['selected_score'],seconds=report['seconds_training_and_validation']))
        validation.extend(dict(index=i,**spec,step=r['step'],score=r['score']) for r in report['validation'])
        for state in range(6):
            p,_,_=read(f'train{i}_test{state}'); j=next(j for j,t in enumerate(m['reference']) if t['arm']=='mt' and t['state']==state)
            ref,_,_=read(f'reference_{j}')
            efficiency.append(dict(index=i,**spec,state=state,score=energy_distance(p[:,-1],ref[:,8])))
    table('training',training);table('validation',validation);table('data_efficiency',efficiency)
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for kind,style in [('flow','-o'),('deterministic','--s')]:
        for i in [r['index'] for r in training if r['kind']==kind and r['fraction']==1.]:
            v=[r for r in validation if r['index']==i]
            ax[0].plot([r['step'] for r in v],[r['score'] for r in v],style,label=f'{kind} seed {training[i]["seed"]}')
    for seed in c['seeds']:
        v=[r for r in training if r['kind']=='flow' and r['seed']==seed]
        ax[1].plot([r['paths'] for r in v],[np.mean([q['score'] for q in efficiency if q['index']==r['index']]) for r in v],'-o',label=f'seed {seed}')
    ax[0].set(xlabel='Training updates',ylabel='Development endpoint energy distance');ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Whole training trajectories',ylabel='Fresh-state endpoint energy distance at 0.08');ax[1].legend()
    export(fig,'Fig01_selection_and_data','Left: fixed development criterion, smaller is better. Right: all three seeds, same update budget; descriptive average over six fixed initial states, not a power certificate.')
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for arm,style in [('mt','-'),('frozen','--'),('selected','-.'),('deterministic',':'),('gaussian','--')]:
        key=next(k for k,q in records.items() if q['arm']==arm and q['state']==1 and q['duration']==32.)
        p,t,_=read(f'{key[0]}_{key[1]}');mz=p[...,2].mean(-1)
        ax[0].plot(t,mz.mean(0),style,label=arm)
        ax[0].fill_between(t,np.quantile(mz,.1,axis=0),np.quantile(mz,.9,axis=0),alpha=.08)
        ax[1].hist(mz[:,-1],bins=np.linspace(-1,1,41),density=True,histtype='step',linestyle=style,label=arm)
    ax[0].set(xlabel='Reduced time',ylabel='Mz: mean and 10–90% path range');ax[1].set(xlabel='Endpoint Mz',ylabel='Density')
    ax[0].legend();ax[1].legend()
    export(fig,'Fig02_matched_baselines','Prespecified tilted initial state 1, theta=0.11, L=25. Deterministic has no conditional randomness; bands are path spread, not confidence intervals.')
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for axis,length in zip(ax,c['sites']):
        for arm,style in [('heun','-o'),('frozen','--s'),('selected','-.^'),('deterministic',':D')]:
            means=[]
            for h in (.01,.08,.8,8.,32.):
                vals=[r['mz_w1'] for r in metrics if r['arm']==arm and r['sites']==length and r['time']==h and r['duration']==32.]
                means.append(np.mean(vals))
            axis.plot([.01,.08,.8,8.,32.],means,style,label=arm)
        axis.set(xscale='log',xlabel='Reduced time',ylabel='Mean conditional Mz W1',title=f'L={length}');axis.legend(fontsize=7)
    export(fig,'Fig03_size_time','Same parameter values, three independently specified initial states per temperature and size. Error is averaged over six fixed initial conditions. Heun–MT is a numerical/reference discrepancy, not exact truth error.')
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for axis,event in zip(ax,['reverse_core_residence','unresolved_endpoint']):
        for offset,arm in enumerate(['mt','heun','frozen','selected','deterministic']):
            rows=[r for r in event_rows if r['arm']==arm and r['duration']==200. and r['dwell']==.1 and r['event']==event]
            xx=np.arange(len(rows))+.12*(offset-2);yy=np.array([r['probability'] for r in rows])
            axis.errorbar(xx,yy,yerr=[yy-np.array([r['lo95'] for r in rows]),np.array([r['hi95'] for r in rows])-yy],fmt='o',label=arm,capsize=2)
        axis.set(xticks=np.arange(4),xticklabels=['25/.11','25/.13','100/.11','100/.13'],xlabel='L/theta; wall starts',ylabel='Probability',title=event,ylim=(-.03,1.03));axis.legend(fontsize=7)
    export(fig,'Fig04_extended_events','T=200, 64 stochastic branches per wall initial, conditional 95% Wilson intervals (core-estimation uncertainty not included). Deterministic repeated outputs have no sampling interval. Cores calibrated independently per L/theta. Core residence is not a completed reversal from an equilibrated positive basin. Zero events remain zero with an upper bound.')
    fig,ax=plt.subplots(2,1,figsize=(11,6),layout='constrained')
    for axis,arm in zip(ax,['mt','selected']):
        key=next(k for k,q in records.items() if q['arm']==arm and q['state']==14 and q['duration']==200.)
        p,t,_=read(f'{key[0]}_{key[1]}')
        im=axis.imshow(p[0,:,:,2].T,origin='lower',aspect='auto',extent=[0,t[-1],0,100],vmin=-1,vmax=1,cmap='RdBu_r',interpolation='nearest')
        axis.set(xlabel='Reduced time',ylabel='Site',title=f'{arm}: L100, theta .11, branch 0 (not event-selected)')
        fig.colorbar(im,ax=axis,label='sz')
    export(fig,'Fig05_large_chain_mechanism','Direct L100 generation and matched reference, fixed branch index. T=200 diagnostic is NOT the literature window T=750000 and cannot establish its lifetime law.')
    bench=json.loads((ROOT/'results/benchmark.json').read_text());table('same_gpu_timing',bench['rows'])
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for arm,style in [('flow','-o'),('mt','--s')]:
        for n in (1,64,128):
            vals=[np.median([r['seconds'] for r in bench['rows'] if r['arm']==arm and r['sites']==L and r['paths']==n])/n for L in c['sites']]
            ax[0].plot(c['sites'],vals,style,label=f'{arm} batch {n}')
    ax[0].set(xlabel='Sites',ylabel='Seconds/path for duration 0.08',yscale='log');ax[0].legend(fontsize=7)
    rows=[r for r in bench['rows'] if r['arm']=='flow' and r['paths']==64]
    projected=[np.median([r['seconds'] for r in rows if r['sites']==L])*750000/.08/86400 for L in c['sites']]
    ax[1].plot(c['sites'],projected,'-o');ax[1].set(xlabel='Sites',ylabel='Projected GPU-days / 64 paths',title='Linear projection to T=750000; NOT measured')
    export(fig,'Fig06_same_gpu_cost','Same allocation; first repeat excluded, median of three measured repeats. MT float64 vs flow float32, so speed is not equal-accuracy acceleration. Long-time projection excludes extra storage and interruptions.')
    html=['<!doctype html><meta charset="utf-8"><title>Paper evidence completion</title><h1>补全实验：待科学审核</h1>',
          '<p>下列图仅在所有依赖实验成功后生成。代码运行成功不代表路径统计通过。完整文献时长尚未执行。</p>']
    for name in files:
        encoded=base64.b64encode((ROOT/'figures'/f'{name}.png').read_bytes()).decode()
        html.append(f'<h2>{name}</h2><p>{captions[name]}</p><img style="max-width:100%" alt="{captions[name]}" src="data:image/png;base64,{encoded}">')
    (ROOT/'index.html').write_text('\n'.join(html))
    save(ROOT/'results/analysis.json',dict(status='COMPLETE_DESCRIPTIVE_ANALYSIS_NOT_CERTIFIED',figures=files,captions=captions,
         manifest_sha256=digest(ROOT/'manifest.json'),remaining=['Full literature duration 750000 and recurrent lifetime statistics',
         'Independent scientific review of plots and reference convergence', 'Precision/power confirmation beyond budget-limited 64 paths per initial']))


if __name__=='__main__':
    run()
