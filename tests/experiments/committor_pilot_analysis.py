"""Finite-horizon competing first arrivals; empirical cores are not basin certificates."""
import json
import os
import math
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest,save
from tests.experiments.committor_pilot import RAW,OUT,load


def wilson(k,n):
    if n<1 or not 0<=k<=n:raise ValueError('invalid binomial counts')
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,mid-radius),min(1.,mid+radius)]


def first_hit(mask,frames):
    if frames<1:raise ValueError('invalid residence interval')
    result=np.full(len(mask),-1,dtype=int);run=np.zeros(len(mask),dtype=int)
    for t in range(mask.shape[1]):
        run=np.where(mask[:,t],run+1,0)
        hit=(result<0)&(run>=frames);result[hit]=t
    return result


def classify(mz,times,a,b,dwell):
    if not a<0<b or not np.isfinite(mz).all():raise ValueError('invalid cores/data')
    dt=times[1]-times[0];np.testing.assert_allclose(np.diff(times),dt,atol=1e-10)
    frames=round(dwell/dt)+1
    if not np.isclose((frames-1)*dt,dwell):raise ValueError('unaligned dwell')
    ia=first_hit(mz>=b,frames);ib=first_hit(mz<=a,frames)
    ta=np.where(ia>=0,times[np.maximum(ia,0)],np.inf)
    tb=np.where(ib>=0,times[np.maximum(ib,0)],np.inf)
    labels=np.where(tb<ta,-1,np.where(ta<tb,1,0))
    tau=np.minimum(ta,tb)
    return labels,tau


def probabilities(labels,tau,horizon):
    n=len(labels);b=(labels==-1)&(tau<=horizon);a=(labels==1)&(tau<=horizon)
    nb=int(b.sum());na=int(a.sum());nu=n-nb-na
    return dict(n=n,A=na,B=nb,unresolved=nu,p_B=nb/n,p_A=na/n,p_unresolved=nu/n,
        p_B_ci95=wilson(nb,n),p_A_ci95=wilson(na,n),unresolved_ci95=wilson(nu,n),
        eventual_core_probability_bounds=[nb/n,1-na/n],
        restricted_mean_first_arrival=float(np.minimum(tau,horizon).mean()))


def read(name):
    r=json.loads((OUT/(name+'.json')).read_text())
    assert r['status']=='COMPLETE_NUMERICAL_ONLY' and r['manifest_sha256']==digest(OUT/'manifest.json')
    assert digest(r['path'])==r['sha256']
    with np.load(r['path']) as z:p=z['spins'];t=z['reduced_time']
    assert np.isfinite(p).all() and np.max(abs(np.linalg.norm(p,axis=-1)-1))<1e-5
    return p,t,r


def core_proposals(mz,quantiles):
    pos=mz[mz>0];neg=mz[mz<0]
    if len(pos)<2 or len(neg)<2:return []
    return [dict(quantile=q,A_min=float(np.quantile(pos,q)),B_max=float(np.quantile(neg,1-q))) for q in quantiles]


def run():
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('CPU Slurm required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m,cfg=load(); figdir=OUT/'figures';figdir.mkdir(exist_ok=False)
    core={};calibration=[]
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ti,theta in enumerate(cfg['theta']):
        for ci in range(2):
            p,t,r=read(f'calibration_{ti*2+ci}');mz=p[:,:,:,2].mean(-1)
            tail=mz[:,t>=cfg['calibration']['tail_start']]
            proposals=core_proposals(tail,cfg['calibration']['quantiles'])
            if ci==1:core[theta]=proposals
            # Per-path late-window absolute-magnetization differences; not independent frames.
            delta=np.abs(mz[:,(t>=150)&(t<200)]).mean(1)-np.abs(mz[:,(t>=100)&(t<150)]).mean(1)
            rng=np.random.default_rng(2140000000+ti*2+ci)
            draws=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1)
            calibration.append(dict(theta=theta,case=ci,source_sha256=r['sha256'],cores=proposals,
                late_window_abs_m_change=float(delta.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
                scope='Path-resampled descriptive interval; no stationarity or metastability acceptance threshold.'))
            ax[ti,0].hist(tail.ravel(),bins=np.linspace(-1,1,51),density=True,histtype='step',label=f'MT dt={cfg["calibration"]["cases"][ci]["dt"]}')
            ax[ti,1].plot(t,np.abs(mz).mean(0),label=f'dt={cfg["calibration"]["cases"][ci]["dt"]}')
        ax[ti,0].set(xlabel='Chain Mz, late-window frames',ylabel='Descriptive density',title=f'theta={theta}; correlated frames')
        ax[ti,1].set(xlabel='Reduced time',ylabel='Mean |Mz|',ylim=(0,1));ax[ti,0].legend();ax[ti,1].legend()
    fig.savefig(figdir/'basin_reference.png',dpi=150);plt.close(fig)
    save(OUT/'basin_audit.json',dict(status='EMPIRICAL_CORE_PROPOSALS_NOT_CERTIFIED',rows=calibration,
        dwell='Prespecified 0,0.1,0.5 sensitivity only; not calibrated commitment time',
        scope='Positive/negative conditional tail quantiles do not establish bimodality, stationarity or metastability.'))
    if any(not core[t] for t in cfg['theta']):
        save(OUT/'analysis.json',dict(status='HOLD_NO_TWO_SIDED_CORE_PROPOSALS',reason='Inspect calibration; do not invent a threshold.'))
        return
    arms=['reference0','reference1','baseline','flow11','flow22','flow33'];rows=[];priorities=[]
    c=cfg['shooting']
    for ai,state in enumerate(m['rows']):
        reference,rt,_=read(f'reference0_state{ai}');refmean=reference.mean(0)
        fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        primary={}
        for arm in arms:
            p,t,r=read(f'{arm}_state{ai}')
            assert p.shape==reference.shape
            np.testing.assert_allclose(t,rt,atol=1e-12)
            np.testing.assert_allclose(p[:,0],reference[:,0],atol=2e-7,rtol=0)
            mz=p[:,:,:,2].mean(-1);spin_z=p[:,:,:,2]
            mean_error=np.sqrt(((p.mean(0)-refmean)**2).mean((1,2)))
            crossings=(spin_z[:,:,:-1]*spin_z[:,:,1:]<0).sum(-1)
            strongest=np.argmax(abs(np.diff(spin_z,axis=-1)),axis=-1)+.5
            arrays=dict(reduced_time=t,mz=mz,spatial_std_z=spin_z.std(-1),
                zero_crossing_bond_count=crossings,strongest_gradient_bond=strongest,
                mean_spin=p.mean(0),mean_rmse=mean_error)
            for proposal in core[state['theta']]:
                q=proposal['quantile'];a=proposal['B_max'];b=proposal['A_min']
                for dwell in c['dwell_sensitivity']:
                    labels,tau=classify(mz,t,a,b,dwell)
                    arrays[f'q{q}_dwell{dwell}_labels']=labels
                    arrays[f'q{q}_dwell{dwell}_confirmation_time']=tau
                    stats={str(h):probabilities(labels,tau,h) for h in c['horizons']}
                    rows.append(dict(arm=arm,state=ai,theta=state['theta'],kind=state['kind'],quantile=q,
                        A_min=b,B_max=a,dwell=dwell,statistics=stats,source_sha256=r['sha256'],
                        starts_inside_core=bool(mz[0,0]>=b or mz[0,0]<=a)))
                    if q==cfg['calibration']['primary_quantile'] and dwell==0:
                        primary[arm]=stats[str(c['duration'])]
                        bcurve=np.mean((labels[:,None]==-1)&(tau[:,None]<=t[None]),axis=0)
                        ucurve=np.mean(tau[:,None]>t[None],axis=0)
                        arrays['B_cumulative_incidence']=bcurve;arrays['unresolved_survival']=ucurve
                        axs[0,0].plot(t,bcurve,label=arm);axs[0,1].plot(t,ucurve,label=arm)
            axs[1,0].plot(t,mean_error,label=arm)
            axs[1,1].plot(t,crossings.mean(0),label=arm)
            np.savez_compressed(OUT/f'observables_{arm}_state{ai}.npz',**arrays)
        axs[0,0].set(xlabel='Reduced time',ylabel='P(first arrival in B by t)',ylim=(0,1))
        axs[0,1].set(xlabel='Reduced time',ylabel='Unresolved probability',ylim=(0,1))
        axs[1,0].set(xlabel='Reduced time',ylabel='Mean-spin RMSE vs MT')
        axs[1,1].set(xlabel='Reduced time',ylabel='Mean sign-change bonds (proxy)')
        axs[0,0].legend(fontsize=8);fig.suptitle(f'State {ai}, {state["kind"]}, theta={state["theta"]}; exploratory cores')
        fig.savefig(figdir/f'state_{ai}.png',dpi=150);plt.close(fig)
        score=max(abs(primary[arm]['p_B']-primary['reference0']['p_B']) for arm in ['flow11','flow22','flow33'])
        priorities.append(dict(state=ai,score=score,unresolved_reference=primary['reference0']['p_unresolved'],
            source=state,reference_branches=str(RAW/f'reference0_state{ai}.npz'),
            purpose='Candidate for future reference-only augmentation; no automatic training or probability reweighting.'))
    fig,ax=plt.subplots(figsize=(6,6),layout='constrained');ax.plot([0,1],[0,1],'k--')
    for arm in arms[1:]:
        selected=[r for r in rows if r['arm']==arm and r['quantile']==.1 and r['dwell']==0]
        x=[];y=[];xe=[];ye=[]
        for r in selected:
            ref=next(s for s in rows if s['arm']=='reference0' and s['state']==r['state'] and s['quantile']==.1 and s['dwell']==0)
            a=ref['statistics'][str(c['duration'])];b=r['statistics'][str(c['duration'])]
            x.append(a['p_B']);y.append(b['p_B']);xe.append([a['p_B']-a['p_B_ci95'][0],a['p_B_ci95'][1]-a['p_B']]);ye.append([b['p_B']-b['p_B_ci95'][0],b['p_B_ci95'][1]-b['p_B']])
        ax.errorbar(x,y,xerr=np.maximum(0,np.asarray(xe).T),yerr=np.maximum(0,np.asarray(ye).T),fmt='o',alpha=.6,label=arm)
    ax.set(xlabel='MT P(B first by horizon)',ylabel='Candidate P(B first by horizon)',xlim=(-.03,1.03),ylim=(-.03,1.03),title='Finite-horizon cores, not infinite-time committor\nPointwise Wilson 95% intervals');ax.legend(fontsize=8)
    fig.savefig(figdir/'arrival_probability_calibration.png',dpi=150);plt.close(fig)
    save(OUT/'augmentation_priority.json',dict(status='PROPOSAL_ONLY_NO_RETRAINING',rows=sorted(priorities,key=lambda r:-r['score']),
        scope='Selected states are exploratory, not representative occupancy samples. High unresolved rate requires longer reference observation, not forced labels.'))
    save(OUT/'analysis.json',dict(status='COMPLETE_EXPLORATORY_NOT_CERTIFIED',rows=rows,manifest_sha256=digest(OUT/'manifest.json'),
        caveats=['Finite horizon probabilities, not infinite-time committor','Empirical cores and dwell sensitivity are not certified magnetic basins',
        'Eventual bounds apply to the specified core-hitting rule and ignore sampling uncertainty; separate Wilson intervals are pointwise',
        'No lifetime estimate from successful branches alone','Sign-change and largest-gradient quantities are spatial proxies, not certified nucleation/domain-wall classifiers',
        'No pooled independence across model seeds or source families','Calibration uncertainty is not propagated into branch intervals',
        'No generic equilibrium reweighting transplanted into non-equilibrium path probabilities']))


if __name__=='__main__':run()
