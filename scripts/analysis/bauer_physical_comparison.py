"""Post-hoc physical diagnostics of existing L25 paths; no training or certification."""
import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/research/runs/bauer_first_loop/20260913_v1'
BASE = ROOT/'assets/research/bauer_L25_first_loop/20260913_v1'
FIG = BASE/'model_comparison'
OUT = BASE/'analysis/physical_diagnostics'
TIME = np.arange(201)*80.
NAMES = ['negative_endpoint', 'crossed_zero', 'committed_switch', 'crossing_return', 'unresolved_transition']
COLORS = {'R1': '#222222', 'R2': '#888888', 'RFM11': '#0072B2', 'RFM22': '#D55E00', 'RFM33': '#009E73'}


def events(mz, threshold=.6, dwell=160.):
    """All paths start in + basin. Residence is confirmed only at saved frames."""
    flat = np.asarray(mz).reshape(-1, mz.shape[-1])
    labels = np.zeros((len(flat), 5), bool)
    first = np.full(len(flat), np.inf)
    for j, path in enumerate(flat):
        assert path[0] >= threshold
        crossed = np.flatnonzero(path < 0)
        labels[j, 0] = path[-1] < 0
        labels[j, 1] = len(crossed) > 0
        start = None
        for k, value in enumerate(path):
            if value <= -threshold:
                if start is None: start = k
                if TIME[k]-TIME[start] >= dwell:
                    first[j] = TIME[k]; break
            else:
                start = None
        labels[j, 2] = np.isfinite(first[j])
        labels[j, 3] = bool(len(crossed) and np.any(path[crossed[0]+1:] >= threshold))
        labels[j, 4] = abs(path[-1]) < threshold
    return labels.reshape(*mz.shape[:-1], 5), first.reshape(mz.shape[:-1])


def band(values, repetitions=600):
    """Initial-family then noise bootstrap; pointwise only, unreliable for rare events."""
    rng = np.random.default_rng(20260913)
    groups, noises = values.shape[:2]
    gi = rng.integers(groups, size=(repetitions, groups))
    ni = rng.integers(noises, size=(repetitions, groups, noises))
    draws = values[gi[..., None], ni].mean((1, 2))
    return np.quantile(draws, [.025, .975], axis=0)


def save(fig, name, title):
    fig.suptitle(title, fontsize=12)
    fig.savefig(FIG/(name+'.png'), dpi=120)
    plt.close(fig)


def load():
    datasets, hashes = {}, {}
    for theta in (.11, .13):
        source = DATA/f'reference_T{theta}.h5'
        with h5py.File(source, 'r') as h:
            assert h.attrs['complete'] and h.attrs['save_dt'] == 80
            ref = h['spins'][:].astype(np.float64)
        hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        group = {'R1': ref[:, :16], 'R2': ref[:, 16:]}
        for seed in (11, 22, 33):
            for steps in (32, 64):
                source = DATA/f'rfm_{seed}_T{theta}{"_steps64" if steps == 64 else ""}.npy'
                group[f'RFM{seed}'+('_64' if steps == 64 else '')] = np.load(source).astype(np.float64)
                hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        for paths in group.values():
            assert paths.shape == (4, 16, 201, 25, 3) and np.isfinite(paths).all()
            assert np.allclose(paths[:, :, 0], ref[:, :1, 0], atol=1e-6)
        datasets[theta] = group
    return datasets, hashes


def spatial(paths, frame):
    s = paths[:, :, frame]
    return np.stack([(s[..., :-r, :]*s[..., r:, :]).sum(-1).mean(-1) for r in range(1, 25)], -1)


def coarse_mechanism(paths, first):
    """First sustained 5-site negative patch: observable proxy, not certified nucleation."""
    records = []
    for j, path in enumerate(paths.reshape(-1, 201, 25, 3)):
        z = path[..., 2]
        block = np.stack([z[:, i:i+5].mean(-1) for i in range(21)], -1)
        mask = block < -.3
        stable = mask[:-2] & mask[1:-1] & mask[2:]
        found = np.argwhere(stable)
        row = dict(initial=j//16, path=j%16, patch_observed=False, patch_site=None,
                   patch_time=None, proxy_to_switch_time=None, simultaneous_first_blocks=0)
        if len(found):
            t = int(found[0, 0]); sites = np.flatnonzero(stable[t])
            # Average tied blocks to avoid preferentially choosing the left edge.
            row.update(patch_observed=True, patch_site=float(np.mean(sites+3)),
                       patch_time=float(TIME[t+2]), simultaneous_first_blocks=len(sites))
            switch = first.ravel()[j]
            if np.isfinite(switch) and switch >= TIME[t+2]:
                row['proxy_to_switch_time'] = float(switch-TIME[t+2])
        records.append(row)
    return records


def main():
    FIG.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    data, hashes = load()
    mz = {t: {n: p[..., 2].mean(-1) for n, p in g.items()} for t, g in data.items()}
    ev = {t: {n: events(m) for n, m in g.items()} for t, g in mz.items()}
    rows, results = [], {}
    for t, group in data.items():
        results[str(t)] = {}
        for name, paths in group.items():
            labels, first = ev[t][name]
            result = dict(probabilities=dict(zip(NAMES, labels.mean((0, 1)).tolist())),
                          probability_pointwise_ci95=band(labels).tolist(),
                          endpoint_variance=float(mz[t][name][..., -1].var()),
                          rmst=float(np.minimum(first, TIME[-1]).mean()),
                          spatial_correlation_final=spatial(paths, 200).mean((0, 1)).tolist(),
                          increment_variance={str(lag*80): float(np.var(mz[t][name][..., lag:]-mz[t][name][..., :-lag])) for lag in (1, 4, 16)})
            results[str(t)][name] = result
            for i in range(4):
                for j in range(16):
                    rows.append(dict(theta=t, model=name, initial=i, path=j,
                        **dict(zip(NAMES, labels[i,j].astype(int))),
                        first_switch_time=float(first[i,j]) if np.isfinite(first[i,j]) else '',
                        censored=int(not np.isfinite(first[i,j])), endpoint_mz=float(mz[t][name][i,j,-1])))
    with (OUT/'path_events.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    fig, ax = plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for row,t in enumerate(data):
        for i in range(4):
            a=ax[row,i]
            for name in ('R1','RFM11'):
                m=mz[t][name][i]
                a.plot(TIME,m.T,color=COLORS[name],lw=.4,alpha=.23)
                q=np.quantile(m,[.1,.5,.9],axis=0)
                a.fill_between(TIME,q[0],q[2],color=COLORS[name],alpha=.12)
                a.plot(TIME,q[1],color=COLORS[name],lw=1.5,label=name)
            a.set(title=f'theta={t}; initial {i}',ylim=(-1.05,1.05),xlabel='Reduced time',ylabel='Mz')
            a.legend(fontsize=8)
    save(fig,'conditional_path_ensembles','Matched initial states: all 16 paths; median and 10-90% range (not confidence intervals)')

    fig,ax=plt.subplots(2,4,figsize=(15,6),layout='constrained')
    for row,t in enumerate(data):
        for col,k in enumerate((25,50,100,200)):
            for name in COLORS:
                ax[row,col].hist(mz[t][name][...,k].ravel(),bins=np.linspace(-1,1,21),density=True,
                                histtype='step',color=COLORS[name],label=name,lw=1.2)
            ax[row,col].set(title=f'theta={t}; time={TIME[k]:.0f}',xlabel='Mz',ylabel='Density')
    ax[0,0].legend(fontsize=7)
    save(fig,'magnetization_distributions_over_time','Multi-time marginal distributions | pooled overview; conditional ensembles shown separately')

    fig,ax=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for row,t in enumerate(data):
        for i in range(4):
            for name in COLORS:
                first=ev[t][name][1][i]
                ax[row,i].step(TIME,(first[:,None]>TIME).mean(0),where='post',color=COLORS[name],label=name)
            ax[row,i].set(title=f'theta={t}; initial {i}',ylim=(-.03,1.03),xlabel='Reduced time',ylabel='P(no committed switch yet)')
    ax[0,0].legend(fontsize=7)
    save(fig,'switching_survival','Per-initial survival | 16 paths each | provisional basin +/-0.6, residence160; censored paths retained')
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for a,t in zip(ax,data):
        for name in COLORS:
            curves=(ev[t][name][1][...,None]>TIME).astype(float)
            lo,hi=band(curves)
            a.step(TIME,curves.mean((0,1)),where='post',label=name,color=COLORS[name])
            a.fill_between(TIME,lo,hi,step='post',color=COLORS[name],alpha=.1)
        a.set(title=f'theta={t}',ylim=(-.03,1.03),xlabel='Reduced time',ylabel='Survival');a.legend(fontsize=8)
    save(fig,'switching_survival_uncertainty','Pooled survival: exploratory pointwise 95% bootstrap bands; four initial families, not certification')

    fig,ax=plt.subplots(2,4,figsize=(16,7),layout='constrained')
    for row,t in enumerate(data):
        for i in range(4):
            for k,name in enumerate(COLORS):
                ax[row,i].plot(np.arange(5)+(k-2)*.035,ev[t][name][0][i].mean(0),'o-',ms=3,label=name,color=COLORS[name])
            ax[row,i].set(xticks=range(5),xticklabels=['end<0','cross0','commit','return','unresolved'],
                          ylim=(-.05,1.05),title=f'theta={t}; initial {i}',ylabel='Probability')
            ax[row,i].tick_params(axis='x',rotation=40)
    ax[0,0].legend(fontsize=7)
    save(fig,'transition_event_probabilities','Nonexclusive event labels | provisional basin0.6/residence160; return means positive-basin revisit after zero crossing')
    fig,ax=plt.subplots(2,5,figsize=(16,7),layout='constrained')
    for row,t in enumerate(data):
        for col,label in enumerate(NAMES):
            for k,name in enumerate(COLORS):
                values=ev[t][name][0][...,col];lo,hi=band(values);mean=values.mean()
                ax[row,col].plot([k,k],[lo,hi],color=COLORS[name],lw=2)
                ax[row,col].plot(k,mean,'o',color=COLORS[name],ms=4)
            ax[row,col].set(title=f'theta={t}\n{label}',xticks=range(5),xticklabels=list(COLORS),ylim=(-.03,1.03),ylabel='Probability')
            ax[row,col].tick_params(axis='x',rotation=50)
    save(fig,'transition_event_uncertainty','Pooled probabilities: pointwise95% initial/noise bootstrap; zero-event intervals may degenerate, NOT evidence of zero risk')

    fig,ax=plt.subplots(4,3,figsize=(13,12),layout='constrained')
    for ti,t in enumerate(data):
        for col,k in enumerate((50,100,200)):
            for name in ('R1','RFM11'):
                p=data[t][name][:,:,k];m=mz[t][name][...,k]
                energies=[-(p[...,:-1,:]*p[...,1:,:]).sum((-1,-2))/25,-.1*(p[...,2]**2).mean(-1)]
                for e,y in enumerate(energies):
                    ax[2*ti+e,col].scatter(m,y,s=12,alpha=.5,color=COLORS[name],label=name)
                    ax[2*ti+e,col].set(xlim=(-1,1),ylim=((-1,0) if e==0 else (-.1,0)),xlabel='Mz',
                        ylabel=('Exchange energy / spin' if e==0 else 'Anisotropy energy / spin'),title=f'theta={t}; time={TIME[k]:.0f}')
    ax[0,0].legend()
    save(fig,'energy_magnetization_joint','Joint magnetization-energy structure | R1 vs fixed RFM seed11; all 64 paths')

    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,t in enumerate(data):
        for col,k in enumerate((50,100,200)):
            for name in COLORS:
                values=spatial(data[t][name],k);lo,hi=band(values)
                ax[row,col].plot(range(1,25),values.mean((0,1)),color=COLORS[name],label=name)
                ax[row,col].fill_between(range(1,25),lo,hi,color=COLORS[name],alpha=.08)
            ax[row,col].set(title=f'theta={t}; time={TIME[k]:.0f}',xlabel='Site separation r',ylabel='C(r,t)',ylim=(-.1,1.02))
    ax[0,0].legend(fontsize=7)
    save(fig,'spatial_correlations','Open-chain pair averages | exploratory pointwise 95% initial/noise bootstrap bands')
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for row,t in enumerate(data):
        for name in COLORS:
            p=data[t][name];bonds=(p[...,:-1,:]*p[...,1:,:]).sum(-1)
            for col,selection in enumerate(([0,1,22,23],list(range(2,22)))):
                ax[row,col].plot(TIME,bonds[...,selection].mean((0,1,3)),label=name,color=COLORS[name])
                ax[row,col].set(title=f'theta={t}; '+('four edge bonds' if col==0 else 'twenty interior bonds'),
                                xlabel='Reduced time',ylabel='Nearest-neighbor correlation',ylim=(.4,1.02))
    ax[0,0].legend(fontsize=8)
    save(fig,'edge_interior_correlations','Open-boundary diagnostics: edge vs interior bond alignment; no periodic wraparound')

    fig,ax=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for row,t in enumerate(data):
        for col,k in enumerate((1,4,16)):
            for name in COLORS:
                m=mz[t][name];delta=(m[...,k:]-m[...,:-k]).ravel()
                ax[row,col].hist(delta,bins=np.linspace(-2,2,61),density=True,histtype='step',color=COLORS[name],label=name)
            ax[row,col].set(title=f'theta={t}; lag={k*80}',xlabel='Mz increment',ylabel='Density',xlim=(-1.2,1.2),yscale='log')
        for name in COLORS:
            m=mz[t][name];center=m-m.mean(axis=1,keepdims=True)
            cov=(center[...,50,None]*center).mean((0,1))
            ax[row,3].plot(TIME,cov,color=COLORS[name],label=name)
        ax[row,3].set(title=f'theta={t}; anchor time4000',xlabel='Reduced time',ylabel='Within-initial covariance')
    ax[0,0].legend(fontsize=7)
    save(fig,'temporal_correlations_and_increments','Increment histograms pool time (diagnostic only); covariance removes per-initial time-dependent ensemble mean')

    mechanisms={};selections=[]
    fig,ax=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for row,t in enumerate(data):
        mechanisms[str(t)]={}
        for name in COLORS:
            rec=coarse_mechanism(data[t][name],ev[t][name][1]);mechanisms[str(t)][name]=rec
            sites=[r['patch_site'] for r in rec if r['patch_observed']]
            delay=[r['proxy_to_switch_time'] for r in rec if r['proxy_to_switch_time'] is not None]
            ax[row,0].hist(sites,bins=np.arange(1,27,2),histtype='step',color=COLORS[name],label=name)
            if delay:
                x=np.sort(delay);ax[row,1].step(x,np.arange(1,len(x)+1)/len(x),where='post',marker='.',color=COLORS[name],label=f'{name}: n={len(x)}')
            ax[row,2].bar(list(COLORS).index(name),len(sites)/64,color=COLORS[name])
        ax[row,0].set(title=f'theta={t}: first patch position',xlabel='Mean of tied block centers',ylabel='Count out of64');ax[row,0].legend(fontsize=7)
        ax[row,1].set(title='Delay: patch confirmation to switch',xlabel='Reduced time',ylabel='Conditional ECDF');ax[row,1].legend(fontsize=7)
        ax[row,2].set(title='Any persistent negative patch',xticks=range(5),xticklabels=list(COLORS),ylim=(0,1),ylabel='Fraction')
    save(fig,'nucleation_and_propagation','COARSE PROXY ONLY: 5-site mean z<-0.3 for3 saved frames; not certified nucleation or wall-tracking statistics')
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for row,t in enumerate(data):
        for col,name in enumerate(('R1','RFM11')):
            committed=np.flatnonzero(ev[t][name][0][0,:,2]);crossed=np.flatnonzero(ev[t][name][0][0,:,1])
            j=int(committed[0] if len(committed) else crossed[0] if len(crossed) else 0)
            reason='first committed' if len(committed) else 'first zero-crossing' if len(crossed) else 'fallback path0'
            im=ax[row,col].pcolormesh(TIME,np.arange(1,26),data[t][name][0,j,:,:,2].T,vmin=-1,vmax=1,cmap='RdBu_r',shading='nearest')
            ax[row,col].set(title=f'{name}, theta={t}, initial0, path{j}\n{reason}',xlabel='Reduced time',ylabel='Site')
            selections.append(dict(theta=t,model=name,initial=0,path=j,rule=reason))
    fig.colorbar(im,ax=ax.ravel().tolist(),label='s_z')
    save(fig,'matched_mechanism_examples','Same selection rule, NOT matched noise or necessarily matched mechanism; examples cannot estimate frequencies')

    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,t in enumerate(data):
        for seed in (11,22,33):
            names=[f'RFM{seed}',f'RFM{seed}_64']
            vals=[results[str(t)][n] for n in names]
            ax[row,0].plot([32,64],[v['probabilities']['committed_switch'] for v in vals],'o-',label=f'seed{seed}')
            ax[row,1].plot([32,64],[v['endpoint_variance'] for v in vals],'o-')
            ax[row,2].plot([32,64],[v['rmst'] for v in vals],'o-')
        for col,key in enumerate(('probabilities','endpoint_variance','rmst')):
            ref=results[str(t)]['R1'][key]
            ax[row,col].axhline(ref['committed_switch'] if col==0 else ref,color='black',ls='--',label='R1')
            ax[row,col].set(xlabel='Flow integration steps',xticks=[32,64],title=f'theta={t}: '+['committed fraction','endpoint variance','restricted waiting time'][col])
    ax[0,0].legend(fontsize=8)
    save(fig,'sampling_steps_and_seed_stability','Existing paired-latent32/64 samples only | 16/128 NOT AVAILABLE | no retraining')

    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    with np.load(DATA/'development.npz') as d:
        for row,t in enumerate(data):
            dev=d['development'][np.isclose(d['development_condition'][:,0],t),:,:,2].mean(-1)
            ax[row,0].hist(dev[:,-1],bins=np.linspace(-1,1,21),color='.5')
            for b in (-.6,.6):ax[row,0].axvline(b,color='red',ls='--')
            ax[row,0].set(title=f'theta={t}; development endpoints n={len(dev)}',xlabel='Mz',ylabel='Count')
            for name in COLORS:
                probs=[events(mz[t][name],threshold=b)[0][...,2].mean() for b in (.4,.5,.6,.7,.8)]
                ax[row,1].plot([.4,.5,.6,.7,.8],probs,'o-',color=COLORS[name],label=name)
            ax[row,1].set(title='Sensitivity only, not threshold selection',xlabel='Provisional basin threshold',ylabel='Committed fraction',ylim=(-.03,1.03))
    ax[0,1].legend(fontsize=8)
    save(fig,'basin_definition_sensitivity','Basin certification NOT established | independent development endpoints and post-hoc sensitivity, residence160 fixed')

    fig,ax=plt.subplots(figsize=(12,5),layout='constrained');ax.axis('off')
    table=[['Geometry / initial','Existing checks support numerical constraints','Not a dynamics certificate'],
           ['Endpoint mixture / switching','Large observed RFM-reference discrepancies','Exploratory; basin calibration pending'],
           ['Space/time correlations','Curves and raw differences supplied','No new formal tolerance test'],
           ['Nucleation / propagation','Coarse negative-patch proxy only','Resolution and mechanism validation pending'],
           ['Flow-step convergence','32/64 results available','16/128 missing; no full convergence claim'],
           ['Reference / inference','L25 dt0.02; 4 initial families','Reference convergence and statistical power limited']]
    tab=ax.table(cellText=table,colLabels=['Dimension','Evidence available','Boundary'],cellLoc='left',loc='center',colWidths=[.22,.4,.38])
    tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,2.8)
    save(fig,'validation_dashboard','Evidence dashboard: post-hoc diagnosis, NOT a new PASS/FAIL certificate')
    source=DATA/'development.npz';hashes[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    for name,checksum in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==checksum
    summary=dict(scope='existing_samples_posthoc_not_formal_certification',statistics=results,
        event_definition=dict(threshold=.6,dwell=160,save_dt=80,labels_nonexclusive=True,
                              return_definition='any saved positive-basin revisit after first zero crossing',basin_calibration='NOT_CERTIFIED'),
        bootstrap=dict(repetitions=600,seed=20260913,scope='pointwise initial-family/noise; four families; zero events may give degenerate intervals'),
        missing=['16/128 flow-step samples','certified basin thresholds','fine-time nucleation/wall tracking','L25 reference integration convergence'],
        selections=selections,source_sha256=hashes)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (OUT/'coarse_mechanism_paths.json').write_text(json.dumps(mechanisms,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,indent=2))
    print('Completed physical figures, per-path CSV, source checksums; no training or source changes.')


if __name__=='__main__':main()
