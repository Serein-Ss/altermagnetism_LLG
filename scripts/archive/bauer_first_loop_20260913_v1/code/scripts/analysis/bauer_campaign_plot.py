"""Unselected trajectories and uncertainty-aware campaign figures."""
from pathlib import Path
import json
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(root, pilot_only=False):
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    p=root/'reference/original_dt002/observables.h5'
    if p.exists() and not pilot_only:
        with h5py.File(p) as h:
            case=json.loads(h.attrs['case_json']);mz=h['values'][:5,:,3]
        t=np.arange(mz.shape[1])*case['save_dt']
        fig,axes=plt.subplots(len(mz),1,figsize=(10,1.6*len(mz)),sharex=True,constrained_layout=True,squeeze=False)
        axes=axes[:,0]
        for i,ax in enumerate(axes):
            ax.plot(t,mz[i],lw=.5);ax.set_ylim(-1.05,1.05);ax.set_ylabel(f'Path {i}\nMz/N')
        axes[0].set_title('Bauer original condition: first five paths, without event selection')
        axes[-1].set_xlabel('Reduced time (hbar/J)')
        fig.savefig(out/'original_trajectories.png',dpi=200);fig.savefig(out/'original_trajectories.pdf');plt.close(fig)
    p=root/'pilot_result.json'
    if p.exists():
        data=json.loads(p.read_text());fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        for row in data['records']:
            w=row['windows'];ax.errorbar([a['horizon'] for a in w],[a['reversal_probability'] for a in w],
                yerr=[a['nominal_halfwidth95'] for a in w],marker='o',capsize=3,label=f"theta={row['theta']}")
        ax.set(xlabel='Window (hbar/J)',ylabel='Completed first-passage probability',ylim=(0,1),
               title='Pilot only; bars are nominal binomial precision, not a power certificate')
        ax.legend();fig.savefig(out/'pilot_events.png',dpi=200);fig.savefig(out/'pilot_events.pdf');plt.close(fig)
    if pilot_only:
        return
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True);has=False
    for length in (25,50,100):
        rows=[]
        for p in sorted((root/'reference').glob(f'life_L{length}_*/analysis.json')):
            r=json.loads(p.read_text())
            if r['recurrent_interval_mean'] and r['recurrent_interval_ci95']:rows.append(r)
        if rows:
            has=True;x=[1/r['case']['theta'] for r in rows];y=np.array([r['recurrent_interval_mean'] for r in rows])
            ci=np.array([r['recurrent_interval_ci95'] for r in rows])
            axes[0].errorbar(x,y,yerr=np.maximum(0,np.stack((y-ci[:,0],ci[:,1]-y))),marker='o',capsize=3,label=f'L={length}')
            axes[1].plot(x,[r['completed_events'] for r in rows],marker='o',label=f'L={length}')
    if has:
        axes[0].set_yscale('log');axes[0].set(xlabel='1/theta',ylabel='Completed recurrent interval (hbar/J)',title='Extension temperature grid; finite-window statistic')
        axes[1].set(xlabel='1/theta',ylabel='Completed reversals',title='Event count; no adaptive stopping');axes[1].axhline(500,color='k',ls='--',label='Registered minimum')
        for ax in axes:ax.legend()
        fig.savefig(out/'arrhenius_events.png',dpi=200);fig.savefig(out/'arrhenius_events.pdf')
    plt.close(fig)
