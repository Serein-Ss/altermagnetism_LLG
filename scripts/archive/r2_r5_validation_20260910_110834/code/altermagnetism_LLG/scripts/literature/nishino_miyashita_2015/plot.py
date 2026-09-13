"""Plot only derived/reference data; no trajectory generation or fitting here."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import numpy as np


def plot(report,reference,destination):
    destination=Path(destination);figures=destination/'figures';animations=destination/'animations'
    figures.mkdir(parents=True,exist_ok=True);animations.mkdir(parents=True,exist_ok=True)
    factor=report['selected_factor'] or report['numerics'].get('selected_factor',1)
    rows=[r for r in report['rows'] if r['method']=='paper_midpoint' and r['factor']==factor]
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    colors={'A':'#0072B2','B':'#D55E00'}
    theta=np.linspace(.1,6.,400);x=report['reduced']['field_h']/theta
    axes[0].plot(theta,1/np.tanh(x)-1/x,'k-',label='Exact Langevin')
    for case,marker in [('A','x'),('B','s')]:
        selected=sorted([r for r in rows if r['case']==case],key=lambda r:r['theta'])
        axes[0].errorbar([r['theta'] for r in selected],[r['mean_check']['mean'] for r in selected],
            yerr=[r['mean_check']['ci95_halfwidth'] for r in selected],fmt=marker,
            capsize=3,color=colors[case],label=f'Case {case}, independent-moment 95% CI')
    axes[0].set(xlabel=r'Reduced temperature $\theta$',ylabel=r'$\langle s_z\rangle$',
                title=f"{report['phase']} | {report['status']} | dt={.005/factor:g}",ylim=(0,1))
    axes[0].legend(fontsize=8)
    ref=Path(reference)/'fig1_published_page3.png'
    if ref.exists(): axes[1].imshow(plt.imread(ref))
    else: axes[1].text(.5,.5,'Reference extraction unavailable',ha='center')
    axes[1].axis('off');axes[1].set_title('Published Fig. 1; visual reference, not digitized points')
    fig.savefig(figures/'figure1_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for case in ('A','B'):
        r=next(r for r in rows if r['case']==case and r['theta_index']==3)
        axes[0,0].plot(r['tau'],r['mean_trace'],color=colors[case],label=f"Case {case}, theta={r['theta']}")
        axes[0,1].plot(r['tau'],r['mean_within_replica_std_z'],color=colors[case],label=case)
        histogram=r['endpoint_histogram'];edges=np.asarray(histogram['edges']);counts=np.asarray(histogram['counts'])
        axes[1,0].stairs(counts/(counts.sum()*np.diff(edges)),edges,color=colors[case],label=case)
        selected=[q for q in rows if q['case']==case]
        axes[1,1].errorbar([q['theta'] for q in selected],
            [q['mean_check']['mean']-q['mean_check']['target'] for q in selected],
            yerr=[q['mean_check']['ci95_halfwidth'] for q in selected],fmt='o',color=colors[case],label=case)
    axes[0,0].set(xlabel=r'$\tau$',ylabel=r'Mean $s_z$',title='Relaxation; full observation interval')
    axes[0,1].set(xlabel=r'$\tau$',ylabel=r'Within-replica std($s_z$)',title='Independent moments; no spatial interaction')
    axes[1,0].set(xlabel=r'Final $s_z$',ylabel='Probability density',title='Terminal distribution')
    axes[1,1].axhspan(-.01,.01,color='gray',alpha=.15)
    axes[1,1].set(xlabel=r'$\theta$',ylabel='Mean minus exact',title='Entire CI must lie in tolerance band')
    for ax in axes.flat: ax.legend(fontsize=8)
    fig.savefig(figures/'equilibrium_diagnostics.png',dpi=180);plt.close(fig)
    example=next(r for r in rows if r['case']=='A' and r['theta_index']==3)
    t=np.asarray(example['example_tau']);z=np.asarray(example['example_z'])
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.set(xlim=(0,t[-1]),ylim=(-1,1),xlabel=r'$\tau$',ylabel=r'$s_z$',
           title='16 fixed, unscreened independent moments: case A, theta=2')
    lines=[ax.plot([],[],alpha=.6,lw=.8)[0] for _ in range(z.shape[1])]
    def frame(i):
        for k,line in enumerate(lines): line.set_data(t[:i+1],z[:i+1,k])
        return lines
    animation=FuncAnimation(fig,frame,frames=len(t),blit=True)
    animation.save(animations/'independent_moment_paths.gif',writer=PillowWriter(fps=12),dpi=90)
    plt.close(fig)
    return [str(p) for p in sorted(destination.rglob('*')) if p.is_file()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--derived',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(plot(json.loads(a.derived.read_text()),a.reference,a.output)))


if __name__=='__main__': main()
