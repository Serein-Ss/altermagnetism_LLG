"""Explicitly exploratory extensions using already extracted derived arrays."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.analysis.analyze_remaining_validation import write,summary,windows
from scripts.analysis.report_remaining_validation import arrays
from scripts.literature.gomonay_2024.model import dispersion
from scripts.core.literature_config import load_runtime,sha256


def main():
    aid='r2_r5_analysis_20260910';out=ROOT/'output/literature_reproduction'/aid
    rows=json.loads((out/'metrics.json').read_text());extra=dict(bauer=[],helix=[],spectral=[],checks={})
    for dt in (.02,.01,.005):
        group=[r for r in rows if r['paper']=='bauer_2011' and r['parameters']['dt']==dt]
        endpoint=np.concatenate([arrays(r,aid)['mean'][-1,:,2] for r in group])
        rng=np.random.default_rng(20260910)
        means=endpoint[rng.integers(0,len(endpoint),(2000,len(endpoint)))].mean(1)
        extra['bauer'].append(dict(dt=dt,endpoint=summary(endpoint),endpoint_mean_pointwise_ci95=np.quantile(means,[.025,.975]),
            bootstrap_unit='independent chain, not time frame; 2000 draws, seed 20260910',
            observed_crossings=sum(r['crossed_zero_count'] for r in group),
            endpoint_by_seed=[r['endpoint']['mean'] for r in group],
            caution='Descriptive CIs, not simultaneous equivalence intervals. Same-seed dt runs are not independent paired Wiener paths.'))
    for r in rows:
        if r['protocol']=='helix':
            z=arrays(r,aid);s=z['final_spins'];phi=np.arctan2(-s[...,0],s[...,1]);dx=r['parameters']['dx']
            q=np.angle(np.exp(1j*(np.roll(phi,-1,axis=1)-phi)))/dx
            extra['helix'].append(dict(index=r['index'],local_q_sd=q.std(1),local_q_mean=q.mean(1),
                late_energy_windows=windows(z['time'],z['energy_per_spin']),
                final_max_abs_nz=r['final_max_abs_nz'],meaning='Angular increments; near-pole phase can be poorly defined; inspect nz too.'))
        if r['protocol'] in ('spinwave','control'):
            path=ROOT/'data/literature_reproduction'/r['paper']/'derived'/r['run_id']/aid/'spectrum.npz'
            with np.load(path,allow_pickle=False) as z:
                power=z['power'].sum(axis=(1,3));omega=z['omega'];kx=z['kx'];ky=z['ky']
                plus=np.flatnonzero(omega>0);minus=np.flatnonzero(omega<0)
                wp=omega[plus[np.argmax(power[plus],axis=0)]];wm=-omega[minus[np.argmax(power[minus],axis=0)]]
                fraction=power.sum(0)/power.sum()
                reduced=load_runtime(ROOT/'conf/literature/gomonay_2024.yaml').reduced
                if r['protocol']=='control':reduced=dict(reduced,J_tilde=0.)
                theoretical=np.sort(np.asarray(dispersion(kx,ky,reduced)),axis=0)
                error=np.max(abs(np.sort(np.stack((wp,wm)),axis=0)-theoretical),axis=0)
                # A reporting mask, not a pass criterion. Preserve all values.
                chosen=(fraction>=1e-5)&(np.hypot(kx,ky)>1e-8)
                extra['spectral'].append(dict(index=r['index'],total_path_points=len(kx),display_points=int(chosen.sum()),
                    all_kx=kx,all_ky=ky,all_power_fraction=fraction,all_error=error,all_display_mask=chosen,
                    displayed_error=summary(error[chosen]) if chosen.any() else None,
                    display_rule='power fraction >=1e-5 and nonzero k; post-hoc visualization mask, not certification'))
                if r['index'] in (45,51):
                    ids=np.flatnonzero(np.isclose(kx,ky)&(kx>=0));ids=ids[np.argsort(kx[ids])]
                    freq=np.flatnonzero((omega>0)&(omega<6));v=power[freq][:,ids]
                    # Log intensity referenced to each panel max, explicitly labelled.
                    v=np.log10(np.maximum(v/max(v.max(),1e-300),1e-8))
                    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
                    im=ax.pcolormesh(kx[ids]/np.pi,omega[freq],v,shading='nearest',vmin=-8,vmax=0,cmap='viridis')
                    pred=dispersion(kx[ids],ky[ids],reduced)
                    ax.plot(kx[ids]/np.pi,pred[0],'w--',lw=1,label='Internal analytic +')
                    ax.plot(kx[ids]/np.pi,pred[1],'w:',lw=1,label='Internal analytic -')
                    ax.set(xlabel='kx / pi = ky / pi',ylabel='Positive angular frequency',title=r['protocol']+'; positive circular channel only')
                    ax.legend(fontsize=8);fig.colorbar(im,ax=ax,label='log10(power / panel maximum), floor -8')
                    dest=ROOT/'assets/literature_reproduction/gomonay_2024'/aid/'figures'/f'spectrum_{r["protocol"]}.png'
                    fig.savefig(dest,dpi=160,facecolor='white');plt.close(fig)
    extra['checks']=dict(analyzed_tasks=len(rows),raw_hash_verified_files=sum(len(r['raw_sha256']) for r in rows),
        unit_spin_norm_max=max(r.get('max_saved_norm_error') or 0 for r in rows),
        missing_preprojection_error=True,formal_production_certified=False)
    write(out/'supplement.json',extra)
    write(out/'supplement_provenance.json',dict(code_sha256=sha256(Path(__file__)),
        input_sha256=sha256(out/'metrics.json'),status='exploratory_diagnostics_not_additional_gate'))
    print('Supplement complete')


if __name__=='__main__':main()
