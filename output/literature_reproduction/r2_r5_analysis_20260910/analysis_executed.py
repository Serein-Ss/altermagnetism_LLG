"""Read-only R2-R5 campaign audit and derived numerical diagnostics.

No production certification from post-hoc tolerances. Run in zrs-mag.
Raw files are hash-checked, streamed and never modified. Plotting reads derived data.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
import h5py
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.core.literature_config import read_document,sha256
from scripts.literature.bauer_2011.analyze import reversals,statistics
from scripts.literature.hirst_mn2au_2022.analyze import damped_afmr
from scripts.literature.hirst_mn2au_2022.wall_analysis import fit_wall
from scripts.literature.gomonay_2024.analyze import spectral_paths,wall_observables
from scripts.literature.gomonay_2024.model import dispersion
from scripts.literature.laliena_crnb3s6_2020.analyze import texture_observables


def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(clean(value),f,indent=2,allow_nan=False)


def summary(x):
    x=np.asarray(x).ravel()
    return dict(n=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)) if len(x)>1 else None,
                median=float(np.median(x)),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)),
                minimum=float(x.min()),maximum=float(x.max()))


def windows(t,y):
    """Descriptive equal-duration late windows, not an equivalence test."""
    t=np.asarray(t);y=np.asarray(y)
    lo=t[0]+.5*(t[-1]-t[0]);mid=t[0]+.75*(t[-1]-t[0])
    a=y[(t>=lo)&(t<mid)].mean(0);b=y[t>=mid].mean(0)
    return dict(intervals=[[float(lo),float(mid)],[float(mid),float(t[-1])]],
                first=a,second=b,change=b-a,certified=False)


def peaks(spec,r):
    omega=spec['omega'];power=spec['power'].sum(axis=(1,3))
    rows=[]
    for kx,ky in ((np.pi/2,0),(0,np.pi/2),(np.pi/2,np.pi/2),(-np.pi/2,np.pi/2)):
        j=int(np.argmin((spec['kx']-kx)**2+(spec['ky']-ky)**2))
        ip=np.flatnonzero(omega>0);im=np.flatnonzero(omega<0)
        pos=ip[np.argmax(power[ip,j])];neg=im[np.argmax(power[im,j])]
        predicted=np.asarray(dispersion(spec['kx'][j],spec['ky'][j],r))
        measured=np.array([omega[pos],-omega[neg]])
        rows.append(dict(k=[float(spec['kx'][j]),float(spec['ky'][j])],positive_peak=omega[pos],
            negative_peak=omega[neg],signed_splitting=omega[pos]+omega[neg],
            predicted_branches=predicted,unordered_max_abs_error=float(np.max(abs(np.sort(measured)-np.sort(predicted)))),
            peak_power=[power[pos,j],power[neg,j]],power_fraction=float(power[:,j].sum()/max(power.sum(),1e-300)),
            note='Global maxima in each frequency half-plane, no reference-guided peak selection; weakly excited modes not certified.'))
    return rows


def analyze_task(task,analysis_id):
    paper=task['paper'];p=task['parameters'];protocol=task['protocol'];run=task['run_id']
    raw=ROOT/'data/literature_reproduction'/paper/'raw'/run
    out=ROOT/'output/literature_reproduction'/paper/run
    dest=ROOT/'data/literature_reproduction'/paper/'derived'/run/analysis_id
    dest.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((out/'manifest.json').read_text())
    if sha256(task['config'])!=task['config_sha256']:raise ValueError('Configuration hash mismatch')
    doc=read_document(task['config']);r=doc['reduced']
    result=dict(index=task['index'],paper=paper,run_id=run,protocol=protocol,parameters=p,
                production_certified=False,raw_sha256={},configuration_sha256=task['config_sha256'])
    for name,digest in manifest['files_sha256'].items():
        path=raw/name
        if path.is_symlink() or not path.resolve().is_relative_to(raw.resolve()):raise ValueError('Unsafe raw path')
        actual=sha256(path)
        if actual!=digest:raise ValueError('Raw hash mismatch: '+name)
        result['raw_sha256'][name]=actual
    if protocol=='branch':
        with zipfile.ZipFile(raw/'branch.npz') as z:
            if sum(i.file_size for i in z.infolist())>128*2**20:raise ValueError('Oversized branch archive')
        with np.load(raw/'branch.npz',allow_pickle=False) as z:
            gamma=z['gamma'];i=int(np.argmax(gamma));profiles=z['profiles']
            result.update(fold_bracketed=bool(0<i<len(gamma)-1 and gamma[i]>gamma[i-1] and gamma[i]>gamma[i+1]),
                sampled_gamma_max=float(gamma[i]),reference_gamma_from_GUIDE=1.2405,
                reference_relative_error=float(gamma[i]/1.2405-1),max_BVP_residual=float(z['residual'].max()),
                branch_points=len(gamma))
            np.savez_compressed(dest/'series.npz',gamma=gamma,center_theta=profiles[:,0,0],x=z['x'],
                fold_profile=profiles[i],residual=z['residual'])
    else:
        with h5py.File(raw/'trajectory.h5','r') as h:
            if not h.attrs.get('complete',False):raise ValueError('Incomplete raw')
            for key in h:
                if not isinstance(h.get(key,getlink=True),h5py.HardLink):raise ValueError('External/soft HDF link')
                if isinstance(h[key],h5py.Dataset) and (h[key].is_virtual or h[key].external):raise ValueError('External storage')
            t=h['time'][:];mean=[];spread=[];norm=0.;finite=True;wall=[];texture=[]
            for frame in h['spins']:
                finite &= bool(np.isfinite(frame).all())
                flat=frame.reshape(frame.shape[0],-1,3)
                mean.append(flat.mean(1));spread.append(flat.std(1))
                norm=max(norm,float(np.max(abs(np.linalg.norm(flat,axis=-1)-1))))
                if paper=='hirst_mn2au_2022' and protocol=='wall':wall.append(fit_wall(frame))
                if paper=='laliena_crnb3s6_2020':texture.append(texture_observables(frame,p['dx']))
            if not finite:raise ValueError('Nonfinite trajectory')
            series=dict(time=t,mean=np.array(mean),spatial_sd=np.array(spread),final_spins=h['spins'][-1])
            result.update(saved_frames=len(t),duration=float(t[-1]-t[0]),finite=True,
                max_saved_norm_error=norm if not protocol.startswith('llb') else None,
                projection_predictor_error_saved=False,
                norm_gate='not_applicable_LLB' if protocol.startswith('llb') else ('pass' if norm<1e-10 else 'fail'))
            if 'energy' in h:
                energy=h['energy'][:]/np.prod(h['spins'].shape[2:-1]);series['energy_per_spin']=energy
                result['energy']=dict(initial=energy[0],final=energy[-1],maximum_saved_increase=np.diff(energy,axis=0).max(0),
                    relative_excursion=np.max(abs(energy-energy[0]),axis=0)/np.maximum(abs(energy[0]),1e-30))
            if paper=='bauer_2011':
                mz=series['mean'][...,2]
                result.update(endpoint=summary(mz[-1]),negative_endpoint_count=int((mz[-1]<0).sum()),
                    crossed_zero_count=int((mz.min(0)<0).sum()),late_windows=windows(t,mz))
                result['operational_event_sensitivity']=[dict(threshold=a,dwell=d,events=sum(len(reversals(t,v,a,d)) for v in mz.T))
                    for a in (.6,.7,.8) for d in (1.,5.,10.)]
                result['survival_diagnostic']=statistics(t,mz,threshold=.7,dwell=5.,bootstrap=1000,seed=20260910)
                result['survival_note']='Post-hoc threshold/dwell sensitivity only; not a calibrated basin. Zero-event bootstrap is degenerate, not evidence of zero probability.'
            elif paper=='hirst_mn2au_2022':
                a=h['m_a'][:];b=h['m_b'][:];series.update(m_a=a,m_b=b)
                if protocol=='wall':
                    widths=np.array([[f['neel_width_over_a'] for f in row] for row in wall]);series['neel_width_over_a']=widths
                    result.update(wall_final=wall[-1],wall_late_windows=windows(t,widths),
                        target_width_over_a=31.2/.333,reference_note='GUIDE 31.2 nm / YAML a=0.333 nm; compare reduced pi*delta0, not tanh delta0.')
                elif protocol=='equilibrium':
                    lengths=np.linalg.norm(a,axis=-1);series['sublattice_length']=lengths
                    result.update(late_windows=windows(t,lengths),late_length_by_replica=lengths[t>=t[-1]/2].mean(0),
                        independent_trajectories=a.shape[1],uncertainty='One trajectory per condition; no independent-replicate CI.')
                else:
                    result['afmr']=[damped_afmr(t,a[:,b,2]) for b in range(a.shape[1])]
                    result['resolved_cycles']=[v['parameters'][1]*(t[-1]-t[0])/(2*np.pi) if v['resolved'] else None for v in result['afmr']]
            elif paper=='gomonay_2024':
                if protocol in ('spinwave','control'):
                    spec=spectral_paths(h);np.savez_compressed(dest/'spectrum.npz',**spec)
                    theory=dict(r,J_tilde=0.) if protocol=='control' else r
                    result.update(spectral_peaks=peaks(spec,theory),frequency_bin=spec['frequency_bin'],
                        nyquist=float(np.pi/np.diff(t).max()),reference_type='Internal analytic formula; external digitized curve comparison still missing.')
                else:
                    obs=wall_observables(h);vals=np.array(obs['observables']);series['wall_observables']=vals
                    result.update(wall=obs,wall_late_windows=windows(t,vals[:,:,1]))
            else:
                for key in texture[0]:series[key]=np.array([x[key] for x in texture])
                result.update(final_winding=series['winding'][-1],final_mean_wavevector=series['mean_wavevector'][-1],
                    final_max_abs_nz=series['max_abs_nz'][-1],late_wavevector=windows(t,series['mean_wavevector']))
                if protocol=='current':
                    length=p['sites']*p['dx'];center=np.unwrap(series['center']*2*np.pi/length,axis=0)*length/(2*np.pi)
                    series['unwrapped_center']=center;late=t>=t[-1]/2
                    result.update(velocity_late=[float(np.polyfit(t[late],center[late,b],1)[0]) for b in range(center.shape[1])],
                        velocity_expected=r['beta']/r['alpha']*p['u'],width_late_windows=windows(t,series['phase_gradient_rms_width']),
                        interpretation='Velocity/width valid only for a surviving isolated soliton; RMS width is not the BVP tanh width.')
            np.savez_compressed(dest/'series.npz',**series)
    write(dest/'metrics.json',result)
    report=out/analysis_id;report.mkdir(exist_ok=False)
    write(report/'report.json',result)
    return result


def plot_task(task,analysis_id):
    paper=task['paper'];run=task['run_id'];protocol=task['protocol']
    source=ROOT/'data/literature_reproduction'/paper/'derived'/run/analysis_id
    dest=ROOT/'assets/literature_reproduction'/paper/run/analysis_id/'figures';dest.mkdir(parents=True,exist_ok=False)
    with np.load(source/'series.npz',allow_pickle=False) as z:
        fig,axes=plt.subplots(1,3,figsize=(13,3.7),constrained_layout=True)
        if protocol=='branch':
            axes[0].plot(z['center_theta'],z['gamma'],'o-',ms=2);axes[0].axhline(1.2405,color='k',ls='--',label='GUIDE target');axes[0].legend()
            axes[0].set(xlabel='Center theta (rad)',ylabel='Gamma')
            axes[1].plot(z['x'],z['fold_profile'][0]);axes[1].set(xlabel='Reduced x',ylabel='Theta at sampled fold (rad)')
            axes[2].plot(z['residual']);axes[2].set(xlabel='Continuation sample',ylabel='BVP residual')
        else:
            t=z['time']
            if 'm_a' in z:
                for c,ls in enumerate(('-','--',':')):axes[0].plot(t,z['m_a'][:,0,c],ls,label='xyz'[c])
                axes[0].set(ylabel='Sublattice a magnetization');axes[0].legend()
            else:
                axes[0].plot(t,z['mean'][...,2],alpha=.65);axes[0].set(ylabel='Spatial mean s_z')
            axes[0].set(xlabel='Reduced time')
            if 'neel_width_over_a' in z:axes[1].plot(t,z['neel_width_over_a']);axes[1].set(ylabel='Neel width / a')
            elif 'wall_observables' in z:axes[1].plot(t,z['wall_observables'][:,:,1]);axes[1].set(ylabel='Wall tanh width / a0')
            elif 'winding' in z:axes[1].plot(t,z['winding']);axes[1].set(ylabel='Winding (all replicas)')
            else:axes[1].plot(t,z['spatial_sd'][...,2]);axes[1].set(ylabel='Spatial SD of s_z')
            axes[1].set(xlabel='Reduced time')
            if 'energy_per_spin' in z:axes[2].plot(t,z['energy_per_spin']);axes[2].set(ylabel='Reduced energy / spin')
            else:axes[2].plot(t,np.linalg.norm(z['m_a'],axis=-1));axes[2].set(ylabel='LLB sublattice length')
            axes[2].set(xlabel='Reduced time')
        fig.suptitle(f'{paper}: {protocol}, task {task["index"]} (validation only)',fontsize=11)
        fig.savefig(dest/'diagnostics.png',dpi=160,facecolor='white');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--analysis-id',required=True);args=parser.parse_args()
    if Path(args.analysis_id).name!=args.analysis_id:raise ValueError('one-component analysis id required')
    m=json.loads((args.campaign/'manifest.json').read_text());results=[]
    for task in m['tasks']:
        print('Analyzing',task['index'],task['paper'],task['protocol'],flush=True)
        results.append(analyze_task(task,args.analysis_id));plot_task(task,args.analysis_id)
    out=ROOT/'output/literature_reproduction'/args.analysis_id;out.mkdir(exist_ok=False)
    write(out/'metrics.json',results)
    sources=[Path(__file__),ROOT/'GUIDE/STRICT_LITERATURE_REPRODUCTION_PLAN.md']+list((ROOT/'scripts/literature').rglob('*.py'))
    write(out/'manifest.json',dict(analysis_id=args.analysis_id,campaign=str(args.campaign.relative_to(ROOT)),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sources},campaign_sha256=sha256(args.campaign/'manifest.json'),
        versions=dict(numpy=np.__version__,scipy=scipy.__version__,h5py=h5py.__version__,matplotlib=matplotlib.__version__),
        status='analysis_complete_not_production_certified',command=sys.argv,
        statistical_scope='Descriptive, operational survival sensitivity; no post-hoc equivalence threshold or p-value fishing.'))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()

