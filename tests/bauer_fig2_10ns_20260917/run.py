"""Three fixed-seed Bauer Figure 2 paths; CPU simulation and offline gallery."""
import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import yaml
from .kernel import ensemble

BASE=Path(__file__).resolve().parent
HBAR_MEV_PS=0.6582119569
KB_MEV_K=0.08617333262


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def counts(duration_ps,dt_ps,save_ps):
    steps=round(duration_ps/dt_ps); stride=round(save_ps/dt_ps)
    if steps<1 or stride<1 or steps%stride or not np.isclose(steps*dt_ps,duration_ps) or not np.isclose(stride*dt_ps,save_ps):
        raise ValueError('time grid must be exactly divisible')
    return steps,stride


def observables(s,k):
    energy=-(s[...,:-1,:]*s[...,1:,:]).sum((-1,-2))-k*(s[...,2]**2).sum(-1)
    return s.sum(-2),energy/s.shape[-2],np.abs(np.linalg.norm(s,axis=-1)-1).max(-1)


def simulate(cfg,out):
    p=cfg['physics']; n=cfg['numerics']; initial=cfg['initial']
    t0=HBAR_MEV_PS/p['exchange_meV']; dt_ps=n['dt_fs']/1000; dt=dt_ps/t0
    duration=n['duration_ns']*1000
    steps,stride=counts(n['block_ps'],dt_ps,n['save_ps'])
    total_steps,_=counts(duration,dt_ps,n['save_ps'])
    if total_steps%steps: raise ValueError('duration must contain whole blocks')
    batch=n['trajectories']; seeds=np.asarray(n['seeds'],dtype=np.int64)
    if batch!=len(seeds) or batch>3 or len(set(seeds))!=batch: raise ValueError('at most three unique seeds')
    if p['boundary']!='open' or p['external_field']!=0 or p['convention']!='bauer_ll':
        raise ValueError('this kernel implements only the stated zero-field open Bauer chain')
    start=time.monotonic()
    s=np.zeros((batch,p['length'],3)); s[...,2]=1
    prep_steps,prep_stride=counts(initial['preparation_ps'],dt_ps,n['save_ps'])
    prep,prep_error=ensemble(s,seeds,dt,prep_steps,prep_stride,p['anisotropy_over_J'],p['damping_lambda'],p['temperature_kBT_over_J'],0,True)
    state=prep[:,-1].copy()
    frames=total_steps//stride+1
    with h5py.File(out/'trajectory.h5','x') as h:
        h.attrs['complete']=False; h.attrs['config_json']=json.dumps(cfg)
        h.attrs['time_unit']='ps'; h.attrs['temperature_K']=p['temperature_kBT_over_J']*p['exchange_meV']/KB_MEV_K
        h.attrs['dt_reduced']=dt; h.attrs['preparation_is_equilibrium_certified']=False
        h.create_dataset('seed',data=seeds)
        h.create_dataset('preparation_spins',data=prep,compression='lzf')
        h.create_dataset('preparation_time_ps',data=np.arange(prep.shape[1])*n['save_ps'])
        h.create_dataset('time_ps',data=np.arange(frames)*n['save_ps'])
        spins=h.create_dataset('spins',(batch,frames,p['length'],3),dtype='f4',chunks=(1,min(256,frames),p['length'],3),compression='lzf')
        magnet=h.create_dataset('M_total',(batch,frames,3),dtype='f8')
        energy=h.create_dataset('energy_per_spin_J',(batch,frames),dtype='f8')
        norm=h.create_dataset('max_norm_error_float64',(batch,frames),dtype='f8')
        spins[:,0]=state; magnet[:,0],energy[:,0],norm[:,0]=observables(state,p['anisotropy_over_J'])
        raw_error=prep_error.copy(); saved=1
        for block in range(total_steps//steps):
            block_seeds=seeds+(block+1)*1000
            samples,err=ensemble(state,block_seeds,dt,steps,stride,p['anisotropy_over_J'],p['damping_lambda'],p['temperature_kBT_over_J'],0,True)
            if not np.isfinite(samples).all(): raise ValueError('nonfinite trajectory')
            part=samples[:,1:]; end=saved+part.shape[1]
            spins[:,saved:end]=part
            magnet[:,saved:end],energy[:,saved:end],norm[:,saved:end]=observables(part,p['anisotropy_over_J'])
            state=samples[:,-1].copy(); raw_error=np.maximum(raw_error,err); saved=end
            h.attrs['completed_frames']=saved; h.flush()
            checkpoint=out/'checkpoint.tmp.npz'
            np.savez(checkpoint,state=state,next_block=block+1)
            checkpoint.replace(out/'checkpoint.npz')
            write_json(out/'progress.json',{'completed_ps':(saved-1)*n['save_ps'],'requested_ps':duration,'wall_seconds':time.monotonic()-start})
        h.attrs['complete']=True; h.attrs['max_preprojection_norm_error']=raw_error
        report={'simulation_complete':True,'distribution_certified':False,'trajectories':batch,'frames_per_path':frames,
                'duration_ns':n['duration_ns'],'temperature_K':float(h.attrs['temperature_K']),'dt_fs':n['dt_fs'],
                'dt_reduced':dt,'save_ps':n['save_ps'],'initial_M_total':magnet[:,0].tolist(),
                'initial_protocol':initial,'seed_schedule':'preparation=seed; recorded block b uses seed+(b+1)*1000',
                'max_saved_float64_norm_error':float(norm[:].max()),'max_preprojection_norm_error':raw_error.tolist(),
                'wall_seconds':time.monotonic()-start,'job_id':os.getenv('SLURM_JOB_ID'),
                'warning':'No completed-switch/lifetime claim; projected weak RK and chosen time step are not distribution-certified.'}
    write_json(out/'simulation_report.json',report)


def render(out,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from PIL import Image
    report=json.loads((out/'simulation_report.json').read_text())
    smoke=cfg['numerics']['duration_ns']!=10
    heading='短流程测试，不是10 ns正式结果' if smoke else 'Bauer 图2：3条独立10 ns轨迹'
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>'+heading+'</title><style>body{max-width:1250px;margin:30px auto;padding:15px;font-family:system-ui;line-height:1.7}img{max-width:100%;height:auto}pre{white-space:pre-wrap;background:#f2f4f6;padding:15px}</style>',
          '<h1>'+heading+'</h1><p>这是随机LL模拟器结果，不是生成模型结果。物理条件来自Bauer Figure 2：L=100，开放边界，K/J=0.1，kBT/J=0.11，λ=0.1，无外场。J=50 meV用于换算，T≈63.82 K。所有预设seed均展示，不筛选翻转结果。</p>',
          '<p>初态从+z开始，固定有限温预演化后进入记录阶段；不强制总Mz=85、不预设畴壁，也不宣称初态已严格热化。20 ps制备、10 ns窗口、0.25 fs步长及每步最终归一化均是项目选择。原文图2窗口约9.87 ns。</p>',
          '<p>静态图分三段：上方是100个磁矩的总Mz，下方横轴为时间、纵轴为全链站点，蓝色负sz、绿色正sz。按固定时间间隔抽取最多4000列用于图像显示，原始数据保留所有保存帧；图像不足以排除短暂事件。GIF仅是等间隔抽样的动态概览，不插值，不能用于判定精确翻转时间。</p>',
          '<pre>'+html.escape(json.dumps(report,indent=2,ensure_ascii=False))+'</pre>']
    def embed(path):
        mime='gif' if path.suffix=='.gif' else 'png'
        return '<img src="data:image/'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
    with h5py.File(out/'trajectory.h5') as h:
        if not h.attrs['complete']: raise ValueError('cannot render incomplete data as complete')
        t=h['time_ps'][:]/1000; length=h['spins'].shape[2]
        for b in range(h['spins'].shape[0]):
            s=h['spins'][b]; mz=h['M_total'][b,:,2]
            fig,axes=plt.subplots(2,3,figsize=(16,7),gridspec_kw={'height_ratios':[1,2]},constrained_layout=True)
            edges=np.linspace(0,t[-1],4)
            for j in range(3):
                ids=np.flatnonzero((t>=edges[j])&(t<=edges[j+1]))
                ids=ids[np.unique(np.linspace(0,len(ids)-1,min(4000,len(ids)),dtype=int))]
                axes[0,j].plot(t,mz,color='black',lw=.6); axes[0,j].set(xlim=edges[j:j+2],ylim=(-length,length),ylabel='Total Mz',title=f'Seed {int(h["seed"][b])}: interval {j+1}')
                im=axes[1,j].imshow((s[ids,:,2]>=0).T,origin='lower',aspect='auto',interpolation='nearest',
                       extent=(t[ids[0]],t[ids[-1]],-.5,length-.5),vmin=0,vmax=1,cmap=ListedColormap(['#315ca8','#43a047']))
                axes[1,j].set(xlabel='Time (ns)',ylabel='Chain site')
            bar=fig.colorbar(im,ax=axes[1,:],ticks=[0,1],shrink=.7); bar.ax.set_yticklabels(['sz < 0','sz >= 0'])
            path=out/f'path_{b}_figure2.png'; fig.savefig(path,dpi=150); plt.close(fig)
            page.append(f'<h2>轨迹{b}，seed={int(h["seed"][b])}</h2>'+embed(path))
            indices=np.unique(np.linspace(0,len(t)-1,min(cfg['visualization']['gif_frames'],len(t)),dtype=int))
            fig,axes=plt.subplots(2,1,figsize=(12,5),gridspec_kw={'height_ratios':[2,1]},constrained_layout=True)
            axes[0].plot(t,mz,lw=.6); axes[0].set(xlabel='Time (ns)',ylabel='Total Mz',ylim=(-length,length))
            cursor=axes[0].axvline(0,color='red')
            heat=axes[1].imshow(s[0].T,origin='lower',aspect='auto',cmap='coolwarm',vmin=-1,vmax=1,interpolation='nearest')
            axes[1].set_yticks([0,1,2]); axes[1].set_yticklabels(['sx','sy','sz']); axes[1].set_xlabel('Chain site (all sites)')
            fig.colorbar(heat,ax=axes[1],label='Spin component',ticks=[-1,0,1])
            frames=[]
            for k in indices:
                cursor.set_xdata([t[k],t[k]]); heat.set_data(s[k].T)
                axes[0].set_title(f'Path {b} | t={t[k]:.4f} ns | overview subsampling, not physical playback speed')
                fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
            path=out/f'path_{b}.gif'; frames[0].save(path,save_all=True,append_images=frames[1:],duration=100,loop=0); plt.close(fig)
            page.append(embed(path))
        fig,axes=plt.subplots(2,1,figsize=(10,6),constrained_layout=True)
        for b in range(h['spins'].shape[0]):
            axes[0].plot(t,h['energy_per_spin_J'][b],label=f'path {b}')
            axes[1].plot(t,h['max_norm_error_float64'][b],label=f'path {b}')
        axes[0].set(xlabel='Time (ns)',ylabel='Energy / spin / J'); axes[0].legend()
        axes[1].set(xlabel='Time (ns)',ylabel='max abs(norm-1), float64 saved states'); axes[1].legend()
        path=out/'diagnostics.png'; fig.savefig(path,dpi=150); plt.close(fig); page.append(embed(path))
    page.append('<p>磁矩以float32保存，模长诊断由写盘前float64状态计算；读回float32可能出现约1e-7误差。有限温能量不要求单调下降。未进行寿命拟合或统计结论认证。</p></html>')
    (out/'index.html').write_text('\n'.join(page))
    write_json(out/'complete.json',{'simulation_and_gallery_complete':True,'distribution_certified':False})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    cfg=yaml.safe_load((BASE/'config.yaml').read_text())
    if args.smoke:
        cfg['initial']['preparation_ps']=.1
        cfg['numerics'].update(duration_ns=.003,block_ps=1.)
        cfg['visualization']['gif_frames']=7
    out=BASE/('smoke_results' if args.smoke else 'results'); out.mkdir(exist_ok=True)
    if (out/'trajectory.h5').exists(): raise FileExistsError('refusing to overwrite existing trajectories')
    write_json(out/'source_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'kernel.py',BASE/'run.py',BASE/'config.yaml']})
    simulate(cfg,out); render(out,cfg)


if __name__=='__main__': main()
