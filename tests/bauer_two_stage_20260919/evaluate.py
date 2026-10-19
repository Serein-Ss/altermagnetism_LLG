"""Frozen dev selection; independent four-way size/time diagnostics, never automatic certification."""
import argparse
import json
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from tests.bauer_active_data_20260918.run import simulate
from tests.bauer_closed_loop_20260919.metrics import compare,summary
from .models import InitialNet,Net,sample_initial,sample_path
from .train import initial_metrics
from .common import HERE,cfg,dump,allocation,old_cfg,seed_stream,latent_seed


def load(name,device):
    ck=torch.load(HERE/'runs'/name/'best.pt',map_location=device,weights_only=False)
    net=(InitialNet(ck['config']['width'],ck['config']['depth']) if ck['kind']=='initial'
         else Net('C',ck['config']['width'],ck['config']['depth'])).to(device)
    net.load_state_dict(ck['model']);net.eval()
    return net,ck


def rollout(net,initial,duration,seed,kind,c,steps=None):
    blocks=round(duration/c['path_window_ps'])
    if not np.isclose(blocks*c['path_window_ps'],duration):raise ValueError('whole blocks only')
    times=torch.arange(round(c['path_window_ps']/c['numerics']['save_ps'])+1,device=initial.device)*c['numerics']['save_ps']
    frames=[initial[:,None].cpu().numpy()]
    for b in range(blocks):
        x=sample_path(net,initial,times,latent_seed(seed,b),steps or c['flow_steps'],kind,c)
        if not torch.isfinite(x).all() or float(abs(x.norm(dim=-1)-1).max())>2e-6:raise RuntimeError('invalid path')
        frames.append(x[:,1:].cpu().numpy());initial=x[:,-1]
    return np.concatenate(frames,1)


def generate():
    allocation('cuda');torch.set_num_threads(1);c=cfg();reports={}
    for kind in ['initial','uniform','anchored']:
        for seed in c['seeds']:
            name=f'{kind}_{seed}';reports[name]=json.loads((HERE/'runs'/name/'complete.json').read_text())
    chosen={'initial':min((n for n in reports if n.startswith('initial_')),key=lambda n:reports[n]['dev_score'])}
    chosen['path']=min((n for n in reports if not n.startswith('initial_')),key=lambda n:reports[n]['dev_score'])
    dump(HERE/'selection.json',dict(chosen=chosen,runs=reports,selection='development only, frozen before new test futures'))
    ini,_=load(chosen['initial'],'cuda');net,ck=load(chosen['path'],'cuda')
    from tests.bauer_closed_loop_20260919.run import records,reference
    convergence=[]
    for n in c['train_sizes']:
        row=next(r for r in records(old_cfg(),'dev') if r['L']==n)
        truth,t,s0=reference(row,old_cfg());s0=F.normalize(torch.tensor(s0,device='cuda',dtype=torch.float32),dim=-1)[None].repeat(16,1,1)
        paths={steps:sample_path(net,s0,torch.tensor(t,device='cuda',dtype=torch.float32),91200+n,steps,ck['kind'],c).cpu().numpy() for steps in [32,64,128]}
        for steps,x in paths.items():
            convergence.append(dict(L=n,steps=steps,paired_rmse_vs128=float(np.sqrt(np.mean((x-paths[128])**2))),**compare(x.astype('f8'),truth)))
    dump(HERE/'sampling_convergence.json',convergence)
    out=HERE/'generated';out.mkdir(exist_ok=False);timings=[]
    for n in c['train_sizes']+c['transfer_sizes']:
        torch.cuda.synchronize();start=time.perf_counter()
        initial=sample_initial(ini,n,64,seed_stream(95000,n),c['flow_steps'],'cuda');torch.cuda.synchronize()
        timings.append(dict(stage='initial',L=n,count=64,seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name()))
        with h5py.File(out/f'L{n}.h5','x') as h:
            h.attrs['complete']=False;h.create_dataset('initial',data=initial.cpu().numpy())
            h.create_dataset('time_ps',data=np.arange(1001)*c['numerics']['save_ps'])
            h.attrs['config_json']=json.dumps(c);h.attrs['selection_json']=json.dumps(chosen)
            h.attrs['initial_seed']=seed_stream(95000,n)
            for j in range(c['evaluation_parents']):
                with h5py.File(HERE/'data'/f'L{n}'/f'test_future_{j}.h5') as ref:s0=ref['initial'][:]
                for tag,s in [('RG',torch.tensor(s0,device='cuda',dtype=torch.float32)),('GG',initial[j])]:
                    anchor=F.normalize(s,dim=-1)[None].repeat(c['branches'],1,1)
                    torch.cuda.synchronize();start=time.perf_counter()
                    x=rollout(net,anchor,100.,seed_stream(960000+n*100+j,0 if tag=='RG' else 1),ck['kind'],c)
                    torch.cuda.synchronize();seconds=time.perf_counter()-start
                    d=h.create_dataset(f'{tag}_{j}',data=x,compression='lzf')
                    d.attrs['path_seed']=seed_stream(960000+n*100+j,0 if tag=='RG' else 1);h.flush()
                    timings.append(dict(stage=tag,L=n,parent=j,branches=c['branches'],duration_ps=100.,seconds=seconds,gpu=torch.cuda.get_device_name()))
                    print(n,tag,j,seconds,flush=True)
            h.attrs['complete']=True
    dump(out/'runtime.json',dict(measurements=timings,speed_scope='Raw GPU throughput with host transfer, no claim of accuracy-matched speedup'))


def cpu_reference(length):
    allocation();c=cfg();out=HERE/'generated';timing=[]
    with h5py.File(out/f'L{length}.h5') as h:
        assert h.attrs['complete'];initial=h['initial'][:]
    with h5py.File(out/f'GR_L{length}.h5','x') as h:
        h.attrs['complete']=False
        h.create_dataset('time_ps',data=np.arange(1001)*c['numerics']['save_ps']);h.attrs['config_json']=json.dumps(c)
        for j in range(c['evaluation_parents']):
            s=np.repeat(initial[j:j+1],2*c['branches'],0).astype('f8');s/=np.linalg.norm(s,axis=-1,keepdims=True)
            seeds=np.array([seed_stream(970000+length*100+j,k) for k in range(len(s))],dtype='i8')
            start=time.perf_counter();x,err=simulate(s,seeds,c,c['numerics']['dt_fs'],100.)
            timing.append(dict(parent=j,seconds=time.perf_counter()-start,branches=len(s),duration_ps=100.))
            h.create_dataset(f'paths_{j}',data=x.astype('f4'),compression='lzf')
            h.create_dataset(f'seeds_{j}',data=seeds);h.flush()
        h.attrs['complete']=True
    import numba
    numba.set_num_threads(1);torch.set_num_threads(1)
    chosen=json.loads((HERE/'selection.json').read_text())['chosen'];net,ck=load(chosen['path'],'cpu');initial_net,_=load(chosen['initial'],'cpu')
    s=initial[:1].astype('f8');s/=np.linalg.norm(s,axis=-1,keepdims=True);model_s=torch.tensor(s,dtype=torch.float32)
    simulate(s,np.array([4101]),c,c['numerics']['dt_fs'],1.);rollout(net,model_s,1.,4101,ck['kind'],c)
    measurements=[]
    plus=np.zeros_like(s);plus[...,2]=1.
    prep=dict(c,numerics=dict(c['numerics'],save_ps=c['numerics']['preparation_ps']))
    sample_initial(initial_net,length,1,4300,c['flow_steps'])
    for j in range(3):
        start=time.perf_counter();simulate(s,np.array([4200+j]),c,c['numerics']['dt_fs'],1.);llg=time.perf_counter()-start
        start=time.perf_counter();rollout(net,model_s,1.,4200+j,ck['kind'],c);model=time.perf_counter()-start
        start=time.perf_counter();simulate(plus,np.array([4300+j]),prep,c['numerics']['dt_fs'],c['numerics']['preparation_ps']);llg_prep=time.perf_counter()-start
        start=time.perf_counter();sample_initial(initial_net,length,1,4300+j,c['flow_steps']);model_init=time.perf_counter()-start
        measurements.append(dict(llg_seconds=llg,model_seconds=model,raw_ratio=llg/model,
                                 llg_preparation_seconds=llg_prep,model_initial_seconds=model_init,
                                 end_to_end_raw_ratio=(llg_prep+llg)/(model_init+model)))
    dump(out/f'cpu_L{length}.json',dict(reference=timing,same_cpu_one_thread_1ps=measurements,
         cpu=__import__('platform').processor(),benchmark='Same node, one thread, one path, 0.1ps output; warm; no file IO; not accuracy matched'))


def assess():
    c=cfg();oc=old_cfg();rows=[];initial_rows=[];gates={};overall=[]
    for n in c['train_sizes']+c['transfer_sizes']:
        with h5py.File(HERE/'data'/f'L{n}'/'test_initial.h5') as h:real=h['spins'][:]
        with h5py.File(HERE/'generated'/f'L{n}.h5') as h,h5py.File(HERE/'generated'/f'GR_L{n}.h5') as gr:
            assert h.attrs['complete'] and gr.attrs['complete']
            initial_rows.append(dict(L=n,**initial_metrics(h['initial'][:],real)))
            for horizon in c['reference_horizons_ps']:
                end=round(horizon/c['numerics']['save_ps'])+1
                for j in range(c['evaluation_parents']):
                    with h5py.File(HERE/'data'/f'L{n}'/f'test_future_{j}.h5') as ref:rr=ref['spins'][:,:end].astype('f8')
                    rg=h[f'RG_{j}'][:,:end].astype('f8');gg=h[f'GG_{j}'][:,:end].astype('f8');gref=gr[f'paths_{j}'][:,:end].astype('f8')
                    for tag,pred,truth,independent in [('RG',rg,rr[:16],rr[16:]),('GG',gg,gref[:16],gref[16:])]:
                        raw=compare(pred,truth);noise=compare(independent,truth)
                        rows.append(dict(L=n,family='plus_z_20ps',parent_id=f'L{n}_{j}',horizon_ps=horizon,arm=tag,
                                         raw=raw,reference_discrepancy=noise,**{k:raw[k]-noise[k] for k in raw}))
                    overall.append(dict(L=n,parent=j,horizon_ps=horizon,GR_vs_RR=compare(gref[:16],rr[:16]),GG_vs_RR=compare(gg,rr[:16]),
                                        warning='Unpaired independent initial samples; descriptive, not conditional matching'))
                for tag in ['RG','GG']:
                    selected=[r for r in rows if r['L']==n and r['horizon_ps']==horizon and r['arm']==tag]
                    stats=summary(selected,oc['margins'],c['bootstrap_repeats'])
                    gates[f'L{n}_{horizon:g}ps_{tag}']=dict(statistics=stats,pass_screen=all(v['status']=='WITHIN_PILOT_MARGIN' for v in stats.values()))
    initial_pass=all(r['max_w1']<=c['initial_margin'] for r in initial_rows)
    dump(HERE/'assessment.json',dict(initial=initial_rows,initial_pass=initial_pass,parents=rows,gates=gates,overall=overall,
         permit_long_diagnostic=initial_pass and all(v['pass_screen'] for v in gates.values()),scientific_certification=False,
         note='Paired excess over 16-vs-16 reference discrepancy; six parents per size, non-equivalence pilot gate. Literature conclusions remain unproven.'))


def long_generate():
    allocation('cuda');c=cfg();gate=json.loads((HERE/'assessment.json').read_text())
    if not gate['permit_long_diagnostic']:
        dump(HERE/'long_status.json',dict(status='BLOCKED_BY_PREDECLARED_SCREENS',reason='Initial, conditional size or time-ladder screen failed; do not promote an unvalidated model',details='assessment.json'));return
    chosen=json.loads((HERE/'selection.json').read_text())['chosen']
    ini,_=load(chosen['initial'],'cuda');net,ck=load(chosen['path'],'cuda');torch.set_num_threads(1)
    torch.cuda.synchronize();started=time.perf_counter()
    s=sample_initial(ini,100,c['long_paths'],980001,c['flow_steps'],'cuda')
    with h5py.File(HERE/'literature_model.h5','x') as h:
        frames=round(c['long_duration_ps']/c['numerics']['save_ps'])+1
        d=h.create_dataset('spins',(c['long_paths'],frames,100,3),dtype='f4',chunks=(1,1001,100,3),compression='lzf')
        h.attrs['complete']=False;d[:,0]=s.cpu().numpy();at=1
        h.attrs['config_json']=json.dumps(c);h.attrs['selection_json']=json.dumps(chosen)
        h.attrs['initial_seed']=980001;h.attrs['path_seed_root']=980002
        for block in range(round(c['long_duration_ps']/100)):
            x=rollout(net,s,100.,latent_seed(980002,block),ck['kind'],c)
            d[:,at:at+x.shape[1]-1]=x[:,1:];at+=x.shape[1]-1
            s=torch.tensor(x[:,-1],device='cuda');h.attrs['frames_written']=at;h.flush()
            dump(HERE/'long_progress.json',dict(block=block+1,time_ps=(block+1)*100))
        h.create_dataset('time_ps',data=np.arange(frames)*c['numerics']['save_ps']);h.attrs['complete']=True
    torch.cuda.synchronize()
    dump(HERE/'long_status.json',dict(status='GENERATED_DIAGNOSTIC_NOT_LITERATURE_CERTIFIED',duration_ps=c['long_duration_ps'],paths=c['long_paths'],seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['generate','reference','assess','long']);p.add_argument('--length',type=int);p.add_argument('--device',choices=['cuda']);a=p.parse_args()
    if a.phase in ('generate','long') and a.device!='cuda':p.error('GPU phases require --device cuda')
    if a.phase=='generate':generate()
    elif a.phase=='reference':cpu_reference(a.length)
    elif a.phase=='assess':assess()
    else:long_generate()
