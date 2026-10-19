"""Three-seed controlled base-law comparison and independently trained initial model."""
import argparse
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import wasserstein_distance
from scripts.model.gate_f import path_interpolate
from tests.bauer_active_data_20260918.run import descriptors
from tests.bauer_closed_loop_20260919.run import records,reference,batch
from tests.bauer_closed_loop_20260919.metrics import compare,summary
from tests.mdgen_path_refactor_20260918.data import PathWindows
from .common import HERE,ROOT,cfg,dump,allocation,old_cfg
from .models import InitialNet,Net,base_path,sample_initial,sample_path


def initial_metrics(a,b):
    x,y=descriptors(a),descriptors(b)
    values=[float(wasserstein_distance(x[:,i],y[:,i])) for i in range(x.shape[1])]
    return dict(mean_w1=float(np.mean(values)),max_w1=max(values),descriptor_w1=values,
                note='Unstandardized bounded descriptor W1; not full configuration-distribution equivalence')


def initial_data(split):
    result={}
    for n in cfg()['train_sizes']:
        with h5py.File(HERE/'data'/f'L{n}'/f'{split}_initial.h5') as h:result[n]=h['spins'][:].astype('f4')
    return result


def evaluate(net,kind,c,subset=None):
    if kind=='initial':
        rows=[]
        for n,s in initial_data('dev').items():
            x=sample_initial(net,n,len(s),800+n,c['dev_flow_steps'],'cuda').cpu().numpy()
            rows.append(dict(L=n,**initial_metrics(x,s)))
        return dict(score=float(np.mean([r['mean_w1'] for r in rows])),parents=rows)
    oc=old_cfg();rows=[]
    selected=records(oc,'dev')
    if subset:selected=selected[:subset]
    for j,r in enumerate(selected):
        truth,t,init=reference(r,oc)
        initial=F.normalize(torch.tensor(init,device='cuda',dtype=torch.float32),dim=-1)[None].repeat(16,1,1)
        times=torch.tensor(t,device='cuda',dtype=torch.float32)
        x=sample_path(net,initial,times,74000+j,c['dev_flow_steps'],kind,c).cpu().numpy()
        rows.append(dict(parent_id=r['parent_id'],L=r['L'],family=r['family'],**compare(x.astype('f8'),truth)))
    stats=summary(rows,oc['margins'],c['bootstrap_repeats'])
    return dict(score=float(np.mean([v['mean']/v['margin'] for v in stats.values()])),parents=rows,statistics=stats)


def run(index,smoke=False):
    allocation('cuda');c=cfg();torch.set_num_threads(1)
    kinds=['initial','uniform','anchored'];kind=kinds[index//3];seed=c['seeds'][index%3]
    if smoke:c.update(updates=2,validate_every=2,batch=2,dev_flow_steps=2,bootstrap_repeats=20)
    out=HERE/('smoke_runs' if smoke else 'runs')/f'{kind}_{seed}';out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);rng=np.random.default_rng(seed);trng=torch.Generator(device='cuda').manual_seed(seed+2000)
    net=(InitialNet(c['width'],c['depth']) if kind=='initial' else Net('C',c['width'],c['depth'])).cuda()
    if kind!='initial':
        old=torch.load(ROOT/f'tests/bauer_closed_loop_20260919/runs/C_{seed}/best.pt',map_location='cuda',weights_only=False)
        net.load_state_dict(old['model'])
        datasets=[PathWindows([ROOT/old_cfg()['data']/f'L{n}'],'train',1.,stride_frames=10) for n in c['train_sizes']]
    else:data=initial_data('train')
    opt=torch.optim.AdamW(net.parameters(),lr=c['learning_rate'],weight_decay=1e-5)
    history=[];best=float('inf');started=time.perf_counter()
    for step in range(1,c['updates']+1):
        net.train()
        if kind=='initial':
            s=data[c['train_sizes'][int(rng.integers(3))]]
            y=F.normalize(torch.tensor(s[rng.integers(len(s),size=c['batch'])],device='cuda'),dim=-1)
            z=F.normalize(torch.randn(y.shape,device='cuda',generator=trng),dim=-1)
        else:
            y,t=batch(datasets,rng,dict(old_cfg(),batch=c['batch']),'cuda','C')
            z=base_path(y[:,0],len(t),trng,kind,c['anchored_sigma'],c['anchored_time_rho'])
        tau=torch.rand(len(y),device='cuda',generator=trng)
        if kind=='initial':
            dummy=torch.zeros_like(y);dummy[...,2]=1.
            x,v,_=path_interpolate(torch.stack((dummy,z),1),torch.stack((dummy,y),1),tau,trng)
            x,v=x[:,1],v[:,1]
        else:x,v,_=path_interpolate(z,y,tau,trng)
        pred=net(x,tau) if kind=='initial' else net(x,tau,y[:,0],t,63.82484966960072)
        loss=(pred-v).square().sum(-1)
        loss=loss.mean() if kind=='initial' else loss[:,1:].mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not torch.isfinite(grad):raise RuntimeError('nonfinite gradient')
        opt.step()
        if step%100==0:print(kind,seed,step,float(loss),flush=True)
        if step%c['validate_every']==0:
            net.eval();dev=evaluate(net,kind,c,2 if smoke else None)
            history.append(dict(step=step,loss=float(loss),dev_score=dev['score']))
            dump(out/f'dev_{step}.json',dev)
            state=dict(model=net.state_dict(),kind=kind,seed=seed,step=step,config=c,dev_score=dev['score'])
            if dev['score']<best:
                best=dev['score'];torch.save(state,out/'best.pt')
            torch.save(dict(state,optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=trng.get_state()),out/'last.pt')
            dump(out/'history.json',history)
    torch.cuda.synchronize()
    dump(out/'complete.json',dict(kind=kind,seed=seed,dev_score=best,seconds=time.perf_counter()-started,
                                 gpu=torch.cuda.get_device_name(),updates=c['updates'],smoke=smoke,
                                 protocol=c['initial_protocol'],scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0);p.add_argument('--device',choices=['cuda'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.index,a.smoke)
