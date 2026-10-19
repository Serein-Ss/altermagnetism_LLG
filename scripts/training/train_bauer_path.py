"""Explicitly launched development training; no automatic job submission."""
import argparse,json,time
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from scripts.model.bauer_path import rotate
from scripts.model.path_sampling import draw,differentiable_sample,require_device
from scripts.inference.bauer_path import load_checkpoint
from scripts.datasets.bauer_path_windows import PathWindows
from scripts.datasets.bauer_conditional import load_bank,RefillWindows
from scripts.training.bauer_path_teacher import labels
from scripts.training.bauer_path_objectives import spatial_moment_loss
from scripts.training.bauer_path_recipe import schedule,rate,roles

ROOT=Path(__file__).resolve().parents[2]


def native_batch(datasets,rng,batch,device):
    ds=datasets[int(rng.integers(len(datasets)))];samples=[]
    for _ in range(batch):
        i=int(rng.integers(len(ds)))
        if rng.random()<.5:
            g=int(np.searchsorted(ds.ends,i,side='right'));start=ds.ends[g-1] if g else 0;nstarts=len(ds.records[g][2])
            i=start+((i-start)//nstarts)*nstarts
        samples.append(ds[i]['spins'])
    return F.normalize(torch.tensor(np.stack(samples),device=device),dim=-1)


def train(config,seed,out,device='cuda',updates=None,fresh_rng=False):
    device=require_device(device);config=yaml.safe_load(Path(config).read_text());total=updates or config['updates']
    if total<1:raise ValueError('positive update count required')
    # A short software smoke can consume a prefix; declared experiments use full cycles.
    ids,branches=schedule(982300+seed,16*((total+15)//16))
    ckpath=ROOT/config['checkpoint'].format(seed=seed);net,ck=load_checkpoint(ckpath,device);net.train()
    if config['physical_lag_ps']!=.1:raise ValueError('checkpoint has fixed0.1ps lag')
    opt=torch.optim.AdamW(net.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair=np.random.default_rng();pair.bit_generator.state=ck['pair_rng']
    # CPU and CUDA use different RNG algorithms. Never call CPU smoke an exact resume.
    gs={k:torch.Generator(device=device) for k in ck['rng']}
    for index,(name,generator) in enumerate(gs.items()):
        if fresh_rng:generator.manual_seed(983000+seed+index)
        else:generator.set_state(ck['rng'][name])
    aug=torch.Generator(device=device).manual_seed(982100+seed);extra=torch.Generator(device=device).manual_seed(982200+seed)
    aux=torch.Generator(device=device).manual_seed(982400+seed);aux_rng=np.random.default_rng(982500+seed)
    real=load_bank(ROOT/config['real_bank']);model=load_bank(ROOT/config['model_bank']);bank=(real,model,model)
    gpu=[[F.normalize(torch.tensor(x,device=device),dim=-1) for x in b] for b in bank]
    pool={}
    for row in json.loads((ROOT/config['refill_manifest']).read_text()):
        with h5py.File(ROOT/row['old_file']) as h:old=h['spins'][:,:11]
        with h5py.File(ROOT/row['new_file']) as h:new=h['spins'][:].astype(np.float32)
        pool[row['parent_id']]=np.concatenate((old,new))
    datasets=[RefillWindows(PathWindows([ROOT/config['native_data']/f'L{n}'],'train',1.,10),pool,refill) for n in config['sizes']]
    out=Path(out);out.mkdir(parents=True,exist_ok=False);history=[];tick=time.perf_counter()
    for step in range(1,total+1):
        whole=native_batch(datasets,rng,config['batch'],device)
        starts=np.where(pair.random(len(whole))<.5,0,pair.integers(1,10,len(whole)));bi=torch.arange(len(whole),device=device)
        starts=torch.as_tensor(starts,device=device);y=torch.stack((whole[bi,starts],whole[bi,starts+1]),1)
        z=draw(y[:,0],11,gs['source'],0.)[:,:2];tau=torch.rand(len(y),device=device,generator=gs['tau'])
        cut=gs['cut'].get_state();torch.randn(whole.shape,device=device,generator=gs['cut'])
        x,v,_=path_interpolate(z,y,tau,torch.Generator(device=device).set_state(cut))
        angle=torch.rand(len(y),device=device,generator=aug)*(2*np.pi)
        x,v,anchor=rotate(x,angle),rotate(v,angle),rotate(y[:,0],angle)
        pred=net.velocity(x,tau,anchor,y.new_tensor([0.,.1]));fm=(pred[:,1:]-v[:,1:]).square().sum(-1).mean()
        opt.zero_grad(set_to_none=True);fm.backward()
        rr=roles(step);ii,jj=ids[step-1],branches[step-1]
        yr=F.normalize(torch.tensor(np.stack([bank[r][i][j] for r,i,j in zip(rr,ii,jj)]),device=device),dim=-1)
        zr=draw(yr[:,0],2,extra,0.);tr=torch.rand(len(yr),device=device,generator=extra);drafts=[];targets=[]
        for start in (0,16,32):
            xx,_,vv,_=labels(zr[start:start+16],yr[start:start+16],tr[start:start+16],gpu[rr[start]][ii[start]],extra)
            drafts.append(xx);targets.append(vv)
        xr,vr=torch.cat(drafts),torch.cat(targets);angles=torch.rand(3,device=device,generator=aug).repeat_interleave(16)*(2*np.pi)
        anchor=rotate(xr[:,0],angles);xr,vr=rotate(xr,angles),rotate(vr,angles)
        pred=net.velocity(xr,tr,anchor,yr.new_tensor([0.,.1]));replay=(pred[:,1:]-vr[:,1:]).square().sum(-1).mean()
        (config['teacher_weight']*replay).backward()
        if step%8==0:
            role=2*(((step-1)//64)%2);sid=int(aux_rng.integers(len(bank[role])))
            target=gpu[role][sid];initial=target[:1,0].expand(config['auxiliary_branches'],-1,-1);latent=draw(initial,2,aux,0.)
            if step%config['moment_frequency']==0:
                generated=differentiable_sample(net,initial,latent,config['auxiliary_flow_steps'],True);moment,_=spatial_moment_loss(generated,target,True)
                (config['moment_peak_weight']*min(1.,step/config['moment_ramp_updates'])*moment).backward()
        for group in opt.param_groups:group['lr']=rate(step,total,config['learning_rate'],config['learning_rate_floor'],config['warmup_updates'])
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),config['gradient_clip'])
        if not torch.isfinite(fm) or not torch.isfinite(replay) or not torch.isfinite(grad):raise RuntimeError('nonfinite training')
        opt.step()
        if step%100==0 or step==total:history.append(dict(step=step,fm=float(fm.detach()),replay=float(replay.detach()),gradient=float(grad)))
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=ck['step']+total,source_checkpoint=str(ckpath),
        task=dict(seed=seed,repair='portable_whole_teacher'),rng={k:g.get_state() for k,g in gs.items()},
        data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,pair_rng=pair.bit_generator.state,
        extra_generator=extra.get_state(),augmentation_generator=aug.get_state(),auxiliary_generator=aux.get_state(),
        auxiliary_rng=aux_rng.bit_generator.state,fresh_rng=fresh_rng,scientific_certification=False),out/'checkpoint.pt')
    (out/'training.json').write_text(json.dumps(dict(history=history,seconds=time.perf_counter()-tick,config=config,updates=total,balanced_cycles_complete=total%16==0,fresh_rng=fresh_rng,scientific_certification=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='conf/research/bauer_path/training.yaml');p.add_argument('--seed',type=int,choices=[11,22,33],default=11)
    p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--updates',type=int)
    p.add_argument('--fresh-rng',action='store_true',help='Explicit new transport RNG streams, e.g. CPU smoke; not an exact CUDA resume');a=p.parse_args()
    torch.set_num_threads(1);train(a.config,a.seed,a.output,a.device,a.updates,a.fresh_rng)
