"""Two-frame training versus frozen matched-budget joint-path control."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from collections import Counter
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.sampling_balance_repair.run import RefillWindows
from tests.branch_coverage_repair.run import pools,parents as train_parents
from tests.reference_resolution.run import parents
from tests.bauer_comprehensive_20260923.run import require_gpu,real_data,next_batch
from tests.bauer_p2_20260923.common import cfg,dump
from tests.temporal_source_repair.source import draw
from .model import make_transition,pair_window,rollout

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def tasks():
    return [dict(domain='bauer',arm='hierarchical',seed=s,repair=r,updates=12000) for s in (11,22,33) for r in ('prefix','adjacent')]


def prepare():
    assert not (HERE/'source_lock.json').exists(),'Already frozen'
    dump(HERE/'protocol.json',dict(tasks=tasks(),
        hypothesis='Separate first0.1ps conditional fit from1ps path organization and accumulated rollout bias. Shorter trained window alone is not physical certification.',
        history='Earlier conditional_restart and bauer_teacher_state_20260920 restart old models at true states; they are not current-backbone two-frame training. Current-backbone matched experiment not found. Reuse temporal_source_repair rho0 joint11 controls, final12000, rather than retrain them.',
        package='New explicit Bauer two-frame package [0,.1]ps using unchanged hierarchical backbone. Internal variable-length software engine is guarded by strict physical grid and temperature/boundary checks. Serialized metadata rejects historical11-frame checkpoints; initialization copies weights only, never relabels trained checkpoints.',
        physics='Unchanged Bauer open L16/25/32;J50meV,kappa.1,alpha.1,theta.11,T63.824849668K,zero field. Reference base dt.25fs/fine.125fs; saved.1ps. No new physical data.',
        training='Same native windows and72branch-refill as temporal_source_repair rho0. From scratch identical shared parameters, AdamW cosine.001->.00003,12000updates,batch8. prefix takes original first2frames. adjacent selects j=0 with probability.5, otherwise uniform1..9, using separate RNG. Retains origin parent splits; correlated windows not independent test units. Candidate intentionally changes conditioning-state coverage and reduces exposure to original anchors.',
        pairing='Target whole11-frame windows, source-generator state and tau state hashes must match both new arms and frozen joint11 rho0 controls. Consume11-frame Gaussian source and full11-frame cut-noise RNG before slicing first2source frames; target-pair selection uses independent RNG. Source marginal unchanged; no future labels used in rollout. Same updates are NOT equal FLOPs or supervised frame counts.',
        evaluation='Final12000 fixed primary;6000checkpoint saved without adaptive selection.18dev parents128paths and24seen parents64paths. New models generate10 consecutive.1ps steps with fresh RNG, deduplicated boundaries; midpoint128/512 per transition for dev,128 for seen. Joint control uses existing matching generated counts/seed/refs. Report first0.1ps separately using new two-frame mean,covariance,variance,energy,spatial statistics; full1ps via existing evaluator. Save complete paths,RNG state and timings. No one-to-one LLG noise matching.',
        uncertainty='Whole-trajectory bootstrap100 per fixed parent with both reference and model sampling variation. No lag2/5 computed from2frames. Reused dev, not final holdout. Seen probe futures not used for training. No assumed accuracy, speedup or equivalent compute. No automatic promotion.',
        decisions='If first0.1ps fails, joint-block complexity alone is not sufficient explanation. If first step improves but1ps rollout degrades, investigate accumulated bias/conditioning-state coverage. prefix vs adjacent isolates a specific state-sampling intervention, not all coverage. Compare allseed andsizes, disclose tradeoffs.',
        budget='6rtx4090 training/evaluation jobs estimated40-90min each;6fat stats10-25min each;aggregate<3min. Smoke<5min. Submit formal then no monitoring.',
        skill_use='Karpathy isolation; experimental-design paired controls and explicit historical comparator limitations; Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/temporal_source_repair/source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    for i in (0,2,4):
        files += [ROOT/f'tests/temporal_source_repair/train/{i:02d}/complete.json',ROOT/f'tests/temporal_source_repair/train/{i:02d}/step12000/paths.h5']
    for f in files:lock[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


@torch.no_grad()
def evaluate(net,out,smoke):
    records=[]
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False;h.attrs['model_contract']=json.dumps(net.get_extra_state())
        for role,rows in (('dev',parents()),('seen',train_parents())):
            for i,row in enumerate(rows[:1] if smoke else rows):
                path=ROOT/(f'tests/reference_resolution/reference/{i:02d}/paths.h5' if role=='dev' else f'tests/branch_coverage_repair/reference/{i:02d}/probe.h5')
                with h5py.File(path) as ref:
                    assert ref.attrs['complete'];initial=ref['base/spins'][0,0] if role=='dev' else ref['spins'][0,0]
                n=4 if smoke else (128 if role=='dev' else 64)
                a=F.normalize(torch.tensor(initial,device='cuda',dtype=torch.float32),dim=-1)[None].expand(n,-1,-1)
                g=h.create_group(f'{role}/parent{i:02d}');g.attrs['parent']=json.dumps(row)
                g.create_dataset('time_ps',data=np.arange(11)*.1)
                levels=((4,8) if role=='dev' else (4,)) if smoke else ((128,512) if role=='dev' else (128,))
                for steps in levels:
                    pieces=[];rngstates=[];torch.cuda.synchronize();tick=time.perf_counter()
                    for b in range(0,n,64):
                        gen=torch.Generator(device='cuda').manual_seed(939100+i+(1000 if role=='seen' else 0)+b*100)
                        pieces.append(rollout(net,a[b:b+64],gen,steps).cpu().numpy());rngstates.append(gen.get_state().cpu().numpy())
                    torch.cuda.synchronize();elapsed=time.perf_counter()-tick;x=np.concatenate(pieces)
                    norm=float(abs(np.linalg.norm(x,axis=-1)-1).max());anchor=float(abs(x[:,0]-a.cpu().numpy()).max())
                    assert x.shape[1]==11 and np.isfinite(x).all() and norm<2e-6 and anchor==0
                    g.create_dataset(f'midpoint{steps}',data=x,compression='lzf')
                    g.create_dataset(f'rng_final{steps}',data=np.stack(rngstates))
                    records.append(dict(role=role,parent=i,steps_per_transition=steps,seconds=elapsed,network_evaluations_per_path=20*steps,norm_error=norm,anchor_error=anchor))
        h.attrs['complete']=True
    return records


def train(index,smoke):
    require_gpu('cuda');task=tasks()[index];net=make_transition(task);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    rng=np.random.default_rng(task['seed']);refill=np.random.default_rng(938000+task['seed']);pair_rng=np.random.default_rng(939000+task['seed'])
    gs={k:torch.Generator(device='cuda').manual_seed(base+task['seed']) for k,base in [('source',938200),('tau',938300),('target',938400),('cut',938500)]}
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(train_parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg();data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    total=32 if smoke else 12000;history=[];stream=hashlib.sha256();initial_hash=hashlib.sha256();pair_counts=Counter()
    for p in net.parameters():initial_hash.update(p.detach().cpu().numpy().tobytes())
    tick=time.perf_counter();net.train()
    for step in range(1,total+1):
        whole,_=next_batch(task,config,data,rng,gs['target'],'cuda')
        for k in ('source','tau'):stream.update(gs[k].get_state().numpy().tobytes())
        stream.update(whole.detach().cpu().numpy().tobytes())
        selected=np.where(pair_rng.random(len(whole))<.5,0,pair_rng.integers(1,10,len(whole)))
        if task['repair']=='prefix':selected[:]=0
        pair_counts.update(map(int,selected));y=pair_window(whole,torch.as_tensor(selected,device='cuda'))
        z=draw(y[:,0],11,gs['source'],0.)[:,:2];tau=torch.rand(len(y),device='cuda',generator=gs['tau'])
        # Consume full-sized interpolation noise exactly like the frozen joint control.
        cut_state=gs['cut'].get_state();torch.randn(whole.shape,device='cuda',generator=gs['cut'])
        temp=torch.Generator(device='cuda').set_state(cut_state);x,v,cut=path_interpolate(z,y,tau,temp)
        pred=net.velocity(x,tau,y[:,0],y.new_tensor([0.,.1]));loss=(pred[:,1:]-v[:,1:]).square().sum(-1).mean()
        lr=3e-5+(.001-3e-5)*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        assert torch.isfinite(loss) and torch.isfinite(grad);opt.step()
        if smoke or step%100==0:history.append(dict(step=step,loss=float(loss.detach()),gradient=float(grad),lr=lr,cut=cut))
        if step%1000==0:print(task,step,flush=True)
        if step in (6000,total):
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=step,
                rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
                pair_rng=pair_rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/f'step{step}.pt')
    torch.cuda.synchronize();training=time.perf_counter()-tick
    ck=torch.load(out/f'step{total}.pt',map_location='cpu',weights_only=True);net.load_state_dict(ck['model']);net.eval()
    tick=time.perf_counter();ev=evaluate(net,out,smoke);torch.cuda.synchronize()
    dump(out/'history.json',history)
    dump(out/'complete.json',dict(task=task,step=total,train_seconds=training,evaluation_seconds=time.perf_counter()-tick,
        initial_hash=initial_hash.hexdigest(),paired_input_hash=stream.hexdigest(),schedule_hash=audit['hash'].hexdigest(),
        pair_counts=dict(pair_counts),counts=dict(audit['counts']),data_hashes=hashes,evaluation=ev,device=torch.cuda.get_device_name(),scientific_certification=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    train(a.index,a.action=='smoke')


if __name__=='__main__':main()
