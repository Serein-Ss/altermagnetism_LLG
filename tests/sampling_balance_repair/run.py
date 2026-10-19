"""Preserve the original parent/size/window sampler; expand eligible futures only."""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from tests.bauer_p2_20260923.common import cfg, dump
from tests.bauer_comprehensive_20260923.run import real_data, next_batch, require_gpu
from tests.bauer_closed_loop_20260919.run import batch
from tests.mdgen_path_refactor_20260918.data import PathWindows
from tests.condition_time_repair.run import build, expanded
from tests.branch_coverage_repair.run import checkpoint, pools, parents, seen
from tests.curriculum_repair.run import fm_loss
from tests.moment_bias_repair.run import evaluate

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def tasks():
    # Adjacent pairs share the same seed; submission can run both arms together.
    return [dict(domain='bauer',arm='hierarchical',seed=s,repair=r,updates=6000)
            for s in (22,11,33) for r in ('native_old','native_expanded')]


class RefillWindows:
    """Wrap the original index contract without changing any sampler probabilities."""
    def __init__(self,base,pool,rng,enabled,audit):
        self.base=base;self.pool=pool;self.rng=rng;self.enabled=enabled;self.audit=audit
        self.records=base.records;self.ends=base.ends

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        item=self.base[index]
        g=int(np.searchsorted(self.ends,index,side='right'))
        offset=index-(self.ends[g-1] if g else 0)
        _,row,starts,_=self.records[g];branch,window=divmod(offset,len(starts));start=starts[window]
        parent=row['parent_id'];eligible=start==0 and parent in self.pool
        # Record original selections before replacing a branch; traces must match across arms.
        trace=[parent,branch,start,eligible]
        self.audit['hash'].update(json.dumps(trace).encode())
        counts=self.audit['counts'];counts['total']+=1;counts[f'L{item["spins"].shape[-2]}']+=1
        counts[f'parent:{parent}']+=1;counts[f'family:{row["family"]}']+=1
        counts['anchor' if start==0 else 'later_window']+=1
        x=item['spins']
        if eligible:
            counts['eligible']+=1;k=int(self.rng.integers(len(self.pool[parent])))
            if self.enabled:
                x=self.pool[parent][k].copy();counts['new_branch' if k>=8 else 'old_pool_branch']+=1
        # This adapter is for batch(), whose sole consumed field is spins.
        return dict(spins=x)


class MetadataWindows(PathWindows):
    """Audit selections cheaply: no trajectory data read in __getitem__."""
    def __getitem__(self,index):
        g=int(np.searchsorted(self.ends,index,side='right'));row=self.records[g][1]
        size=int(row['parent_id'].split('_')[0][1:])
        return dict(spins=np.broadcast_to(np.array([0.,0.,1.],np.float32),(11,size,3)))


def sampling_audit():
    config=cfg();data=[MetadataWindows([ROOT/config['data']/f'L{n}'],'train',1.,stride_frames=10) for n in config['sizes']]
    inventory=[dict(L=n,parents=len(d.records),windows_per_branch=sorted({len(r[2]) for r in d.records})) for n,d in zip(config['sizes'],data)]
    dummy={r['parent_id']:np.zeros((72,11,16,3),np.float32) for r in parents()};rows=[]
    for seed in (11,22,33):
        ck=torch.load(checkpoint(seed),map_location='cpu',weights_only=True)
        rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng'];audit=dict(counts=Counter(),hash=hashlib.sha256())
        refill=np.random.default_rng(seed+935000);wrapped=[RefillWindows(d,dummy,refill,False,audit) for d in data]
        for _ in range(6000):batch(wrapped,rng,config,'cpu','B')
        counts=dict(audit['counts']);total=counts['total']
        rows.append(dict(seed=seed,counts=counts,trace_hash=audit['hash'].hexdigest(),fractions={k:counts.get(k,0)/total for k in ('L16','L25','L32','anchor','eligible')}))
    return dict(inventory=inventory,native=rows,theoretical_native=dict(L16=1/3,L25=1/3,L32=1/3,anchor=.505,eligible=.505/6),
        theoretical_previous=dict(L16=2/3,L25=1/6,L32=1/6,anchor=.7525,eligible=.5+.505/12),
        interpretation='Previous campaign deliberately oversampled 24 L16 anchors. Native wrapper restores original size/parent/window probabilities. Numerical degradation causality remains to be tested.')


def prepare():
    assert not (HERE/'source_lock.json').exists(),'Already frozen'
    torch.set_num_threads(1)
    dump(HERE/'sampling_audit.json',sampling_audit())
    dump(HERE/'protocol.json',dict(tasks=tasks(),
        scope='Bauer open L16/25/32, J50meV,kappa0.1,alpha0.1,theta0.11,zero field. No new data, no architecture/physics/loss changes. Supported1ps/.1ps grid.',
        design='Two concurrent native-sampling arms x three seeds, same control step24000 weights and Adam,6000 FM-only cosine3e-4->3e-6 updates. Native_old is bitwise identical in input selections to original batch(). Native_expanded changes only eligible original t0 windows of the24 L16 parents to uniform draws from existing72 branches. Original size,parent,family,origin-window frequencies and all other windows unchanged. Independent refill RNG, drawn in both arms.',
        caveat='Restoring original probabilities reduces expanded-branch exposures versus previous targeted experiment; dose is not matched. Cross-campaign targeted comparisons are contextual, not concurrent causal controls. No unseen L25/L32 branches fabricated. No claim that original weighting is universally optimal.',
        evaluation='Frozen before metrics reused from branch_coverage_repair at step24000. After:24seen parents128generated,36independent dev64generated,18expanded dev256generated. Euler128 and midpoint128 as before. Seen-parent unseen-noise references remain diagnosis-only; train/probe files kept separate. Whole-branch bootstrap100; report seeds and sizes separately, no composite acceptance score.',
        decision='Improvement must retain seen-parent temporal benefit and not degrade L25/L32 relative to native-old and pre-repair. Tradeoffs reported rather than automatic certification; reused dev is not final holdout. No arbitrary new tolerance.',
        budget='6rtx4090 jobs15-30min each;fat report25-60min. Smoke paired arms1-3min; no long monitoring.',
        skills='Karpathy minimal isolated sampler adapter; experimental-design concurrent paired controls. Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/branch_coverage_repair/source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',HERE/'sampling_audit.json']
    files+=list((ROOT/'tests/branch_coverage_repair/reference').glob('*/*.h5'))
    files+=[ROOT/'tests/condition_time_repair/assets/summary.json']
    for f in files:lock[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def train(index,smoke):
    require_gpu('cuda');task=tasks()[index];net,opt,_=build(dict(task,repair='control'))
    ck=torch.load(checkpoint(task['seed']),map_location='cpu',weights_only=True)
    assert ck['step']==24000;net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng(935000+task['seed'])
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg()
    data=[RefillWindows(d,pool,refill,task['repair']=='native_expanded',audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    steps=32 if smoke else task['updates'];history=[];noise=hashlib.sha256();start=time.perf_counter();net.train()
    for step in range(1,steps+1):
        y,t=next_batch(task,config,data,rng,gs['target'],'cuda')
        for k in ('source','tau'):noise.update(gs[k].get_state().numpy().tobytes())
        lr=3e-6+(.0003-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,steps-1)))
        for group in opt.param_groups:group['lr']=lr
        loss,_=fm_loss(net,y,t,gs);opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(loss) and torch.isfinite(grad);opt.step()
        if smoke or step%100==0:history.append(dict(step=24000+step,loss=float(loss.detach()),lr=lr))
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=24000+steps,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
        auxiliary_rng=ck['auxiliary_rng']),out/'checkpoint.pt')
    dump(out/'history.json',history);net.eval();seen(net,out,smoke);ev=evaluate(net,out,[],smoke);extra=expanded(net,out,smoke)
    dump(out/'complete.json',dict(task=task,train_seconds=elapsed,evaluation=ev,expanded=extra,
        data_hashes=hashes,counts=dict(audit['counts']),schedule_hash=audit['hash'].hexdigest(),source_tau_hash=noise.hexdigest(),
        device=torch.cuda.get_device_name(),scientific_certification=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','smoke']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    train(a.index,a.action=='smoke')


if __name__=='__main__':main()
