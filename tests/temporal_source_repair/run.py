"""Paired source-law repair on the current backbone; all historical files frozen."""
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
from tests.spin_path_transformer.run import make_net
from tests.sampling_balance_repair.run import RefillWindows
from tests.branch_coverage_repair.run import pools,parents as train_parents
from tests.reference_resolution.run import parents,reference_inputs
from tests.bauer_comprehensive_20260923.run import require_gpu,real_data,next_batch
from tests.bauer_p2_20260923.common import cfg,dump
from tests.transport_audit.run import integrate
from .source import draw,contract,check_package

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def tasks():
    return [dict(domain='bauer',arm='hierarchical',seed=s,rho=r,updates=12000) for s in (11,22,33) for r in (0.,.8)]


def prepare():
    assert not (HERE/'source_lock.json').exists(),'Already frozen'
    dump(HERE/'protocol.json',dict(tasks=tasks(),
        hypothesis='Test whether stationary temporal correlation of source futures reduces transport learning difficulty and early physical increment variance bias. Not assumed correct or a diagnosis already established.',
        historical_audit='empirical_transport_repair/extended_summary.json already contains three-seed matched empirical8 CFM versus exact conditional velocity with no uniform advantage. Do not repeat. Earlier bauer_repair_20260924 used global shared rotational source on older backbone; this is per-site decaying temporal correlation on current SpinPathTransformer, not first-ever correlated-source claim.',
        source='eta1=eps1; eta_t=rho*eta_(t-1)+sqrt(1-rho^2)*eps_t for future frames2..10; sites independent. z=normalize(2*S0+eta). rho0 control vs fixed rho.8 candidate, same single-frame marginal. Full joint support retained. Correlation is latent regularization, NOT LLG noise/time dynamics. Independent-frame oracle density must NOT be reused for rho.8.',
        physics='Bauer open L16/25/32; J50meV,kappa.1,alpha.1,theta.11,T63.824849668K,zero field.1ps,11frames,.1ps saved grid. No new LLG data or boundary/physical-time changes.',
        training='Both arms fresh, identical current hierarchical width64/depth4 initialization per seed, original native parent/size/window sampler plus existing72-branch refill for eligible24L16 train parents. FM-only,12000updates cosine1e-3->3e-5, batch8. Same target/raw-noise/tau random streams. No checkpoint source substitution. Same updates not equal measured compute.',
        stages='6000 and12000; final12000 primary,6000 descriptive. Save model/Adam/RNG/source contract. Evaluation does not consume training RNG. No adaptive stop or checkpoint selection.',
        evaluation='18independent dev parents128paths, two samplers midpoint128/512 from same latent; compare both independent base128/fine128.24seen train parents64new latent paths against probe128 independent futures at midpoint128. Preserve every generated path, raw latent seeds and times. Per-transport-tau velocity training diagnostics on fixed held-out random draws; MSE targets differ across source laws, not directly comparable physical accuracy.',
        uncertainty='Whole-branch bootstrap100 including independent reference variation. Reused dev not final test. Report all3seeds/sizes; reference-reference spread not tolerance. No automatic promotion or equivalence. No off-policy true velocity oracle for correlated source.',
        budget='6rtx4090 jobs estimated20-45min each;6fat stats10-25min each; aggregate<3min. Paired GPU smoke<5min. Formal submission then no monitoring.',
        skill_use='Karpathy isolated source change; experimental-design paired controls; Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/transport_path_repair/source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    for f in files:lock[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


@torch.no_grad()
def evaluate(net,folder,task,smoke):
    records=[];times=torch.arange(11,device='cuda')*.1;velocity=[]
    with h5py.File(folder/'paths.h5','x') as h:
        h.attrs['complete']=False;h.attrs['source_contract']=json.dumps(contract(task['rho']))
        cases=[]
        for i,row in enumerate(parents()[:1] if smoke else parents()):
            path=ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5'
            cases.append(('dev',i,row,path,'base/spins'))
        for i,row in enumerate(train_parents()[:1] if smoke else train_parents()):
            cases.append(('seen',i,row,ROOT/f'tests/branch_coverage_repair/reference/{i:02d}/probe.h5','spins'))
        for role,i,row,path,key in cases:
            with h5py.File(path) as ref:
                assert ref.attrs['complete'];raw=ref[key][:]
            n=4 if smoke else (128 if role=='dev' else 64)
            a=F.normalize(torch.tensor(raw[:1,0],device='cuda',dtype=torch.float32),dim=-1).expand(n,-1,-1)
            seed=938100+i+(1000 if role=='seen' else 0)
            z=draw(a,11,torch.Generator(device='cuda').manual_seed(seed),task['rho'])
            g=h.create_group(f'{role}/parent{i:02d}');g.attrs['parent']=json.dumps(row);g.attrs['latent_seed']=seed
            g.create_dataset('time_ps',data=np.arange(11)*.1);g.create_dataset('latent',data=z.cpu().numpy(),compression='lzf')
            levels=(8,16) if smoke and role=='dev' else ((8,) if smoke else ((128,512) if role=='dev' else (128,)))
            for steps in levels:
                torch.cuda.synchronize();start=time.perf_counter();xs=[]
                for b in range(0,n,64):
                    init=a[b:b+64]
                    xs.append(integrate(lambda x,t:net.velocity(x,t,init,times),z[b:b+64],0.,1.,steps,'midpoint'))
                x=torch.cat(xs);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                norm=float(abs(x.norm(dim=-1)-1).max());anchor=float(abs(x[:,0]-a).max())
                assert torch.isfinite(x).all() and norm<2e-6 and anchor==0
                g.create_dataset(f'midpoint{steps}',data=x.cpu().numpy(),compression='lzf')
                records.append(dict(role=role,parent=i,steps=steps,seconds=elapsed,norm_error=norm,anchor_error=anchor))
            y=F.normalize(torch.tensor(raw[:n],device='cuda',dtype=torch.float32),dim=-1);y[:,0]=a
            for tau in (.05,.25,.5,.75,.9):
                tt=z.new_full((n,),tau);vstate,target,cut=path_interpolate(z,y,tt,torch.Generator(device='cuda').manual_seed(938999+i))
                pred=net.velocity(vstate,tt,a,times)
                velocity.append(dict(role=role,parent=i,tau=tau,sampled_target_mse=float((pred[:,1:]-target[:,1:]).square().sum(-1).mean()),
                                     target_power=float(target[:,1:].square().sum(-1).mean()),cut=cut))
        h.attrs['complete']=True
    dump(folder/'evaluation.json',dict(records=records,velocity=velocity,source_contract=contract(task['rho'])))


def train(index,smoke):
    require_gpu('cuda');task=tasks()[index];net=make_net(task);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    rng=np.random.default_rng(task['seed']);refill=np.random.default_rng(938000+task['seed'])
    gs={k:torch.Generator(device='cuda').manual_seed(base+task['seed']) for k,base in [('source',938200),('tau',938300),('target',938400),('cut',938500)]}
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(train_parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg();data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    total=32 if smoke else task['updates'];stages=(32,) if smoke else (6000,12000)
    history=[];stream=hashlib.sha256();initial_hash=hashlib.sha256()
    for p in net.parameters():initial_hash.update(p.detach().cpu().numpy().tobytes())
    start=time.perf_counter();evaluation_seconds=0.;net.train()
    for step in range(1,total+1):
        y,t=next_batch(task,config,data,rng,gs['target'],'cuda')
        for k in ('source','tau'):stream.update(gs[k].get_state().numpy().tobytes())
        stream.update(y.detach().cpu().numpy().tobytes())
        z=draw(y[:,0],11,gs['source'],task['rho']);tau=torch.rand(len(y),device='cuda',generator=gs['tau'])
        x,v,cut=path_interpolate(z,y,tau,gs['cut']);pred=net.velocity(x,tau,y[:,0],t)
        loss=(pred[:,1:]-v[:,1:]).square().sum(-1).mean();lr=3e-5+(.001-3e-5)*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        assert torch.isfinite(loss) and torch.isfinite(grad);opt.step()
        if smoke or step%100==0:history.append(dict(step=step,loss=float(loss.detach()),gradient=float(grad),lr=lr,cut=cut))
        if step%1000==0:print(task,step,flush=True)
        if step in stages:
            folder=out/f'step{step:05d}';folder.mkdir()
            state=dict(model=net.state_dict(),optimizer=opt.state_dict(),source_contract=contract(task['rho']),task=task,step=step,
                       rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
                       torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            torch.save(state,folder/'checkpoint.pt')
            # Reload proves inference uses the serialized source contract and actual checkpoint.
            saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True);check_package(saved,task['rho']);net.load_state_dict(saved['model'])
            torch.cuda.synchronize();tick=time.perf_counter()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):net.eval();evaluate(net,folder,task,smoke)
            net.train();torch.cuda.synchronize();evaluation_seconds+=time.perf_counter()-tick
            for k,g in gs.items():assert torch.equal(g.get_state(),state['rng'][k])
            assert torch.equal(torch.get_rng_state(),state['torch_rng']) and torch.equal(torch.cuda.get_rng_state(),state['cuda_rng'])
    torch.cuda.synchronize()
    dump(out/'history.json',history)
    dump(out/'complete.json',dict(task=task,stages=stages,train_seconds=time.perf_counter()-start-evaluation_seconds,
         evaluation_seconds=evaluation_seconds,initial_hash=initial_hash.hexdigest(),paired_input_hash=stream.hexdigest(),schedule_hash=audit['hash'].hexdigest(),
         counts=dict(audit['counts']),data_hashes=hashes,device=torch.cuda.get_device_name(),scientific_certification=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    train(a.index,a.action=='smoke')


if __name__=='__main__':main()
