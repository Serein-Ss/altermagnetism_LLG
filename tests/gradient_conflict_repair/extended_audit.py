"""Average frozen TRAIN gradients before diagnosing systematic interference."""
import argparse
import hashlib
import time
import numpy as np
import torch
from torch.nn import functional as F
from .run import (HERE,source,verify,require_gpu,TransitionNet,load_train,pools,train_parents,
                  Counter,cfg,real_data,RefillWindows,next_batch,pair_window,draw,path_interpolate,dump)


def main(index, batches):
    assert batches>=32 and batches<=8192 and batches%32==0
    verify();require_gpu('cuda');seed=(11,22,33)[index]
    ck=torch.load(source(seed)/'checkpoint.pt',map_location='cpu',weights_only=True)
    net=TransitionNet().cuda();net.load_state_dict(ck['model']);net.eval()
    parameters=list(net.parameters());before=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in parameters)).hexdigest()
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair=np.random.default_rng();pair.bit_generator.state=ck['pair_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    extra_rng=np.random.default_rng(947000+seed);extra_g=torch.Generator(device='cuda').manual_seed(947100+seed)
    bank,_=load_train();_,new,_=pools(False);pool={r['parent_id']:x for r,x in zip(train_parents(),new)}
    config=cfg();audit=dict(counts=Counter(),hash=hashlib.sha256())
    data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    def gradient(y,z,tau,generator):
        x,v,_=path_interpolate(z,y,tau,generator)
        pred=net.velocity(x,tau,y[:,0],y.new_tensor([0.,.1]))
        loss=(pred[:,1:]-v[:,1:]).square().sum(-1).mean()
        gg=torch.autograd.grad(loss,parameters,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(parameters,gg)])
    accumulated=[[],[]];blocks=[];tick=time.perf_counter()
    # Store only sixteen-batch gradient sums, not all per-batch vectors.
    sums=[None,None];squares=[0.,0.]
    for step in range(1,batches+1):
        whole,_=next_batch(dict(domain='bauer',arm='hierarchical',seed=seed),config,data,rng,gs['target'],'cuda')
        starts=np.where(pair.random(len(whole))<.5,0,pair.integers(1,10,len(whole)))
        y=pair_window(whole,torch.as_tensor(starts,device='cuda'))
        z=draw(y[:,0],11,gs['source'],0.)[:,:2];tau=torch.rand(len(y),device='cuda',generator=gs['tau'])
        cut=gs['cut'].get_state();torch.randn(whole.shape,device='cuda',generator=gs['cut'])
        native=gradient(y,z,tau,torch.Generator(device='cuda').set_state(cut))
        ids=extra_rng.permutation(48);branches=extra_rng.integers(64,size=48)
        yr=F.normalize(torch.tensor(np.stack([bank[i][j] for i,j in zip(ids,branches)]),device='cuda'),dim=-1)
        zr=draw(yr[:,0],2,extra_g,0.);tr=torch.rand(48,device='cuda',generator=extra_g)
        replay=gradient(yr,zr,tr,extra_g)
        for k,g in enumerate((native,replay)):
            sums[k]=g.clone() if sums[k] is None else sums[k]+g
            squares[k]+=float(g.square().sum())
        if step%16==0:
            for k in (0,1):accumulated[k].append(sums[k].cpu().double());sums[k]=None
    def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30))
    nblocks=batches//16
    widths=[w for w in (1,4,16,64,256,512) if w<=nblocks and nblocks%w==0]
    if nblocks not in widths:widths.append(nblocks)
    for width in widths:
        for start in range(0,nblocks,width):
            a=sum(accumulated[0][start:start+width])/(width*16)
            b=sum(accumulated[1][start:start+width])/(width*16)
            blocks.append(dict(batches=width*16,start_batch=start*16,cosine=cosine(a,b),native_norm=float(a.norm()),replay_norm=float(b.norm())))
    same={str(k):[cosine(sum(accumulated[k][:nblocks//2]),sum(accumulated[k][nblocks//2:]))] for k in (0,1)}
    after=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in parameters)).hexdigest();assert before==after
    result=dict(seed=seed,batches=batches,code_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),frozen_weights=True,model_hash=before,blocks=blocks,same_objective_half_cosine=same,
                rms_individual_gradient=[float(np.sqrt(v/batches)) for v in squares],seconds=time.perf_counter()-tick,
                interpretation='TRAIN-only diagnostic. Block means correlated within each checkpoint; no physical certification, formal pvalues or independent-parent claims.')
    out=HERE/('extended_audit_smoke' if batches<8192 else 'extended_audit')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False);dump(out/'summary.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0);p.add_argument('--batches',type=int,default=8192);a=p.parse_args();main(a.index,a.batches)
