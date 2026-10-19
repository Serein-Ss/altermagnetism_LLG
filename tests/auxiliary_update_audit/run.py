import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import cfg,dump
from tests.spatial_covariance_repair.run import verify as previous
from tests.spatial_covariance_repair.objective import loss,sample
from tests.spatial_covariance_repair.model import initialize,ARMS
from tests.deployment_moment_repair.run import banks,augment
from tests.bauer_comprehensive_20260923.run import require_gpu,real_data,next_batch
from tests.branch_coverage_repair.run import pools,parents
from tests.sampling_balance_repair.run import RefillWindows
from tests.temporal_source_repair.source import draw
from tests.transition_block_repair.model import pair_window
from tests.conditional_teacher_repair.target import labels
from tests.axial_symmetry_repair.symmetry import rotate
from .updates import flat,combine,shadow,cosine

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def source(index):return ROOT/f'tests/spatial_covariance_repair/train/{index:02d}'


def prepare():
    previous();assert not (HERE/'source_lock.json').exists();_,hashes=banks()
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    for index in range(9):files.extend(source(index)/name for name in ('checkpoint.pt','complete.json'))
    for p in files:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',hashes);(HERE/'logs').mkdir(exist_ok=True)


def verify():
    previous()
    for path,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path


def run(index,smoke):
    verify();require_gpu('cuda');torch.set_num_threads(1);tick=time.perf_counter()
    ck=torch.load(source(index)/'checkpoint.pt',weights_only=True,map_location='cpu')
    seed=(11,22,33)[index//3];arm=ARMS[index%3]
    assert ck['step']==126144 and ck['task']['seed']==seed and ck['task']['repair']==arm
    net,opt=initialize(ck,arm,seed,'cuda');opt.load_state_dict(ck['optimizer']);net.train()
    params=list(net.parameters());snapshot=[p.detach().clone() for p in params]
    optimizer=opt.state_dict()
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair=np.random.default_rng();pair.bit_generator.state=ck['pair_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    extra=torch.Generator(device='cuda').manual_seed(983200+seed)
    aug=torch.Generator(device='cuda').manual_seed(983100+seed)
    aux=torch.Generator(device='cuda').manual_seed(983400+seed)
    bank,hashes=banks();gpu=[[F.normalize(torch.tensor(a,device='cuda'),dim=-1) for a in role] for role in bank]
    _,new,_=pools(False);pool={r['parent_id']:a for r,a in zip(parents(),new)}
    counts=dict(counts=Counter(),hash=hashlib.sha256());config=cfg()
    data=[RefillWindows(d,pool,refill,True,counts) for d in real_data(config)]
    rows=[];stream=hashlib.sha256();auxhash=hashlib.sha256()
    probes=[(role,sid) for sid in (0,12,24,36) for role in (0,2)]
    if smoke:probes=probes[:1]
    for number,(role,sid) in enumerate(probes):
        whole,_=next_batch(ck['task'],config,data,rng,gs['target'],'cuda');stream.update(whole.detach().cpu().numpy().tobytes())
        starts=np.where(pair.random(len(whole))<.5,0,pair.integers(1,10,len(whole)))
        y=pair_window(whole,torch.as_tensor(starts,device='cuda'));z=draw(y[:,0],11,gs['source'],0.)[:,:2]
        tau=torch.rand(len(y),device='cuda',generator=gs['tau'])
        cut=gs['cut'].get_state();torch.randn(whole.shape,device='cuda',generator=gs['cut'])
        x,v,_=path_interpolate(z,y,tau,torch.Generator(device='cuda').set_state(cut))
        x,v,a=augment(x,v,y[:,0],aug,True);fm=(net.velocity(x,tau,a,y.new_tensor([0.,.1]))[:,1:]-v[:,1:]).square().sum(-1).mean()
        gn=torch.autograd.grad(fm,params,allow_unused=True)
        targets=[];drafts=[];anchors=[]
        # Three distinct TRAIN states x16 branches, like current replay batching.
        for offset in (0,1,2):
            target=gpu[role][(sid+offset)%48]
            selected=torch.randperm(64,device='cuda',generator=extra)[:16];yr=target[selected]
            zr=draw(yr[:,0],2,extra,0.);tr=torch.rand(16,device='cuda',generator=extra)
            xx,_,vv,_=labels(zr,yr,tr,target,extra)
            angles=torch.rand(1,device='cuda',generator=aug).expand(16)*(2*np.pi)
            xx,vv,aa=rotate(xx,angles),rotate(vv,angles),rotate(yr[:,0],angles)
            targets.append(vv);drafts.append(xx);anchors.append(aa)
            if offset==0:replay_tau=tr
            else:replay_tau=torch.cat((replay_tau,tr))
        xr=torch.cat(drafts);vr=torch.cat(targets);aa=torch.cat(anchors)
        replay=(net.velocity(xr,replay_tau,aa,xr.new_tensor([0.,.1]))[:,1:]-vr[:,1:]).square().sum(-1).mean()
        gr=torch.autograd.grad(replay,params,allow_unused=True);base=combine(gn,gr,.25)
        target=gpu[role][sid];initial=target[:1,0].expand(64,-1,-1);latent=draw(initial,2,aux,0.)
        auxhash.update(np.array([role,sid],dtype=np.int64).tobytes());auxhash.update(latent.cpu().numpy().tobytes())
        generated=sample(net,initial,latent,8 if smoke else 128,True)
        local,_=loss(generated,target,False);total,parts=loss(generated,target,True);spatial=total-local
        gl=torch.autograd.grad(local,params,allow_unused=True,retain_graph=True)
        gspt=torch.autograd.grad(spatial,params,allow_unused=True)
        bv,lv,sv=[flat(a,params) for a in (base,gl,gspt)]
        baseline,clip=shadow(params,optimizer,base);variants=[]
        for mode,ga in (('local',gl),('spatial_only',gspt),('local_plus_spatial',combine(gl,gspt,1.))):
            av=flat(ga,params)
            for weight in (.1,1.):
                delta,clipping=shadow(params,optimizer,combine(base,ga,weight))
                variants.append(dict(mode=mode,weight=weight,weighted_gradient_ratio=float(weight*av.norm()/bv.norm().clamp_min(1e-12)),
                    raw_frequency_normalized_ratio=float(weight*av.norm()/bv.norm().clamp_min(1e-12)/64),
                    cosine_base_aux=cosine(bv,av),update_increment_ratio=float((delta-baseline).norm()/baseline.norm().clamp_min(1e-12)),
                    update_cosine=cosine(delta,baseline),first_order_aux_change=float(av.dot(delta)),
                    baseline_first_order_aux_change=float(av.dot(baseline)),**clipping))
        rows.append(dict(role=role,state=sid,native_loss=float(fm),teacher_loss=float(replay),
            local_loss=float(local),spatial_loss=float(spatial),base_norm=float(bv.norm()),
            local_norm=float(lv.norm()),spatial_norm=float(sv.norm()),local_spatial_cosine=cosine(lv,sv),
            baseline_update_norm=float(baseline.norm()),baseline_clipping=clip,variants=variants,loss_parts=parts))
        assert torch.equal(generated[:,0],initial) and float((generated.detach().norm(dim=-1)-1).abs().max())<2e-6
        for p,before in zip(params,snapshot):assert torch.equal(p,before),'shadow mutated source'
    out=HERE/('smoke' if smoke else 'audit')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    torch.cuda.synchronize()
    dump(out/'report.json',dict(index=index,seed=seed,arm=arm,step=ck['step'],rows=rows,
        native_input_hash=stream.hexdigest(),auxiliary_input_hash=auxhash.hexdigest(),seconds=time.perf_counter()-tick,
        device=torch.cuda.get_device_name(),actual_model_updates=0,scientific_certification=False,
        scope='Frozen TRAIN-only optimization diagnostic. Replay here uses three nearby bank IDs per probe, not exact historical full48 schedule. Terminal Adam/lr, not reconstruction of all6000 updates. Raw frequency ratio not equivalent Adam effect; weight1 diagnostic not adopted training. No DEV or final holdout used.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    prepare() if a.action=='prepare' else run(a.index,a.smoke)
