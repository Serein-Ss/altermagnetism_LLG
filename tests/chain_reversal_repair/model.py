"""Bauer site-order permutation projection; not magnetic-vector reflection."""
import torch
from tests.continuous_axial_repair.model import AxialGauge,phases

ARMS=('control','reversal')


class ReversalGauge(AxialGauge):
    def __init__(self,arm):
        if arm not in ARMS:raise ValueError('unregistered reversal arm')
        super().__init__('gauge2');self.reversal_arm=arm

    def get_extra_state(self):
        state=super().get_extra_state();state.update(package='chain_reversal_repair',reversal_arm=self.reversal_arm,
            site_permutation='identical-end Bauer open chain reversal; vector components unchanged',
            required_anchor_sites=(0,1,-2,-1) if self.reversal_arm=='reversal' else (0,1))
        return state

    def forward(self,x,tau,initial,cell,frame,temperature,time):
        if self.reversal_arm=='control':return super().forward(x,tau,initial,cell,frame,temperature,time)
        if len(cell.shape)!=1 or cell.basis!=1:raise ValueError('audited Bauer single-basis chain only')
        b=len(x)
        value=super().forward(torch.cat((x,x.flip(2))),torch.cat((tau,tau)),
            torch.cat((initial,initial.flip(1))),cell,frame,temperature,time)
        return (value[:b]+value[b:].flip(2))*.5


def initialize(checkpoint,arm,seed,device='cpu'):
    old=AxialGauge('gauge2');old.load_state_dict(checkpoint['model'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(981000+seed);net=ReversalGauge(arm)
    state=dict(checkpoint['model']);state['_extra_state']=net.get_extra_state();net.load_state_dict(state)
    net=net.to(device)
    return net,torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-5)


@torch.no_grad()
def reversal_probe(net):
    from tests.training_restart_repair.data import load_train
    bank,_=load_train();device=next(net.parameters()).device
    rows=[]
    for i,values in enumerate(bank):
        a=torch.nn.functional.normalize(torch.as_tensor(values[0,0],device=device)[None],dim=-1)
        gen=torch.Generator(device=device).manual_seed(981800+i)
        x=torch.nn.functional.normalize(torch.randn(1,2,a.shape[1],3,device=device,generator=gen),dim=-1);x[:,0]=a
        for t in (.1,.5,.9):
            tau=x.new_tensor([t]);times=x.new_tensor([0.,.1])
            v=net.velocity(x,tau,a,times);vr=net.velocity(x.flip(2),tau,a.flip(1),times).flip(2)
            error=(v-vr).square().mean().sqrt();scale=v.square().mean().sqrt().clamp_min(1e-12)
            rows.append(dict(state=i,tau=t,absolute_rms=float(error),relative_rms=float(error/scale)))
    return rows


@torch.no_grad()
def gauge_probe(net):
    from tests.training_restart_repair.data import load_train
    bank,_=load_train();device=next(net.parameters()).device
    initial=torch.nn.functional.normalize(torch.tensor([b[0,0].tolist() for b in bank],device=device,dtype=next(net.parameters()).dtype),dim=-1)
    _,weight=phases(initial)
    # Smooth change through equal-weight anchors; no argmax/pivot selection.
    gen=torch.Generator(device=device).manual_seed(980800)
    x=torch.nn.functional.normalize(torch.randn(1,2,25,3,device=device,generator=gen,dtype=initial.dtype),dim=-1)
    base=initial.new_zeros(1,25,3);base[...,2]=1
    base[0,0]=initial.new_tensor([.8,0.,.6]);base[0,1]=initial.new_tensor([0.,.8,.6])
    base[0,-2]=initial.new_tensor([.6,.0,.8]);base[0,-1]=initial.new_tensor([.0,.6,.8])
    sensitivity=[]
    for eps in (1e-2,1e-3,1e-4,1e-5):
        a=base.clone();b=base.clone();a[0,1,1]+=eps;b[0,1,1]-=eps
        a=torch.nn.functional.normalize(a,dim=-1);b=torch.nn.functional.normalize(b,dim=-1)
        xa=x.clone();xb=x.clone();xa[:,0]=a;xb[:,0]=b
        va=net.velocity(xa,x.new_tensor([.4]),a,x.new_tensor([0.,.1]));vb=net.velocity(xb,x.new_tensor([.4]),b,x.new_tensor([0.,.1]))
        sensitivity.append(dict(eps=eps,input_norm=float((a-b).norm()),velocity_difference=float((va-vb).norm())))
    return dict(anchor_weight_min=float(weight.min()),selected_total_transverse_min=float(initial[:,:2,:2].square().sum((1,2)).min()),
                smooth_equal_weight_probe=sensitivity,scope='TRAIN anchor coverage and synthetic continuity, not physical-path accuracy or global smoothness certification.')


if __name__=='__main__':
    from pathlib import Path
    from tests.bauer_p2_20260923.common import dump
    torch.set_num_threads(1);here=Path(__file__).resolve().parent;root=here.parents[1];rows=[]
    for i,seed in enumerate((11,22,33)):
        ck=torch.load(root/f'tests/continuous_axial_repair/train/{2*i+1:02d}/checkpoint.pt',map_location='cpu',weights_only=True)
        for arm in ARMS:
            net,_=initialize(ck,arm,seed);net.eval();probe=reversal_probe(net)
            rows.append(dict(seed=seed,arm=arm,maximum_relative_rms=max(x['relative_rms'] for x in probe),rows=probe))
    dump(here/'initial_probe.json',rows)
    print([(r['seed'],r['arm'],r['maximum_relative_rms']) for r in rows])
