"""Bauer-only moving-frame SO(2) candidate, not fixed C_K group quadrature."""
import torch
from tests.transition_block_repair.model import TransitionNet
from tests.spatial_gate_repair.model import GatedTransition
from tests.axial_symmetry_repair.symmetry import rotate

ARMS=('control','gauge2')


def phases(initial):
    a=initial.flatten(1,-2)[:,:2]
    if a.shape[1]!=2:raise ValueError('two fixed boundary anchor sites required')
    weights=a[...,:2].square().sum(-1);total=weights.sum(-1,keepdim=True)
    if torch.any(total<=1e-12):raise ValueError('unsupported initial state: both anchor sites are too close to easy-axis poles')
    return torch.atan2(a[...,1],a[...,0]),weights/total


class AxialGauge(TransitionNet):
    def __init__(self,arm):
        if arm not in ARMS:raise ValueError('unregistered axial candidate')
        super().__init__();self.gauge_arm=arm

    def get_extra_state(self):
        state=super().get_extra_state();state.update(package='continuous_axial_repair',gauge_arm=self.gauge_arm,
            gauge_sites=(0,1),minimum_transverse_weight=1e-12,
            additional_symmetry='continuous spin-only SO(2) around Bauer easy axis' if self.gauge_arm=='gauge2' else 'none')
        return state

    def forward(self,x,tau,initial,cell,frame,temperature,time):
        if self.gauge_arm=='control':return super().forward(x,tau,initial,cell,frame,temperature,time)
        if len(cell.shape)!=1 or cell.basis!=1:raise ValueError('audited Bauer single-basis chain only')
        body=x@frame.T;anchor=initial@frame.T;angle,weights=phases(anchor);b=len(x)
        def repeat(a):return a[:,None].expand(b,2,*a.shape[1:]).reshape(b*2,*a.shape[1:])
        xx=rotate(repeat(body),-angle.flatten());aa=rotate(repeat(anchor),-angle.flatten())
        value=super().forward(xx,repeat(tau),aa,cell,torch.eye(3,device=x.device,dtype=x.dtype),temperature,time)
        value=rotate(value,angle.flatten()).reshape(b,2,*x.shape[1:])
        value=(value*weights.reshape(b,2,*([1]*(x.ndim-1)))).sum(1)
        return value@frame


def initialize(checkpoint,arm,seed,device='cpu'):
    old=GatedTransition(1.);old.load_state_dict(checkpoint['model'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(980000+seed);net=AxialGauge(arm)
    state=dict(checkpoint['model']);state['_extra_state']=net.get_extra_state()
    net.load_state_dict(state);net=net.to(device)
    return net,torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-5)


@torch.no_grad()
def gauge_probe(net):
    from tests.training_restart_repair.data import load_train
    bank,_=load_train();device=next(net.parameters()).device
    initial=torch.nn.functional.normalize(torch.tensor([b[0,0].tolist() for b in bank],device=device),dim=-1)
    _,weight=phases(initial)
    # Smooth change through equal-weight anchors; no argmax/pivot selection.
    gen=torch.Generator(device=device).manual_seed(980800)
    x=torch.nn.functional.normalize(torch.randn(1,2,25,3,device=device,generator=gen),dim=-1)
    base=initial.new_zeros(1,25,3);base[...,2]=1
    base[0,0]=initial.new_tensor([.8,0.,.6]);base[0,1]=initial.new_tensor([0.,.8,.6])
    sensitivity=[]
    for eps in (1e-2,1e-3,1e-4,1e-5):
        a=base.clone();b=base.clone();a[0,1,1]+=eps;b[0,1,1]-=eps
        a=torch.nn.functional.normalize(a,dim=-1);b=torch.nn.functional.normalize(b,dim=-1)
        xa=x.clone();xb=x.clone();xa[:,0]=a;xb[:,0]=b
        va=net.velocity(xa,x.new_tensor([.4]),a,x.new_tensor([0.,.1]));vb=net.velocity(xb,x.new_tensor([.4]),b,x.new_tensor([0.,.1]))
        sensitivity.append(dict(eps=eps,input_norm=float((a-b).norm()),velocity_difference=float((va-vb).norm())))
    return dict(anchor_weight_min=float(weight.min()),selected_total_transverse_min=float(initial[:,:2,:2].square().sum((1,2)).min()),
                smooth_equal_weight_probe=sensitivity,scope='TRAIN anchor coverage and synthetic continuity, not physical-path accuracy or global smoothness certification.')
