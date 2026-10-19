"""Software-validated Bauer candidate; physical path accuracy NOT certified.
Legacy extra_state strings are retained for exact existing-checkpoint loading.
"""
import torch
from scripts.model.spin_path_transformer import SpinPathTransformer

def rotate(x,angle):
    angle=torch.as_tensor(angle,device=x.device,dtype=x.dtype)
    if angle.ndim==1:
        if len(angle)!=len(x):raise ValueError('one angle per path')
        angle=angle.reshape(len(x),*([1]*(x.ndim-2)))
    elif angle.ndim!=0:raise ValueError('scalar or batch angles')
    c,s=angle.cos(),angle.sin()
    return torch.stack((c*x[...,0]-s*x[...,1],s*x[...,0]+c*x[...,1],x[...,2]),-1)

class TransitionNet(SpinPathTransformer):
    def __init__(self):
        # 'software' disables only the old fixed-11-frame guard. Our stricter
        # two-frame physical guard below is mandatory on every forward call.
        super().__init__('software','hierarchical',source_shift=2.,periodic=(False,))
        self.temperature_K=63.824849668

    def get_extra_state(self):
        state=super().get_extra_state()
        state.update(domain='bauer',version=2,package='transition_block_repair',frames=2,
                     frame_interval=.1,block_ps=.1,time_unit='ps',source_shift=2.,
                     validated_horizon_ps=None)
        return state

    def forward(self,x,tau,initial,cell,frame,temperature,time):
        if x.shape[1]!=2 or len(time)!=2 or not torch.allclose(time,x.new_tensor([0.,.1]),atol=1e-7,rtol=0):
            raise ValueError('Bauer transition package requires [0,0.1] ps')
        return super().forward(x,tau,initial,cell,frame,temperature,time)


def phases(initial):
    a=initial.flatten(1,-2)[:,:2]
    if a.shape[1]!=2:raise ValueError('two fixed boundary anchor sites required')
    weights=a[...,:2].square().sum(-1);total=weights.sum(-1,keepdim=True)
    if torch.any(total<=1e-12):raise ValueError('unsupported initial state: both anchor sites are too close to easy-axis poles')
    return torch.atan2(a[...,1],a[...,0]),weights/total


class AxialGauge(TransitionNet):
    def __init__(self,arm):
        if arm not in ('control','gauge2'):raise ValueError('unregistered axial candidate')
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


class ReversalGauge(AxialGauge):
    def __init__(self,arm):
        if arm not in ('control','reversal'):raise ValueError('unregistered reversal arm')
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
