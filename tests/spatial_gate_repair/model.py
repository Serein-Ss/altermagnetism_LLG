"""Controlled residual gate; frozen historical backbone is not edited."""
import torch
from tests.transition_block_repair.model import TransitionNet

SCALES={'control':1.,'attenuated':.25,'local':0.}


class GatedTransition(TransitionNet):
    def __init__(self,scale):
        if scale not in SCALES.values():raise ValueError('unregistered spatial gate')
        super().__init__();self.spatial_scale=scale
        for block in self.blocks:block.modulation.register_forward_hook(self._gate)

    def _gate(self,module,args,output):
        # Block chunk 8 is the scalar residual gate of coarse+latent communication.
        c=self.width
        return torch.cat((output[...,:8*c],output[...,8*c:9*c]*self.spatial_scale,output[...,9*c:]),-1)

    def get_extra_state(self):
        state=super().get_extra_state()
        state.update(package='spatial_gate_repair',spatial_scale=self.spatial_scale)
        return state


def initialize(checkpoint,scale,seed,device='cpu'):
    # Validate old metadata before the explicit, recorded package conversion.
    original=TransitionNet();original.load_state_dict(checkpoint['model'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(978000+seed);net=GatedTransition(scale)
    state=dict(checkpoint['model']);state['_extra_state']=net.get_extra_state()
    net.load_state_dict(state);net=net.to(device)
    return net,torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-5)


@torch.no_grad()
def remote_probe(net):
    """One velocity-call sensitivity, NOT physical propagation or endpoint locality."""
    device=next(net.parameters()).device;dtype=next(net.parameters()).dtype
    gen=torch.Generator(device=device).manual_seed(978900)
    a=torch.nn.functional.normalize(torch.randn(4,25,3,device=device,dtype=dtype,generator=gen),dim=-1)
    x=torch.nn.functional.normalize(torch.randn(4,2,25,3,device=device,dtype=dtype,generator=gen),dim=-1);x[:,0]=a
    remote=(torch.arange(25,device=device)-12).abs()>4
    alternate=torch.nn.functional.normalize(torch.randn(x.shape,device=device,dtype=dtype,generator=gen),dim=-1)
    rows=[]
    for tau in (.1,.4,.7,.9):
        t=x.new_full((4,),tau);base=net.velocity(x,t,a,x.new_tensor([0.,.1]))[:,1,12]
        for kind in ('draft','anchor','both'):
            xx=x.clone();aa=a.clone()
            if kind in ('draft','both'):xx[:,1,remote]=alternate[:,1,remote]
            if kind in ('anchor','both'):aa[:,remote]=alternate[:,0,remote];xx[:,0]=aa
            value=net.velocity(xx,t,aa,x.new_tensor([0.,.1]))[:,1,12]
            diff=(value-base).norm(dim=-1)
            rows.append(dict(tau=tau,changed=kind,mean_change=float(diff.mean()),max_change=float(diff.max()),
                             base_mean_norm=float(base.norm(dim=-1).mean())))
    return rows
