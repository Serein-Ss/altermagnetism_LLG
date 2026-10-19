"""Function-preserving depth expansion of the frozen two-frame graph model."""
import torch
from tests.transition_block_repair.model import TransitionNet
from tests.spin_path_transformer.model import Block

class DeepNet(TransitionNet):
    def __init__(self):
        super().__init__()
        self.blocks.extend(Block(self.width,self.architecture) for _ in range(4))
        self.depth=8

    def get_extra_state(self):
        state=super().get_extra_state()
        state.update(package='depth_capacity_repair',version=3,depth=8,
                     initialization='old4blocks_plus4identity_gatedblocks')
        return state

def initialize(ck,enabled,seed,device='cpu'):
    base=TransitionNet();base.load_state_dict(ck['model'])
    original_names=[name for name,_ in base.named_parameters()]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(961000+seed)
        net=DeepNet() if enabled else base
    if enabled:
        result=net.load_state_dict({k:v for k,v in ck['model'].items() if k!='_extra_state'},strict=False)
        assert not result.unexpected_keys
        assert all(k=='_extra_state' or k.startswith(tuple(f'blocks.{i}.' for i in range(4,8))) for k in result.missing_keys)
    net=net.to(device)
    params=dict(net.named_parameters())
    # Named mapping matters: appended blocks change the state_dict parameter order.
    opt=torch.optim.AdamW([params[n] for n in original_names],lr=3e-5,weight_decay=1e-5)
    opt.load_state_dict(ck['optimizer'])
    added=[p for n,p in net.named_parameters() if n not in original_names]
    if added:opt.add_param_group({'params':added,'lr':3e-4,'weight_decay':1e-5})
    return net,opt

@torch.no_grad()
def activation(net,bank):
    if not isinstance(net,DeepNet):return []
    from tests.temporal_source_repair.source import draw
    from scripts.model.gate_f import path_interpolate
    y=torch.stack([b[0] for b in bank]);records=[]
    for tv in (.1,.5,.9):
        g=torch.Generator(device=y.device).manual_seed(961900)
        z=draw(y[:,0],2,g,0.);tau=torch.full((len(y),),tv,device=y.device)
        x,_,_=path_interpolate(z,y,tau,g)
        full=net.velocity(x,tau,y[:,0],y.new_tensor([0.,.1]))
        blocks=net.blocks
        try:
            net.blocks=torch.nn.ModuleList(list(blocks[:4]))
            off=net.velocity(x,tau,y[:,0],y.new_tensor([0.,.1]))
        finally:net.blocks=blocks
        delta=(full[:,1:]-off[:,1:]).square().mean().sqrt()
        records.append(dict(tau=tv,added_velocity_rms=float(delta),
            relative_rms=float(delta/full[:,1:].square().mean().sqrt().clamp_min(1e-12))))
    return records
