"""Explicit Bauer 0.1 ps package using unchanged variable-length backbone operations."""
import torch
from tests.spin_path_transformer.model import SpinPathTransformer
from tests.spin_path_transformer.run import make_net
from tests.temporal_source_repair.source import draw
from tests.transport_audit.run import integrate


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


def make_transition(task,device='cuda'):
    original=make_net(task,device);net=TransitionNet().to(device)
    state={k:v for k,v in original.state_dict().items() if k!='_extra_state'}
    result=net.load_state_dict(state,strict=False)
    assert result.missing_keys==['_extra_state'] and not result.unexpected_keys
    return net


def pair_window(y,starts):
    if y.shape[1]!=11 or starts.shape!=(len(y),) or torch.any(starts<0) or torch.any(starts>9):raise ValueError('pair selection')
    batch=torch.arange(len(y),device=y.device)
    return torch.stack((y[batch,starts],y[batch,starts+1]),1)


@torch.no_grad()
def rollout(net,initial,generator,steps,transitions=10):
    if not isinstance(transitions,int) or transitions<1:raise ValueError('positive transitions required')
    frames=[initial];state=initial;times=initial.new_tensor([0.,.1])
    for _ in range(transitions):
        z=draw(state,2,generator,0.)
        out=integrate(lambda x,t:net.velocity(x,t,state,times),z,0.,1.,steps,'midpoint')
        assert torch.equal(out[:,0],state)
        state=out[:,1];frames.append(state)
    return torch.stack(frames,1)
