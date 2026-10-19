import torch
from tests.chain_reversal_repair.model import ReversalGauge,gauge_probe,reversal_probe

ARMS=('control','local_moments','spatial_moments')


def initialize(checkpoint,arm,seed,device='cpu'):
    if arm not in ARMS:raise ValueError('unregistered moment arm')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(982000+seed);net=ReversalGauge('reversal')
    net.load_state_dict(checkpoint['model']);net=net.to(device)
    return net,torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-5)
