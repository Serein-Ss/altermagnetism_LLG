"""Open-chain reflection audit of frozen checkpoints, not an accuracy experiment."""
from pathlib import Path
import hashlib
import torch
from torch.nn import functional as F
from tests.bauer_closed_loop_20260919.model import Net, hierarchy
from tests.bauer_two_stage_20260919.common import dump

HERE=Path(__file__).resolve().parent


def symmetric_hierarchy(x):
    """Candidate operator only; not installed into any trained network."""
    return .5*(hierarchy(x)+hierarchy(x.flip(-2)).flip(-2))


def main():
    torch.set_num_threads(1);rows=[]
    for seed in [11,22,33]:
        path=HERE.parent/'bauer_moment_training_20260920/runs'/f'w0_{seed}'/'best.pt'
        net=Net('C',64,3).eval()
        net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        torch.manual_seed(45)
        with torch.no_grad():
            for n in [16,25,32,50,100]:
                x=F.normalize(torch.randn(2,2,n,3),dim=-1)
                tau=torch.tensor([.25,.75]);t=torch.tensor([0.,.1]);temp=63.82484966960072
                a=net(x,tau,x[:,0],t,temp)
                z=x.flip(-2);b=net(z,tau,z[:,0],t,temp).flip(-2)
                candidate=symmetric_hierarchy(x)-symmetric_hierarchy(z).flip(-2)
                rows.append(dict(seed=seed,L=n,reflection_max=float(abs(a-b).max()),
                     reflection_relative_rms=float((a-b).square().mean().sqrt()/a.square().mean().sqrt()),
                     candidate_operator_reflection_max=float(abs(candidate).max()),
                     checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    dump(HERE/'symmetry.json',dict(rows=rows, scientific_certification=False,
         note='Synthetic random states with fixed seed. Tests an exact structural symmetry of the open exchange/uniaxial chain, '
         'not model distribution error. Bidirectional pooling is an untrained candidate; frozen models unchanged.'))


if __name__=='__main__':main()
