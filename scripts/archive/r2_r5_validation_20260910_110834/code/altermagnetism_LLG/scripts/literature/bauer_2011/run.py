"""Full Bauer trajectories with MT weak RK; Heun retained for comparison."""
import torch
from scripts.core.reduced_llg import ReducedLLG
from scripts.literature.bauer_2011.model import OpenChain
from scripts.literature.bauer_2011.integrator import BauerWeakRK
from scripts.literature.workflow import arguments,prepare,simulate,complete


def main():
    paper='bauer_2011'; args=arguments(paper,('smoke','weak_rk_smoke','trajectory'))
    r,p,raw,out,manifest=prepare(args,paper)
    model=OpenChain(r)
    s=torch.zeros((p['batch'],p['length'],3),dtype=torch.float64,device=args.device); s[...,2]=1.
    if p['method']=='milstein_tretyakov_5_9':
        llg=BauerWeakRK(model,alpha=r['alpha'],theta=r['theta'],project=p['project'])
    else:
        llg=ReducedLLG(model,alpha=r['alpha'],theta=r['theta'],equation_convention='bauer_ll')
    simulate(raw/'trajectory.h5',s,llg,p)
    complete(raw,out,manifest)


if __name__=='__main__': main()

