"""Three-dimensional ASD: equilibrium, equilibrate-then-tilt AFMR, domain wall."""
import math
import torch
from scripts.core.reduced_llg import ReducedLLG
from scripts.literature.hirst_mn2au_2022.model import Mn2Au,PinnedEndLLG
from scripts.literature.workflow import arguments,prepare,simulate,complete


def main():
    paper='hirst_mn2au_2022'; args=arguments(paper,('smoke','equilibrium','afmr','wall'))
    r,p,raw,out,manifest=prepare(args,paper)
    model=Mn2Au(r,p['shape'],tuple(p['periodic']))
    s=model.ground_state(batch=p['batch'],device=args.device)
    theta=r[p['theta_key']] if 'theta_key' in p else p['theta']
    solver=PinnedEndLLG if args.protocol=='wall' else ReducedLLG
    llg=solver(model,alpha=p['alpha'],theta=theta)
    if args.protocol=='afmr':
        eq=dict(p,steps=p['equilibration_steps'],alpha=1.)
        s=simulate(raw/'equilibration.h5',s,ReducedLLG(model,alpha=1.,theta=theta),eq,
                   observer=model.order_parameters)
        angle=math.radians(p['tilt_degrees']); x,z=s[...,0].clone(),s[...,2].clone()
        s[...,0]=x*math.cos(angle)-z*math.sin(angle)
        s[...,2]=x*math.sin(angle)+z*math.cos(angle)
        p=dict(p,seed=p['seed']+1)  # do not replay equilibration noise
    elif args.protocol=='wall':
        x=torch.arange(p['shape'][0],device=args.device,dtype=s.dtype)
        u=(x-(len(x)-1)/2)/p['initial_width_cells']
        n=torch.stack((-u.tanh(),1/u.cosh(),torch.zeros_like(u)),-1)
        s=n[None,:,None,None,None,:]*s.new_tensor(model.signs)[None,None,None,None,:,None]
        s=s.expand(p['batch'],*p['shape'],4,3).clone()
        s[:,0]=0.; s[:,-1]=0.
        s[:,0,...,0]=s.new_tensor(model.signs)
        s[:,-1,...,0]=-s.new_tensor(model.signs)
    simulate(raw/'trajectory.h5',s,llg,p,observer=model.order_parameters)
    complete(raw,out,manifest)


if __name__=='__main__': main()

