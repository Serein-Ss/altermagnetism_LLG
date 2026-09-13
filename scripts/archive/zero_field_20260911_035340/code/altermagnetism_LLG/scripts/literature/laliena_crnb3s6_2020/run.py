"""Corrected BVP profiles/fold and current-driven, non-winding-fixed LLG."""
import numpy as np
import torch
from scripts.literature.laliena_crnb3s6_2020.model import ChiralChain,CurrentLLG,solve_profile,continue_branch,profile_spins
from scripts.literature.workflow import arguments,prepare,simulate,complete


def main():
    paper='laliena_crnb3s6_2020'; args=arguments(paper,('smoke','profile','branch','helix','current'))
    r,p,raw,out,manifest=prepare(args,paper)
    if args.protocol=='branch':
        data=continue_branch(r,steps=p['continuation_steps'],ds=p['ds'],extent=p['extent'],points=p['points'],tolerance=p['tolerance'])
        np.savez_compressed(raw/'branch.npz',**data)
    elif args.protocol=='profile':
        result=None
        for gamma in np.linspace(0,p['gamma'],p['ramp_steps']+1):
            result=solve_profile(r,float(gamma),extent=p['extent'],points=p['points'],tolerance=p['tolerance'],previous=result)
        x=np.linspace(-p['extent'],p['extent'],2*p['points']-1)
        np.savez_compressed(raw/'profile.npz',x=x,spins=profile_spins(x,result),
                            gamma=p['gamma'],residual=result.rms_residuals.max())
    else:
        if args.protocol=='helix': r=dict(r,h_y=0.)
        model=ChiralChain(r,dx=p['dx'])
        llg=CurrentLLG(model,alpha=r['alpha'],beta=r['beta'],u=p['u'])
        generator=torch.Generator(device=args.device).manual_seed(p['seed'])
        if args.protocol=='current':
            result=solve_profile(r,0.,extent=p['extent'],points=p['points'])
            x=(np.arange(p['sites'])-(p['sites']-1)/2)*p['dx']
            if abs(x).max()>p['extent']: raise ValueError('BVP extent must cover initialization mesh')
            s=torch.tensor(profile_spins(x,result),dtype=torch.float64,device=args.device)[None].repeat(p['batch'],1,1)
        else:
            s=torch.randn((p['batch'],p['sites'],3),dtype=torch.float64,device=args.device,generator=generator)
            s/=s.norm(dim=-1,keepdim=True)
        simulate(raw/'trajectory.h5',s,llg,p)
    complete(raw,out,manifest)


if __name__=='__main__': main()
