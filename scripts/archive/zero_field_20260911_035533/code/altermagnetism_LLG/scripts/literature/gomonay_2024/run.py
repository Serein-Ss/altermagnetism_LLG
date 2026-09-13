"""Full-space sinc spin waves or static/free-moving wall protocols."""
import torch
from scripts.core.reduced_llg import ReducedLLG
from scripts.literature.gomonay_2024.model import DoubleLayer
from scripts.literature.workflow import arguments,prepare,simulate,complete,rk4_step


def main():
    paper='gomonay_2024'; args=arguments(paper,('smoke','spinwave','control','wall','moving_wall'))
    r,p,raw,out,manifest=prepare(args,paper)
    if args.protocol=='control': r=dict(r,J_tilde=0.)
    wall=args.protocol in ('wall','moving_wall')
    model=DoubleLayer(r,p['shape'],wall=wall,periodic=tuple(p['periodic']),orientation=p['orientation'])
    llg=ReducedLLG(model,alpha=p['alpha'],theta=0.)
    if wall:
        s=model.wall_state(velocity=p.get('velocity',0.),batch=p['batch'],device=args.device)
        step=lambda x,t,dt,dw:rk4_step(llg,x,t,dt)
        simulate(raw/'trajectory.h5',s,llg,p,step_fn=step)
    else:
        vectors=p.get('wavevectors',[p['wavevector_cutoff']])
        for index,vector in enumerate(vectors):
            s=model.ground_state(batch=p['batch'],device=args.device)
            coords=model.coordinates(device=s.device,dtype=s.dtype)
            center=(coords.reshape(-1,2).max(0).values+coords.reshape(-1,2).min(0).values)/2
            # sinc(z)=sin(pi*z)/(pi*z); separate independent directions,
            # not a claim that one diagonal pulse excites the entire BZ.
            spatial=torch.sinc(2*((coords-center)*s.new_tensor(vector)).sum(-1))
            direction=s.new_tensor(p['field_direction'])
            field_at=lambda t: p['field_amplitude']*spatial[None,...,None]*direction*torch.sinc(s.new_tensor(2*p['frequency_cutoff']*(t-p['pulse_center'])))
            step=lambda x,t,dt,dw:rk4_step(llg,x,t,dt,field_at)
            filename='trajectory.h5' if len(vectors)==1 else f'trajectory_{index:03d}.h5'
            simulate(raw/filename,s,llg,dict(p,wavevector_cutoff=vector),step_fn=step)
    complete(raw,out,manifest)


if __name__=='__main__': main()
