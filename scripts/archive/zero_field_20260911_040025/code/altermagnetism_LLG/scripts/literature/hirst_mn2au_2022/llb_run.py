"""AFM-LLB AFMR with explicit published-fit equilibrium input (not ASD fitted)."""
import math
import json
import h5py
import torch
from scripts.literature.hirst_mn2au_2022.llb import AFMLLB
from scripts.literature.workflow import arguments,prepare,complete


def main():
    paper='hirst_mn2au_2022'; args=arguments(paper,('llb_smoke','llb_afmr'))
    r,p,raw,out,manifest=prepare(args,paper)
    theta=r[p['theta_key']]
    # Input boundary: me is supplied or computed from the explicitly selected
    # published equilibrium fit. Never label this as a fit to our new ASD data.
    me=p['me'] if 'me' in p else (1-theta/r['theta_TN'])**p['published_fit_exponent']
    model=AFMLLB(r,theta=theta,theta_n=r['theta_TN'],me=me,lam=p['alpha'])
    angle=math.radians(p['tilt_degrees'])
    m=torch.tensor([[[me*math.cos(angle),0.,me*math.sin(angle)],
                     [-me*math.cos(angle),0.,-me*math.sin(angle)]]],dtype=torch.float64,device=args.device)
    times=list(range(0,p['steps']+1,p['save_every']))
    if times[-1]!=p['steps']: times.append(p['steps'])
    with h5py.File(raw/'trajectory.h5','x') as h:
        h.attrs.update(complete=False,model_kind='AFM_LLB',me=me,protocol_json=json.dumps(p))
        h.create_dataset('time',data=[i*p['dt'] for i in times])
        spins=h.create_dataset('spins',(len(times),1,2,3),dtype='f8')
        a=h.create_dataset('m_a',(len(times),1,3),dtype='f8')
        b=h.create_dataset('m_b',(len(times),1,3),dtype='f8')
        frame=0
        with torch.no_grad():
            for i in range(p['steps']+1):
                if i==times[frame]:
                    spins[frame]=m.cpu().numpy(); a[frame]=m[:,0].cpu().numpy(); b[frame]=m[:,1].cpu().numpy()
                    frame+=1; h.flush()
                if i==p['steps']: break
                m=model.step(m,p['dt'])
                if not torch.isfinite(m).all(): raise FloatingPointError('LLB state diverged')
        h.attrs['complete']=True
    complete(raw,out,manifest)


if __name__=='__main__': main()
