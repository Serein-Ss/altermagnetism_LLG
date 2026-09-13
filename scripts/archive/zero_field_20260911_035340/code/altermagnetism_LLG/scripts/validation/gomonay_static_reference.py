"""Independent CPU Metropolis reference for S1, unrotated periodic cells.

Explicit site-pair matrix assembled from the published bond displacements, not
from DoubleLayer/CellHamiltonian. MC sweeps are NOT physical LLG time. This
small-lattice reference runs in one bounded job and is not a preparation pool.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from numba import njit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.core.literature_config import load_runtime,sha256


def coupling_matrix(length,r):
    if length<4: raise ValueError('reference requires L>=4')
    matrix=np.zeros((2*length**2,2*length**2))
    def index(x,y,a): return 2*((x%length)*length+y%length)+a
    for x in range(length):
        for y in range(length):
            a=index(x,y,0)
            for dx,dy in ((0,0),(-1,0),(0,-1),(-1,-1)):
                b=index(x+dx,y+dy,1);matrix[a,b]-=r['J1'];matrix[b,a]-=r['J1']
            for sub,sign in ((0,1),(1,-1)):
                a=index(x,y,sub)
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)):
                    b=index(x+dx,y+dy,sub)
                    matrix[a,b]+=r['J2'] if dx*dy==0 else sign*r['J_tilde']*dx*dy
    return matrix


@njit(cache=False)
def sample(matrix,k,theta,sweeps,stride,seed,initial):
    np.random.seed(seed)
    n=len(matrix);spins=np.zeros((n,3))
    for i in range(n):
        if initial<2: spins[i,2]=(1 if i%2==0 else -1)*(1 if initial==0 else -1)
        else:
            v=np.random.normal(0.,1.,3);spins[i]=v/np.sqrt(np.sum(v*v))
    out=np.empty((sweeps//stride,n,3));accepted=0
    for sweep in range(sweeps):
        for _ in range(n):
            i=np.random.randint(n);v=np.random.normal(0.,1.,3);v/=np.sqrt(np.sum(v*v))
            field=matrix[i]@spins
            delta=-np.sum((v-spins[i])*field)-k*(v[2]**2-spins[i,2]**2)
            if delta<=0 or np.random.random()<np.exp(-delta/theta):
                spins[i]=v;accepted+=1
        if (sweep+1)%stride==0: out[(sweep+1)//stride-1]=spins
    return out,accepted/(sweeps*n)


def run(config,destination):
    import yaml
    config=Path(config);c=yaml.safe_load(config.read_text())
    r=load_runtime(c['material_config']).reduced
    matrix=coupling_matrix(c['length'],r)
    states=[];accept=[]
    for chain in range(4):
        values,rate=sample(matrix,r['K_DW'],c['theta'],c['sweeps'],c['stride'],c['seed']+chain,chain)
        states.append(values);accept.append(rate)
    states=np.asarray(states)
    e=-.5*np.einsum('ctia,ij,ctja->ct',states,matrix,states)-r['K_DW']*(states[...,2]**2).sum(-1)
    signs=np.tile([1.,-1.],c['length']**2)
    neel=(states*signs[None,None,:,None]).mean(2)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as f:
        np.savez_compressed(f,spins=states,energy=e,neel=neel,sweep=np.arange(c['stride'],c['sweeps']+1,c['stride']),
            acceptance=accept,metadata=json.dumps(dict(config=c,config_sha256=sha256(config),
                material_sha256=sha256(c['material_config']),code_sha256=sha256(__file__),
                scope='static_distribution_only_not_physical_time',production_enabled=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.output)
