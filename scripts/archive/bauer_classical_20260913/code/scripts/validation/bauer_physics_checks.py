"""Independent spectral, static-wall and canonical-equilibrium checks."""
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scripts.core.bauer_fast import drift, ensemble
from scripts.workflow.bauer_campaign import save


def static_checks():
    checks=[]
    length=25;s=np.zeros((length,3));s[:,2]=1.;epsilon=1e-6
    jac=np.empty((2*length,2*length))
    for k in range(2*length):
        plus=s.copy();minus=s.copy();plus[k//2,k%2]+=epsilon;minus[k//2,k%2]-=epsilon
        jac[:,k]=((drift(plus,.1,.1)-drift(minus,.1,.1))/(2*epsilon))[:,:2].ravel()
    eigen=np.linalg.eigvals(jac);positive=eigen[eigen.imag>0];positive=positive[np.argsort(positive.imag)]
    omega=.2+2*(1-np.cos(np.arange(length)*np.pi/length))
    error=float(max(abs(positive.imag-omega).max(),abs(positive.real+.1*omega).max()))
    checks.append(dict(name='open_chain_linear_modes',status='PASS' if error<1e-8 else 'FAIL',error=error,tolerance=1e-8))
    length=100;x=np.arange(length);guess=2*np.arctan(np.exp((x-(length-1)/2)*np.sqrt(.2)))
    def energy(angle):
        t=np.r_[0.,angle,np.pi]
        return float(-np.cos(np.diff(t)).sum()-.1*(np.cos(t)**2).sum())
    fit=minimize(energy,guess[1:-1],method='L-BFGS-B',options=dict(ftol=1e-13,gtol=1e-7,maxiter=5000))
    wall=fit.fun+length-1+.1*length;continuum=2*np.sqrt(.2)
    checks.append(dict(name='discrete_static_wall',status='PASS' if fit.success and abs(wall/continuum-1)<.03 else 'FAIL',
                       wall_energy=float(wall),continuum_energy=float(continuum),relative_tolerance=.03,
                       scope='Hamiltonian check; not dynamical reversal validation'))
    return checks


def run(root):
    start=time.monotonic();checks=static_checks()
    rng=np.random.default_rng(919);init=rng.normal(size=(128,1,3));init/=np.linalg.norm(init,axis=-1,keepdims=True)
    for anisotropy in (0.,.1):
        paths,raw=ensemble(init,300000000+int(anisotropy*1000000)+np.arange(128),
                            .005,1000000,2000,anisotropy,.1,.11)
        tail=paths[:,paths.shape[1]//2:,0]
        nodes,weights=np.polynomial.legendre.leggauss(256)
        weight=weights*np.exp(anisotropy*nodes**2/.11)
        expected_z2=float((weight*nodes**2).sum()/weight.sum())
        values=np.c_[tail.mean(1),(tail**2).mean(1)]
        expectation=np.array([0.,0.,0.,(1-expected_z2)/2,(1-expected_z2)/2,expected_z2])
        w=rng.multinomial(128,np.full(128,1/128),size=2000)/128
        point=values.mean(0);radius=float(np.quantile(np.max(abs(w@values-point),axis=1),.975))
        passed=bool(np.max(abs(point-expectation))+radius<=.05)
        checks.append(dict(name=f'canonical_single_spin_K{anisotropy}',status='PASS' if passed else 'INCONCLUSIVE',
                           expected=expectation.tolist(),estimate=point.tolist(),simultaneous_radius=radius,
                           tolerance=.05,confidence=.975,bootstrap_clusters='independent_trajectory',
                           initial_preparation='uniform sphere, first half excluded; not borrowed from LLG equilibrium',
                           max_raw_norm_error=float(raw.max()),
                           scope='Independent Boltzmann quadrature; small system only, target-length tests remain required'))
    save(Path(root)/'physics/result.json',dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'HOLD',
                                             checks=checks,wall_seconds=time.monotonic()-start))
