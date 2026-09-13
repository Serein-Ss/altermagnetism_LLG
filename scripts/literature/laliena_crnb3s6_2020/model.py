"""Dimensionless monoaxial chiral magnet and CORRECTED 2022 BVP.

x=q0*z, t=gamma*B0*t_SI. All constructor inputs are dimensionless.
Energy per grid point approximates integral energy / dx.
"""
import numpy as np
import torch
from scipy.integrate import solve_bvp
from scripts.core.reduced_llg import ReducedLLG


class ChiralChain:
    def __init__(self, reduced, *, dx, periodic=True):
        if dx<=0:
            raise ValueError('positive reduced mesh spacing required')
        self.dx=float(dx); self.periodic=periodic
        self.kappa=float(reduced['kappa']); self.hy=float(reduced['h_y'])

    def energy(self,s):
        a=s if self.periodic else s[:,:-1]
        b=s.roll(-1,1) if self.periodic else s[:,1:]
        bonds=(b-a).square().sum(-1)/(2*self.dx**2)-torch.linalg.cross(a,b)[...,2]/self.dx
        return bonds.sum(1)-.5*self.kappa*s[...,2].square().sum(1)-self.hy*s[...,1].sum(1)

    def field(self,s):
        a=s if self.periodic else s[:,:-1]
        b=s.roll(-1,1) if self.periodic else s[:,1:]
        z=s.new_tensor((0.,0.,1.)).expand_as(a)
        left=(b-a)/self.dx**2-torch.linalg.cross(z,b)/self.dx
        right=(a-b)/self.dx**2+torch.linalg.cross(z,a)/self.dx
        if self.periodic:
            field=left+right.roll(1,1)
        else:
            field=torch.zeros_like(s)
            field[:,:-1]+=left; field[:,1:]+=right
        field[...,2]+=self.kappa*s[...,2]
        field[...,1]+=self.hy
        return field

    def derivative(self,s):
        if not self.periodic:
            raise ValueError('current protocol requires the paper periodic boundary')
        ds=(s.roll(-1,1)-s.roll(1,1))/(2*self.dx)
        return ds-(ds*s).sum(-1,keepdim=True)*s


class CurrentLLG(ReducedLLG):
    def __init__(self, model, *, alpha, beta, u, theta=0.):
        super().__init__(model,alpha=alpha,theta=theta)
        self.beta=float(beta); self.u=float(u)

    def increment(self,s,dt,dw):
        # Signed transport convention: tau=-u*d_x n+beta*u*n x d_x n.
        # Hence rigid velocity v=(beta/alpha)*u and Gamma=v-u.
        # u is a torque coefficient, NOT an unlabelled positive SI current.
        ds=self.model.derivative(s)
        torque=-self.u*ds+self.beta*self.u*torch.linalg.cross(s,ds)
        return super().increment(s,dt,dw)+dt*(torque+self.alpha*torch.linalg.cross(s,torque))/(1+self.alpha**2)


def soliton(x,hy):
    if hy<=0:
        raise ValueError('isolated soliton requires positive transverse field')
    x=np.asarray(x); u=np.sqrt(hy)*x
    phi=4*np.arctan(np.exp(np.clip(u,-700,700)))
    return np.stack((np.full_like(x,np.pi/2),np.zeros_like(x),phi,
                     2*np.sqrt(hy)/np.cosh(np.clip(u,-700,700))))


def bvp_rhs(x,y,*,kappa,hy,gamma,omega=0.):
    """Author Correction equations 13/14, not the superseded 2020 equations."""
    theta,tp,phi,pp=y[:4]
    st,ct=np.sin(theta),np.cos(theta)
    if np.any(np.abs(st)<1e-10):
        raise ValueError('spherical-coordinate pole reached; branch not resolved')
    tpp=(pp*pp-2*pp+kappa)*st*ct-hy*ct*np.cos(phi)-omega*tp+gamma*st*pp
    ppp=(hy*np.sin(phi)-2*(pp-1)*ct*tp-gamma*tp-omega*st*pp)/st
    return np.array((tp,tpp,pp,ppp))


def boundary(a,b):
    # Reflection symmetry pins the center and removes the translation null mode.
    return np.array((a[1],a[2]-np.pi,b[0]-np.pi/2,b[2]-2*np.pi))


def solve_profile(reduced,gamma,*,extent=20.,points=301,tolerance=1e-6,previous=None):
    x=np.linspace(0,extent,points)
    initial=soliton(x,reduced['h_y']) if previous is None else previous.sol(x)[:4]
    result=solve_bvp(lambda x,y:bvp_rhs(x,y,kappa=reduced['kappa'],hy=reduced['h_y'],gamma=gamma),
                     boundary,x,initial,tol=tolerance,max_nodes=20000)
    if not result.success:
        raise RuntimeError('BVP failed (not evidence of critical current): '+result.message)
    return result


def continue_branch(reduced,*,steps=100,ds=.1,extent=20.,points=301,tolerance=1e-6):
    """Pseudo-arclength continuation through the fold; solver failure is fatal.

    Returns the stable-connected and continued branches, not a claim of
    stability. Gamma_c requires fold, mesh, extent and ds convergence checks.
    """
    x=np.linspace(0,extent,points)
    a=solve_profile(reduced,0.,extent=extent,points=points,tolerance=tolerance)
    b=solve_profile(reduced,.02,extent=extent,points=points,tolerance=tolerance,previous=a)
    ya,yb=a.sol(x)[:4],b.sol(x)[:4]; ga,gb=0.,.02
    profiles=[ya,yb]; gammas=[ga,gb]
    residuals=[float(a.rms_residuals.max()),float(b.rms_residuals.max())]
    for _ in range(steps):
        tangent=yb-ya; tg=gb-ga
        norm=np.sqrt(np.trapezoid((tangent*tangent).sum(0),x)+tg*tg)
        tangent/=norm; tg/=norm
        def rhs(z,y,p):
            ref=np.array([np.interp(z,x,v) for v in yb])
            tan=np.array([np.interp(z,x,v) for v in tangent])
            physical=bvp_rhs(z,y,kappa=reduced['kappa'],hy=reduced['h_y'],gamma=p[0])
            return np.vstack((physical,((y[:4]-ref)*tan).sum(0)))
        def bc(left,right,p):
            return np.r_[boundary(left,right),left[4],right[4]+(p[0]-gb)*tg-ds]
        guess=yb+ds*tangent
        integral=np.r_[0.,np.cumsum(.5*np.diff(x)*(ds*(tangent*tangent).sum(0))[1:]+
                                   .5*np.diff(x)*(ds*(tangent*tangent).sum(0))[:-1])]
        out=solve_bvp(rhs,bc,x,np.vstack((guess,integral)),p=[gb+ds*tg],tol=tolerance,max_nodes=20000)
        if not out.success:
            raise RuntimeError('arclength continuation failed: '+out.message)
        ya,yb=yb,out.sol(x)[:4]; ga,gb=gb,float(out.p[0])
        profiles.append(yb); gammas.append(gb); residuals.append(float(out.rms_residuals.max()))
        # Stop after the first fold has five samples on its returning branch.
        # A failed solver never serves as a critical-current detector.
        if len(gammas)-int(np.argmax(gammas))>=6 and np.argmax(gammas)>0:
            break
    return {'x':x,'profiles':np.array(profiles),'gamma':np.array(gammas),
            'residual':np.array(residuals)}


def profile_spins(x,profile):
    """Reconstruct full centered solution using the paper's Eq.4 convention."""
    x=np.asarray(x); y=profile.sol(np.abs(x))
    theta=y[0]; phi=np.where(x<0,2*np.pi-y[2],y[2])
    return np.stack((-np.sin(theta)*np.sin(phi),np.sin(theta)*np.cos(phi),np.cos(theta)),-1)

