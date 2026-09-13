"""Reduced two-sublattice AFM-LLB, PRB106 equations 5/6/9/12--16/19.

Thermal equilibrium inputs me, susceptibility, stiffness and enhancement
exponent are arguments, NOT fitted/invented inside this dynamics model.
Signs follow equations 5 and 12 (dissipative relaxation); the inconsistent
printed signs in Eq.18 are not copied. This convention needs disclosure.
"""
import torch


class AFMLLB:
    def __init__(self,reduced,*,theta,theta_n,me,lam,chi_parallel=None,
                 enhancement_z=None,enhancement_exponent=None,exchange_laplacian=0.):
        if theta<0 or theta_n<=0 or me<0 or lam<0:
            raise ValueError('invalid reduced thermodynamic inputs')
        if theta==theta_n:
            raise ValueError('critical-point singular formula requires a separate limiting model')
        self.theta=theta; self.theta_n=theta_n; self.me=me; self.lam=lam
        self.chi=chi_parallel; self.z=enhancement_z; self.exponent=enhancement_exponent
        self.j=float(reduced['J0_inter']); self.anis=(reduced['d_x'],0.,reduced['d_z'])
        self.stiffness=float(exchange_laplacian)
        if chi_parallel is not None and (chi_parallel<=0 or enhancement_z is None or enhancement_z<=0 or enhancement_exponent is None):
            raise ValueError('longitudinal dynamics requires positive susceptibility and an explicit enhancement model')

    def transverse_field(self,m):
        other=m.flip(-2)
        m2=m.square().sum(-1,keepdim=True)
        if torch.any(m2<=1e-20):
            raise ValueError('zero macrospin is outside the transverse LLB coordinate chart')
        projected=other-m*(m*other).sum(-1,keepdim=True)/m2
        # Callen-Callen K(T)=K0*me^3, field derivative w.r.t. m=me*n.
        b=self.j*projected+2*self.me*m*m.new_tensor(self.anis)
        if self.stiffness:
            if m.ndim!=4: raise ValueError('spatial LLB state must be [batch,cells,2,3]')
            b=b+self.stiffness*(m.roll(-1,1)+m.roll(1,1)-2*m)
        return b

    def rhs(self,m):
        m2=m.square().sum(-1,keepdim=True); length=m2.sqrt()
        b=self.transverse_field(m); cross=torch.linalg.cross
        ratio=self.theta/self.theta_n
        aperp=self.lam*(1-ratio/3 if ratio<1 else 2*ratio/3)
        result=-cross(m,b)-aperp*cross(m,cross(m,b))/m2
        if self.chi is not None:
            if ratio<1:
                if self.me<=0: raise ValueError('positive equilibrium magnetization below TN required')
                radial=(1-m2/self.me**2)/(2*self.chi)
            else:
                radial=-(1+.6*m2/(ratio-1))/self.chi
            apar=self.lam*2*ratio/3*(1+2/(self.z*length**self.exponent))
            result=result+apar*radial*m
        return result

    def step(self,m,dt):
        if dt<=0: raise ValueError('positive dt required')
        a=self.rhs(m); b=self.rhs(m+dt*a/2); c=self.rhs(m+dt*b/2); d=self.rhs(m+dt*c)
        updated=m+dt*(a+2*b+2*c+d)/6
        # Only the transverse approximation conserves each initial length.
        if self.chi is None:
            updated=updated/updated.norm(dim=-1,keepdim=True)*m.norm(dim=-1,keepdim=True)
        return updated
