"""Dimensionless Stratonovich spin dynamics. No material or SI constants here.

Hamiltonians return h and b=-dh/ds. State shape is [batch, sites, 3].
Noise arguments are Wiener increments (variance dt), never resampled in stages.
"""
from __future__ import annotations
import math
from typing import Protocol
import torch


class ReducedHamiltonian(Protocol):
    def energy(self, spins: torch.Tensor) -> torch.Tensor: ...
    def field(self, spins: torch.Tensor) -> torch.Tensor: ...


class BondHamiltonian:
    """Explicit undirected bonds: h=-sum J_ij s_i.s_j-K sum sz²-b.sum s.

    Each bond is provided exactly once. Geometry belongs to the material model,
    not to the integrator. No Mn2Au/Gomonay neighbor tables are inferred here.
    """
    def __init__(self, bonds, coupling, anisotropy=0., field=(0.,0.,0.)):
        self.bonds = torch.as_tensor(bonds, dtype=torch.long).reshape(-1, 2)
        pairs = [tuple(sorted(pair)) for pair in self.bonds.tolist()]
        if any(i == j for i,j in pairs) or len(set(pairs)) != len(pairs):
            raise ValueError("provide each non-self undirected bond exactly once")
        self.coupling = torch.as_tensor(coupling, dtype=torch.float64).expand(len(pairs))
        self.anisotropy = float(anisotropy)
        self.external_field = tuple(field)

    def energy(self, s):
        ij=self.bonds.to(s.device); j=self.coupling.to(s)
        exchange=-(j*(s[:,ij[:,0]]*s[:,ij[:,1]]).sum(-1)).sum(-1)
        return exchange-self.anisotropy*s[...,2].square().sum(-1)-(s*s.new_tensor(self.external_field)).sum((-1,-2))

    def field(self, s):
        ij=self.bonds.to(s.device); j=self.coupling.to(s)
        b=torch.zeros_like(s)+s.new_tensor(self.external_field)
        b.index_add_(1,ij[:,0],s[:,ij[:,1]]*j[None,:,None])
        b.index_add_(1,ij[:,1],s[:,ij[:,0]]*j[None,:,None])
        b[...,2]+=2*self.anisotropy*s[...,2]
        return b


class ReducedLLG:
    def __init__(self, model: ReducedHamiltonian, *, alpha: float,
                 theta: float, equation_convention="gilbert"):
        if not all(math.isfinite(x) and x >= 0 for x in (alpha, theta)):
            raise ValueError("alpha and theta must be finite and nonnegative")
        if equation_convention not in ("gilbert", "bauer_ll"):
            raise ValueError("unknown equation convention")
        self.model=model; self.alpha=alpha; self.theta=theta
        self.equation_convention=equation_convention

    def noise_increment(self, spins, dt, generator):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        return torch.randn(spins.shape, dtype=spins.dtype, device=spins.device,
                           generator=generator)*math.sqrt(dt)

    def increment(self, s, dt, dw):
        b=self.model.field(s)
        thermal=math.sqrt(2*self.alpha*self.theta)*dw
        cross=torch.linalg.cross
        if self.equation_convention == "gilbert":
            total=dt*b+thermal
            return -(cross(s,total)+self.alpha*cross(s,cross(s,total)))/(1+self.alpha**2)
        # Bauer: only precession receives thermal noise; lambda=alpha here.
        return -cross(s,dt*b+thermal)-self.alpha*dt*cross(s,cross(s,b))

    def step(self, s, dt, dw, *, method="paper_midpoint",
             tolerance=1e-13, max_iterations=100):
        if dt <= 0 or dw.shape != s.shape:
            raise ValueError("positive dt and a matching Wiener increment required")
        if method == "geometric_midpoint":
            candidate=s.clone()
            for _ in range(max_iterations):
                updated=s+self.increment((s+candidate)*.5,dt,dw)
                error=(updated-candidate).abs().max().item()
                candidate=updated
                if error < tolerance:
                    residual=(candidate-s-self.increment((s+candidate)*.5,dt,dw)).abs().max().item()
                    if residual < tolerance:
                        norm=(candidate.norm(dim=-1)-1).abs().max()
                        return candidate, torch.stack((norm,norm))
            raise RuntimeError("geometric midpoint failed to converge; no fallback")
        first=self.increment(s,dt,dw)
        if method == "paper_midpoint":
            # Nishino Appendix B5/B6 and its stated half-increment replacement.
            predictor=s+.5*first
            raw=s+self.increment(predictor,dt,dw)
        elif method in ("heun", "heun_projected_predictor"):
            predictor=s+first
            if method == "heun_projected_predictor":
                predictor=predictor/predictor.norm(dim=-1,keepdim=True)
            raw=s+.5*(first+self.increment(predictor,dt,dw))
        else:
            raise ValueError("unknown integrator")
        errors=torch.stack(((predictor.norm(dim=-1)-1).abs().max(),
                            (raw.norm(dim=-1)-1).abs().max()))
        # Explicit project choice, not a claim that paper B5 preserves norms.
        return raw/raw.norm(dim=-1,keepdim=True), errors

