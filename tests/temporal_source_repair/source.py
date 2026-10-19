"""Stationary per-site AR(1) latent; not a physical thermal-noise process."""
import math
import torch
from torch.nn import functional as F


def correlate(noise,rho):
    if not math.isfinite(rho) or not 0<=rho<1:raise ValueError('rho must be in [0,1)')
    if rho==0:return noise
    result=noise.clone()
    # Frame zero will be clamped; frame one starts a stationary Gaussian chain.
    for t in range(2,noise.shape[1]):
        result[:,t]=rho*result[:,t-1]+math.sqrt(1-rho*rho)*noise[:,t]
    return result


def from_noise(initial,noise,rho):
    if noise.shape[0]!=len(initial) or noise.shape[2:]!=initial.shape[1:]:raise ValueError('shape mismatch')
    z=F.normalize(correlate(noise,rho)+2*initial[:,None],dim=-1)
    z[:,0]=initial
    return z


def draw(initial,frames,generator,rho):
    noise=torch.randn((len(initial),frames,*initial.shape[1:]),device=initial.device,dtype=initial.dtype,generator=generator)
    return from_noise(initial,noise,rho)


def contract(rho):
    if rho not in (0.,.8):raise ValueError('unregistered source')
    return dict(version=1,rho=rho,law='per_site_stationary_AR1_gaussian_then_shift2S0_and_normalize',
                block_ps=1.,save_ps=.1,frames=11,boundary='open',temperature_K=63.824849668,
                anchor='supplied_S0',validated_physical_domain=False)


def check_package(checkpoint,rho):
    if checkpoint['source_contract']!=contract(rho):raise ValueError('checkpoint/source mismatch')
