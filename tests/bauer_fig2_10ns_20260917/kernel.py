"""Compiled CPU Bauer dynamics, with the existing PyTorch solver as reference.

Same MT 5.9 weak method and discrete draws; no fastmath and no GPU claim.
Noise seeds identify independent paths, independent of batch/thread scheduling.
The optional projected Heun method is an independently discretized cross-check.
"""
import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def drift(s, anisotropy, damping):
    out = np.empty_like(s)
    for i in range(len(s)):
        x, y, z = s[i]
        bx = by = bz = 0.0
        if i:
            bx += s[i-1, 0]; by += s[i-1, 1]; bz += s[i-1, 2]
        if i+1 < len(s):
            bx += s[i+1, 0]; by += s[i+1, 1]; bz += s[i+1, 2]
        bz += 2*anisotropy*z
        dot = x*bx+y*by+z*bz; norm = x*x+y*y+z*z
        out[i, 0] = -(y*bz-z*by)-damping*(x*dot-bx*norm)
        out[i, 1] = -(z*bx-x*bz)-damping*(y*dot-by*norm)
        out[i, 2] = -(x*by-y*bx)-damping*(z*dot-bz*norm)
    return out


@njit(cache=True)
def cross(a, b):
    out = np.empty_like(a)
    for i in range(len(a)):
        out[i, 0] = a[i, 1]*b[i, 2]-a[i, 2]*b[i, 1]
        out[i, 1] = a[i, 2]*b[i, 0]-a[i, 0]*b[i, 2]
        out[i, 2] = a[i, 0]*b[i, 1]-a[i, 1]*b[i, 0]
    return out


@njit(cache=True)
def mt_step(s, xi, zeta, dt, anisotropy, damping, theta, project=True):
    eps = math.sqrt(2*damping*theta)
    a = drift(s, anisotropy, damping)
    sigma = -cross(s, xi); noise = eps*math.sqrt(dt)*sigma
    k1 = dt*a; k2 = dt*drift(s+k1/2, anisotropy, damping)
    k3 = dt*drift(s+noise+k2/2, anisotropy, damping)
    k4 = dt*drift(s+noise+k3-3*eps**2*dt*s, anisotropy, damping)
    derivative = np.empty_like(s)
    for i in range(len(s)):
        trace = 0.0
        for j in range(3):
            trace += (xi[i,j]**2-zeta[i,j]**2)/2
        for j in range(3):
            v = -trace*s[i,j]
            for k in range(3):
                gamma = 1.0 if j >= k else -1.0
                v += (xi[i,j]*xi[i,k]-gamma*zeta[i,j]*zeta[i,k])*s[i,k]/2
            derivative[i,j] = v
    end = s+noise+dt*(a-eps**2*s)
    raw = (s+noise+eps**2*dt*derivative-eps*dt**1.5*cross(a,xi)/2
           -eps**3*dt**1.5*sigma/2+(k1+2*k2+2*k3+k4)/6
           +eps**2*dt*(-s-end)/2)
    error = 0.0
    for i in range(len(s)):
        norm = math.sqrt((raw[i]**2).sum())
        if not math.isfinite(norm) or norm == 0:
            raise ValueError('nonfinite/zero Bauer spin')
        error = max(error, abs(norm-1))
        if project:
            raw[i] /= norm
    return raw, error


@njit(cache=True)
def heun_step(s, dw, dt, anisotropy, damping, theta):
    eps = math.sqrt(2*damping*theta)
    first = dt*drift(s,anisotropy,damping)-eps*cross(s,dw)
    predictor = s+first
    raw = s+(first+dt*drift(predictor,anisotropy,damping)-eps*cross(predictor,dw))/2
    error = 0.0
    for i in range(len(s)):
        norm = math.sqrt((raw[i]**2).sum())
        if not math.isfinite(norm) or norm == 0:
            raise ValueError('nonfinite/zero Heun spin')
        error = max(error,abs(norm-1)); raw[i] /= norm
    return raw,error


@njit(cache=True)
def trajectory(initial, seed, dt, steps, stride, anisotropy, damping, theta,
               method=0, project=True):
    np.random.seed(seed)
    s = initial.copy()
    result = np.empty((steps//stride+1,len(s),3),np.float64)
    result[0] = s; maximum = 0.0
    for step in range(1,steps+1):
        xi = np.empty_like(s); zeta = np.empty_like(s)
        for i in range(len(s)):
            for j in range(3):
                if method == 0:
                    u = np.random.random()
                    xi[i,j] = -math.sqrt(3.) if u < 1/6 else (math.sqrt(3.) if u > 5/6 else 0.)
                    zeta[i,j] = 1. if np.random.random() >= .5 else -1.
                else:
                    xi[i,j] = np.random.normal()*math.sqrt(dt)
        if method == 0:
            s,err = mt_step(s,xi,zeta,dt,anisotropy,damping,theta,project)
        else:
            s,err = heun_step(s,xi,dt,anisotropy,damping,theta)
        maximum = max(maximum,err)
        if step % stride == 0:
            result[step//stride] = s
    return result,maximum


@njit(cache=True, parallel=True)
def ensemble(initials,seeds,dt,steps,stride,anisotropy,damping,theta,method=0,project=True):
    result = np.empty((len(initials),steps//stride+1,initials.shape[1],3),np.float64)
    errors = np.empty(len(initials),np.float64)
    for i in prange(len(initials)):
        result[i],errors[i] = trajectory(initials[i],seeds[i],dt,steps,stride,
                                        anisotropy,damping,theta,method,project)
    return result,errors
