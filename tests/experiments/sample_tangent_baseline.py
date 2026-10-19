"""Sample-fitted local Gaussian tangent transition, diagnostic only.

Fixed lag; local homogeneous reflection/director-symmetric basis. This is a
restricted conditional distribution, not a replacement for path-law validation.
No exchange, anisotropy, damping or analytic LLG force coefficients are supplied.
"""
import numpy as np


def project(x, v):
    return v - (x*v).sum(-1, keepdims=True)*x


def basis(x, axis):
    left = np.zeros_like(x); right = np.zeros_like(x)
    left[:, 1:] = x[:, :-1]; right[:, :-1] = x[:, 1:]
    neighbor = project(x, left+right)
    director = project(x, (x*axis).sum(-1, keepdims=True)*axis)
    return np.stack([neighbor, np.cross(x, neighbor), director, np.cross(x, director)], -2)


def logarithm(x, y):
    dot = np.clip((x*y).sum(-1, keepdims=True), -1, 1)
    tangent = project(x, y)
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    if np.any((dot < 0) & (norm < 1e-10)):
        raise ValueError('Antipodal short transition has undefined logarithm')
    angle = np.arctan2(norm, dot)
    ratio = np.divide(angle, norm, out=np.ones_like(norm), where=norm>1e-12)
    return tangent*ratio


def exponential(x, v):
    v = project(x, v)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    y = np.cos(norm)*x + np.sinc(norm/np.pi)*v
    return y/np.linalg.norm(y, axis=-1, keepdims=True)


def fit(initial, target, axis):
    features = basis(initial, axis)
    design = features.swapaxes(-1,-2).reshape(-1,4)
    response = logarithm(initial,target).reshape(-1)
    coefficients, _, rank, singular = np.linalg.lstsq(design,response,rcond=None)
    residual = response-design@coefficients
    # Two tangent degrees of freedom per site; MLE scale, not uncertainty estimate.
    sigma = np.sqrt(np.sum(residual**2)/(2*np.prod(initial.shape[:-1])))
    return coefficients, float(sigma), dict(rank=int(rank),singular_values=singular.tolist())


def sample(initial, axis, coefficients, sigma, rng):
    mean = np.einsum('nlkc,k->nlc',basis(initial,axis),coefficients)
    noise = project(initial,rng.normal(size=initial.shape))*sigma
    return exponential(initial,mean+noise)
