"""Reduced exchange on explicit unit-cell displacement templates.

State: [batch, *cell_shape, basis, 3]. A template (a,b,delta,J) is
one undirected bond per cell, with energy -J s_a(R).s_b(R+delta).
Periodic images remain distinct bonds, including in very small test cells.
"""
import torch


def shifted(x, delta, periodic):
    """Return x(R+delta); remove missing neighbors on open boundaries."""
    y = torch.roll(x, tuple(-d for d in delta), tuple(range(1, 1+len(delta))))
    if not all(periodic):
        y = y.clone()
        for axis, (d, wrap) in enumerate(zip(delta, periodic), 1):
            if d and not wrap:
                index = [slice(None)] * y.ndim
                index[axis] = slice(-d, None) if d > 0 else slice(None, -d)
                y[tuple(index)] = 0
    return y


class CellHamiltonian:
    def __init__(self, shape, basis, templates, *, periodic, anisotropy=(0., 0., 0.)):
        self.shape = tuple(int(n) for n in shape)
        self.basis = int(basis)
        self.periodic = tuple(periodic)
        if len(self.shape) != len(self.periodic) or min(self.shape) < 2:
            raise ValueError('cell dimensions must be >=2 and match boundaries')
        self.templates = tuple((a, b, tuple(d), float(j)) for a,b,d,j in templates)
        for a,b,d,j in self.templates:
            if not 0 <= a < basis or not 0 <= b < basis or len(d) != len(shape):
                raise ValueError('invalid bond template')
        self.anisotropy = tuple(anisotropy)

    def energy(self, s):
        e = -(s.square()*s.new_tensor(self.anisotropy)).flatten(1).sum(1)
        for a,b,d,j in self.templates:
            e = e-j*(s[...,a,:]*shifted(s[...,b,:],d,self.periodic)).flatten(1).sum(1)
        return e

    def field(self, s):
        field = 2*s*s.new_tensor(self.anisotropy)
        for a,b,d,j in self.templates:
            field[...,a,:] += j*shifted(s[...,b,:],d,self.periodic)
            field[...,b,:] += j*shifted(s[...,a,:],tuple(-v for v in d),self.periodic)
        return field

    def uniform(self, signs, *, batch=1, axis=2, device='cpu', dtype=torch.float64):
        s = torch.zeros((batch,*self.shape,self.basis,3),device=device,dtype=dtype)
        s[...,axis] = s.new_tensor(signs)
        return s
