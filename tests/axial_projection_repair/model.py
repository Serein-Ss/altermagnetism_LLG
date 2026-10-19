"""Frozen Bauer transport-field C_K projection, not averaging physical paths."""
import math
import torch
from tests.axial_symmetry_repair.symmetry import rotate


class AxialProjection(torch.nn.Module):
    def __init__(self, base, order):
        super().__init__()
        meta = base.get_extra_state()
        if meta['package'] != 'transition_block_repair' or meta['periodic'] != (False,):
            raise ValueError('audited Bauer open transition package only')
        if order not in (1, 4, 8):
            raise ValueError('registered orders: 1,4,8')
        self.base, self.order = base, order

    def get_extra_state(self):
        return dict(base=self.base.get_extra_state(), projection_order=self.order,
                    symmetry='C_K spin rotations about registered easy axis; not exact continuous SO2')

    def velocity(self, x, tau, initial, times, temperature=None, frame=None):
        if self.order == 1:
            return self.base.velocity(x, tau, initial, times, temperature, frame)
        frame = torch.eye(3, device=x.device, dtype=x.dtype) if frame is None else frame
        body, anchor = x @ frame.T, initial @ frame.T
        value = torch.zeros_like(x)
        for k in range(self.order):
            angle = 2 * math.pi * k / self.order
            v = self.base.velocity(rotate(body, angle) @ frame, tau,
                                   rotate(anchor, angle) @ frame, times, temperature, frame)
            value = value + rotate(v @ frame.T, -angle) @ frame
        return value / self.order
