"""Bauer-only adapter to the shared Gate-F backbone; not a physics certificate.

The equation is bound to this model class, not an interchangeable Gilbert label.
Condition columns are theta and lambda. Legacy models/checkpoints are unchanged.
"""
from types import SimpleNamespace
import torch
from torch import nn
from scripts.model.gate_f import CellGraph, GateFNet


def open_chain_graph(length, exchange=1., anisotropy=.1):
    if not isinstance(length, int) or length < 2 or exchange <= 0 or anisotropy < 0:
        raise ValueError('finite ferromagnetic open chain with at least two sites required')
    cell = SimpleNamespace(shape=(length, 1), basis=1, periodic=(False, False),
                           anisotropy=(0., 0., anisotropy),
                           templates=[(0, 0, (1, 0), exchange)])
    return CellGraph(cell)


class BauerGateFNet(GateFNet):
    equation_convention = 'bauer_ll'
    damping_parameter = 'lambda'
    condition_names = ('theta', 'lambda')

    def forward(self, state, tau, initial, condition, time, graph, axis, *, tangent=True):
        if graph.shape[0] != 1 or graph.shape[2] != 1 or any(graph.periodic):
            raise ValueError('Bauer model requires A=1, Ny=1 and open boundaries')
        return super().forward(state, tau, initial, condition, time, graph, axis, tangent=tangent)


def build_model(kind, **kwargs):
    if kind not in ('rfm', 'deterministic', 'euclidean', 'autoregressive'):
        raise ValueError('unknown model kind')
    net = BauerGateFNet(**kwargs)
    if kind == 'autoregressive':
        net.log_concentration = nn.Parameter(torch.tensor(0.))
    return net
