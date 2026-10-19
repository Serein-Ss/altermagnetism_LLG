"""Unoriented anisotropy-axis projection; not applicable to an applied field axis."""
from tests.experiments.local_equivariant_transition import LocalEquivariantTransition


class DirectorTransition(LocalEquivariantTransition):
    def forward(self, x, tau, initial, theta, lag, axis):
        positive = super().forward(x, tau, initial, theta, lag, axis)
        negative = super().forward(x, tau, initial, theta, lag, -axis)
        return (positive+negative)/2
