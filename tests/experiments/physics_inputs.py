"""Information ablation: retain unweighted geometry, remove Hamiltonian features."""
import torch


class GeometryOnlyGraph:
    def __init__(self, graph):
        self.graph = graph
        self.shape = graph.shape
        self.periodic = graph.periodic
        self.kappa = torch.zeros_like(graph.kappa)

    def aggregate(self, x, weighted=False):
        return torch.zeros_like(x) if weighted else self.graph.aggregate(x)

    def field(self, x, axis):
        return torch.zeros_like(x)
