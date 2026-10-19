import torch
from tests.miltat_dataset_20260920.convergence import coupled_step


def test_brownian_coupling_not_seed_reuse():
    class Recorder:
        def __init__(self): self.calls=[]
        def step(self, s, dt, dw, method):
            self.calls.append((dt, dw.clone()))
            return s+dw, None
    solver = Recorder()
    increments = [torch.full((1,2,3), float(k)) for k in [1,2,3,4]]
    states = [torch.zeros(1,2,3) for _ in range(3)]
    out = coupled_step(solver, states, .1, increments)
    assert all(torch.equal(s, torch.full_like(s, 10)) for s in out)
    assert [dt for dt, _ in solver.calls] == [.4,.2,.2,.1,.1,.1,.1]
    assert torch.equal(solver.calls[0][1], solver.calls[1][1]+solver.calls[2][1])
