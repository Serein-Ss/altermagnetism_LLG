"""Verified Bauer open-chain H/(N J); not a general Hamiltonian evaluator."""
def energy(s,kappa=.1):
    return -(s[..., :-1, :]*s[..., 1:, :]).sum(-1).sum(-1)/s.shape[-2]-kappa*(s[..., 2]**2).mean(-1)
