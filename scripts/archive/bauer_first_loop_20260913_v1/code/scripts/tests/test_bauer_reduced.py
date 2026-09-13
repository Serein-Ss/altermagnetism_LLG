from pathlib import Path
import torch
from scripts.core.literature_config import load_runtime
from scripts.core.reduced_llg import BondHamiltonian
from scripts.literature.bauer_2011.model import OpenChain
from scripts.literature.bauer_2011.preflight import static_checks

ROOT=Path(__file__).resolve().parents[2]


def test_open_chain_matches_explicit_bonds():
    cfg=load_runtime(ROOT/'conf/literature/bauer_2011.yaml')
    assert static_checks(cfg)['status']=='pass'
    model=OpenChain(cfg.reduced)
    generic=BondHamiltonian([[i,i+1] for i in range(6)],1.,.1)
    state=torch.randn(3,7,3,dtype=torch.float64)
    torch.testing.assert_close(model.field(state),generic.field(state),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(model.energy(state),generic.energy(state),atol=1e-12,rtol=1e-12)
