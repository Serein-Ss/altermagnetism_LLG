"""Independent bond algebra, SI/reduced noise conversion, and DMI-sign conjugacy."""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
import pytest
from pint.errors import DimensionalityError
import torch
import yaml
from scipy import constants
from scripts.core.literature_config import UREG, quantity, atomic_scales
from scripts.core.reduced_llg import ReducedLLG
from tests.literature_preflight_20260917.models import build

ROOT = Path(__file__).resolve().parents[2]
SOURCE = yaml.safe_load((ROOT/'tests/literature_preflight_20260917/config.yaml').read_text())


def test_explicit_paper_bond_formula():
    torch.manual_seed(921)
    model, reduced, _ = build(SOURCE, 'miltat', shape=(4,3))
    zero = deepcopy(SOURCE); zero['systems']['miltat']['physics']['D']['value'] = 0.
    without, _, _ = build(zero, 'miltat', shape=(4,3))
    s = torch.randn(2,4,3,2,3,dtype=torch.float64)
    s = s/s.norm(dim=-1,keepdim=True)
    pos = model.positions(s).reshape(-1,3)
    spin = s.reshape(2,-1,3); energy = torch.zeros(2,dtype=torch.float64)
    count = 0
    for i in range(len(pos)):
        for j in range(i+1,len(pos)):
            u = pos[j]-pos[i]
            if abs(float(u.norm())-1) < 1e-12:
                d = reduced['D']*torch.linalg.cross(u, u.new_tensor([0,0,1]))
                energy += (d*torch.linalg.cross(spin[:,i],spin[:,j])).sum(-1)
                count += 1
    assert count > 0 and float(abs(energy).max()) > 1e-5
    torch.testing.assert_close(model.energy(s)-without.energy(s), energy, atol=1e-12, rtol=1e-11)


def test_dmi_sign_conjugacy_with_full_open_dipoles():
    torch.manual_seed(922)
    positive = deepcopy(SOURCE); positive['systems']['miltat']['physics']['D']['value'] *= -1
    m, r, _ = build(SOURCE,'miltat',shape=(5,4))
    p, _, _ = build(positive,'miltat',shape=(5,4))
    s = torch.randn(2,5,4,2,3,dtype=torch.float64); s /= s.norm(dim=-1,keepdim=True)
    rotation = s.new_tensor([-1,-1,1])
    torch.testing.assert_close(m.energy(s),p.energy(s*rotation),atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(m.field(s)*rotation,p.field(s*rotation),atol=1e-11,rtol=1e-11)
    a = ReducedLLG(m,alpha=.5,theta=r['theta'])
    b = ReducedLLG(p,alpha=.5,theta=r['theta'])
    dw = torch.randn_like(s)*.01
    x, _ = a.step(s,.0001,dw,method='heun')
    y, _ = b.step(s*rotation,.0001,dw*rotation,method='heun')
    torch.testing.assert_close(x*rotation,y,atol=1e-12,rtol=1e-11)


def test_physical_and_reduced_noise_variance():
    p = SOURCE['systems']['miltat']['physics']
    energy, moment, gamma = quantity(p['J']), quantity(p['moment']), quantity(SOURCE['gamma_convention'])
    scales = atomic_scales(p['J'],p['moment'],SOURCE['gamma_convention'],p['a'])
    temperature = 12.5*UREG.kelvin; alpha = .5
    kbt = constants.Boltzmann*UREG.joule/UREG.kelvin*temperature
    dt = 2.5*UREG.femtosecond; t0 = scales['time_s']*UREG.second
    b0 = energy/moment
    # Physical white field variance in Tesla^2, not H-field in (A/m)^2.
    variance_b = (2*alpha*kbt/(gamma*moment*dt)).to('tesla**2')
    actual = (variance_b/b0**2*(dt/t0)**2).m_as('dimensionless')
    expected = (2*alpha*(kbt/energy)*(dt/t0)).m_as('dimensionless')
    assert actual == pytest.approx(expected,rel=1e-12)
    mu0 = constants.mu_0*UREG.newton/UREG.ampere**2
    gamma_h = mu0*gamma
    variance_h = (2*alpha*kbt/(gamma_h*mu0*moment*dt)).to('(ampere/meter)**2')
    assert (mu0**2*variance_h).m_as('tesla**2') == pytest.approx(variance_b.m_as('tesla**2'),rel=1e-12)
    assert (1/(gamma*b0)).m_as('second') == pytest.approx(scales['time_s'],rel=1e-12)


def test_wrong_gamma_units_rejected():
    p = SOURCE['systems']['miltat']['physics']
    with pytest.raises(DimensionalityError):
        atomic_scales(p['J'],p['moment'],{'value':2.21e5,'unit':'meter/(ampere*second)'},p['a'])


def test_reduction_magnitudes():
    _, reduced, scales = build(SOURCE,'miltat',shape=(4,3))
    assert math.isclose(reduced['K'], .4/29)
    assert math.isclose(reduced['D'], -1.5/29)
    assert .01 < scales['time_s']*1e12 < .04
    assert 200 < scales['field_T'] < 260
    assert np.isfinite(list(scales.values())).all()
