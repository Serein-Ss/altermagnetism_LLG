"""Unit conversion is an input boundary; runtime gets only reduced/numerics."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import pint
from scipy import constants
import yaml

UREG=pint.UnitRegistry()
UREG.define(f"mu_B = {constants.physical_constants['Bohr magneton'][0]} * joule / tesla")
UREG.define(f"mRy = {constants.physical_constants['Rydberg constant times hc in J'][0]*.001} * joule")


@dataclass(frozen=True)
class RuntimeConfig:
    reduced: dict
    numerics: dict


def quantity(row):
    return UREG.Quantity(row['value'], row['unit'])


def conversions(document):
    """Independently recompute the published-to-reduced parameters in YAML."""
    paper=document['paper_id']; source=document['source_parameters']
    if paper == 'nishino_miyashita_2015':
        return {k:float(quantity(source[k]).m_as('dimensionless')) for k in ['field_h','moment','alpha_A','D_B']}
    if paper == 'bauer_2011':
        return {k:float(quantity(source[k]).m_as('dimensionless')) for k in ['exchange','anisotropy','theta','alpha']}
    if paper in ('gomonay_2024','hirst_mn2au_2022'):
        e0=quantity(document['reduction']['energy_scale'])
        names=['J1','J2','J_tilde','K_SW','K_DW'] if paper=='gomonay_2024' else ['J1','J2','J3','J4','J0_same','J0_inter','d_z','d_x']
        out={name:float((quantity(source[name])/e0).m_as('dimensionless')) for name in names}
        if paper=='hirst_mn2au_2022':
            out['c_over_a']=float((quantity(source['c'])/quantity(source['a'])).m_as('dimensionless'))
            for label in ['theta_300','theta_1000','theta_1200','theta_TN']:
                temp=quantity(source[label])
                out[label]=float((constants.Boltzmann*UREG.joule/UREG.kelvin*temp/e0).m_as('dimensionless'))
        return out
    if paper=='laliena_crnb3s6_2020':
        a,d,k,ms=[quantity(source[name]) for name in ['A','D','K','M_s']]
        q0=d/(2*a); b0=d*d/(2*a*ms)
        return {'q0_a':float((q0*quantity(source['a'])).m_as('dimensionless')),
                'kappa':float((4*a*k/(d*d)).m_as('dimensionless')),
                'h_y':float((quantity(source['B_y'])/b0).m_as('dimensionless')),
                'alpha':float(quantity(source['alpha']).m_as('dimensionless')),
                'beta':float(quantity(source['beta']).m_as('dimensionless'))}
    raise ValueError('unknown paper_id')


def atomic_scales(energy, moment, gamma, length):
    """E0, magnetic moment mu_ref (not vacuum permeability), gamma in 1/(s T)."""
    e=quantity(energy); m=quantity(moment); g=quantity(gamma); a=quantity(length)
    if min(e.m_as('joule'),m.m_as('joule/tesla'),g.m_as('1/(second*tesla)'),a.m_as('meter'))<=0:
        raise ValueError('reference scales must be positive')
    return {'energy_J':float(e.m_as('joule')), 'field_T':float((e/m).m_as('tesla')),
            'time_s':float((m/(g*e)).m_as('second')),
            'temperature_K':float((e/(constants.Boltzmann*UREG.joule/UREG.kelvin)).m_as('kelvin')),
            'length_m':float(a.m_as('meter'))}


def read_document(path):
    document=yaml.safe_load(Path(path).read_text())
    for key in ['paper_id','source_parameters','reduction','reduced','numerics','provenance']:
        if key not in document: raise ValueError('missing '+key)
    expected=conversions(document)
    for key,value in expected.items():
        actual=document['reduced'][key]
        if not math.isfinite(actual) or not math.isclose(actual,value,rel_tol=2e-6,abs_tol=1e-10):
            raise ValueError(f'{key}: configured {actual}, independently converted {value}')
    def numbers_only(value):
        if isinstance(value,dict):
            if any(k in value for k in ['unit','value_si','source_parameters']):
                raise ValueError('units/source values cannot enter runtime physics')
            for v in value.values(): numbers_only(v)
        elif isinstance(value,list):
            for v in value: numbers_only(v)
        elif isinstance(value,(int,float)) and not math.isfinite(value):
            raise ValueError('nonfinite runtime parameter')
        elif isinstance(value,str) and value not in ['gilbert','bauer_ll','free_moments','candidate','blocked']:
            raise ValueError('unexpected runtime physical string '+value)
    numbers_only(document['reduced'])
    return document


def load_runtime(path):
    doc=read_document(path)
    return RuntimeConfig(deepcopy(doc['reduced']),deepcopy(doc['numerics']))


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()
