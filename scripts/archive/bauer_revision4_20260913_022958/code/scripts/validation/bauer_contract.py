"""Revision-4 design validation; deliberately never enables training by itself.

Old Gomonay contracts remain untouched. This validates a Bauer design before
the yet-to-be-frozen P1/P2 evidence and dataset are attached.
"""
import math
import yaml
from pathlib import Path


def validate_design(c):
    if c.get('schema') != 'bauer_gate_f_contract_v1' or c.get('production_enabled') is not False:
        raise ValueError('Bauer development schema required; production disabled')
    eq, model = c['equation'], c['model']
    if eq['convention'] != 'bauer_ll' or eq['damping_parameter'] != 'lambda' or eq['time_unit'] != 'hbar/E0':
        raise ValueError('explicit Bauer LL/lambda/time convention required')
    if not math.isfinite(eq['damping_value']) or eq['damping_value'] <= 0:
        raise ValueError('finite positive lambda required')
    if model['boundary'] != 'open' or model['basis'] != 1 or model['ny'] != 1:
        raise ValueError('single-basis open chain required')
    if not all(math.isfinite(model[k]) for k in ('exchange', 'anisotropy')) or model['exchange'] <= 0 or model['anisotropy'] <= 0:
        raise ValueError('finite FM exchange and easy-axis anisotropy required')
    names = {'external_field', 'field_gradient', 'sot', 'stt', 'current', 'pulse'}
    if set(c['zero_drive']) != names or any(c['zero_drive'][k] != 0 for k in names):
        raise ValueError('all six drives explicitly zero')
    if c['primary_features'] != dict(name='energy_and_magnetization_11_times', dimension=44):
        raise ValueError('registered 44-dimensional feature required')
    if c['generalization']['dimension'] not in ('size', 'temperature'):
        raise ValueError('preselect physical generalization dimension')
    return c


def require_trainable(path):
    c = validate_design(yaml.safe_load(Path(path).read_text()))
    if c.get('status') != 'frozen' or c.get('pending'):
        raise ValueError('unresolved Bauer contract: training blocked')
    # A filled YAML is not evidence. No temporary bypass while P1/P2 is pending.
    raise ValueError('Bauer certificate/data-bound training entry not yet certified')
