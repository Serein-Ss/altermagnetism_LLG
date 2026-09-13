"""Fail-closed contract and source-family checks for Gate-F development only."""
import hashlib
import json
import math
from pathlib import Path
import yaml
from scripts.validation.zero_field_certificates import verify


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def audit_records(rows):
    seen, identities = {}, set()
    if not rows:
        raise ValueError('empty manifest')
    for row in rows:
        if row['split'] not in ('train', 'development'):
            raise ValueError('Gate-F is development, not final blind testing')
        for field in ('initial_id', 'source_family_id', 'parent_trajectory_id'):
            if not row.get(field):
                raise ValueError('missing provenance: '+field)
        for field in ('initial_id', 'source_family_id', 'parent_trajectory_id', 'source_chain_id'):
            value = row.get(field)
            if value is None and field == 'source_chain_id' and row['initial_type'] != 'equilibrium_pool':
                continue
            if not value:
                raise ValueError('missing provenance: '+field)
            key = (field, value)
            if key in seen and seen[key] != row['split']:
                raise ValueError('split leakage: '+str(key))
            seen[key] = row['split']
        identity = (row['initial_id'], row['noise_id'])
        if identity in identities:
            raise ValueError('duplicate initial/noise path')
        identities.add(identity)
    return True


def load_contract(path):
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text())
    if c.get('schema') != 'gate_f_contract_v1' or c.get('status') != 'frozen' or c.get('production_enabled') is not False:
        raise ValueError('only frozen development contracts accepted; production remains disabled')
    required = ('contract_id', 'theta', 'dt', 'frames', 'save_dt', 'preparation',
                'initial_manifest', 'data_manifest', 'code_sha256', 'certificates',
                'resource_budget', 'training_seeds', 'latent_seeds', 'bootstrap_seed',
                'feature_scale_floor', 'condition_scale_floor', 'primary_feature_indices', 'comparison_family')
    if any(c.get(k) is None for k in required) or c.get('pending'):
        raise ValueError('unresolved contract fields')
    if len(c['theta']) != 3 or len(set(c['theta'])) != 3 or any(not math.isfinite(t) or t <= 0 for t in c['theta']):
        raise ValueError('three measured positive temperatures required')
    if len(set(c['training_seeds'])) != 3 or c['frames'] not in (101, 201):
        raise ValueError('three training seeds and 101/201 frames required')
    if len(c['latent_seeds']) < 16 or len(set(c['latent_seeds'])) != len(c['latent_seeds']) or c.get('integration_steps') != [16, 32, 64, 128]:
        raise ValueError('>=16 distinct latent seeds and registered step comparison required')
    if c['bootstrap_replicates'] < 2000 or c['max_looks'] != 2 or c['simultaneous_confidence_per_look'] != .975:
        raise ValueError('two-look error budget must not be relaxed')
    if c['model']['alpha'] != .05 or c['model']['orientation'] != '100' or c['model']['sizes'] != [16, 32] or c['model']['periodic'] != [True, True]:
        raise ValueError('this entry supports the initial 100 L16/32 contract only')
    if c['model']['thermal_model_id'] != 'gomonay_easy_axis_K_DW' or not math.isclose(c['model']['reduced']['K_DW'], .047/11.1):
        raise ValueError('wrong thermal model')
    for key, value in {'J1': 1., 'J2': 1.88/11.1, 'J_tilde': .8/11.1, 'K_SW': 0.}.items():
        if not math.isclose(c['model']['reduced'][key], value):
            raise ValueError('changed Hamiltonian requires a new protocol')
    if c['noise_per_initial'] != 16 or set(c['initial_types']) != {'ground_perturbed', 'random_sphere', 'equilibrium_pool'}:
        raise ValueError('only initial 1152-path contract supported; supplement must be separately registered')
    for name in ('dt', 'save_dt', 'condition_scale_floor'):
        if not math.isfinite(c[name]) or c[name] <= 0:
            raise ValueError('positive finite '+name+' required')
    if any(c['training'].get(k) is None for k in ('epochs', 'learning_rate', 'gradient_accumulation')):
        raise ValueError('training budget not frozen')
    for key in ('epochs', 'gradient_accumulation'):
        if not isinstance(c['training'][key], int) or c['training'][key] < 1:
            raise ValueError('positive integer training '+key+' required')
    if not math.isfinite(c['training']['learning_rate']) or c['training']['learning_rate'] <= 0:
        raise ValueError('positive finite learning rate required')
    for filename, sha in c['code_sha256'].items():
        if digest(filename) != sha:
            raise ValueError('code changed: '+filename)
    if not c['code_sha256'] or set(c['certificates']) != {'P1', 'P2'}:
        raise ValueError('code bindings and P1/P2 certificates required')
    for spec in c['certificates'].values():
        if not spec['bindings'] or spec['bindings'].get('thermal_model_id') != c['model']['thermal_model_id']:
            raise ValueError('thermal model certificate binding missing')
        for key, value in {'model': c['model'], 'theta': c['theta'], 'dt': c['dt'],
                           'save_dt': c['save_dt'], 'frames': c['frames']}.items():
            if spec['bindings'].get(key) != value:
                raise ValueError('certificate does not bind actual Gate-F conditions: '+key)
        verify(spec['path'], spec['scope'], spec['bindings'])
    for key in ('initial_manifest', 'data_manifest', 'resource_budget'):
        if digest(c[key]['path']) != c[key]['sha256']:
            raise ValueError(key+' hash mismatch')
    budget = json.loads(Path(c['resource_budget']['path']).read_text())
    for key in ('max_optimizer_steps', 'walltime_seconds', 'max_output_bytes'):
        if not isinstance(budget.get(key), (int, float)) or budget[key] <= 0:
            raise ValueError('missing resource bound: '+key)
    c['verified_budget'] = budget
    c['contract_sha256'] = digest(path)
    return c


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('contract')
    args = p.parse_args()
    print(json.dumps(load_contract(args.contract), indent=2))
