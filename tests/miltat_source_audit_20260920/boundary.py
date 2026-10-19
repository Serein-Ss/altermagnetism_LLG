"""Same-state truncation of full-dipole fields; NOT a dynamic boundary certificate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from tests.literature_preflight_20260917.models import build
from tests.bauer_two_stage_20260919.common import dump

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def centered_crop(shape, target):
    if any(n < m or (n-m)%2 for n,m in zip(shape,target)):
        raise ValueError('Target must fit with centered integer cell offsets')
    return tuple(slice((n-m)//2, (n+m)//2) for n,m in zip(shape,target))


def main():
    torch.set_num_threads(2)
    path = ROOT/'tests/miltat_1ns_20260919/results/preparation_final.pt'
    source = yaml.safe_load((ROOT/'tests/literature_preflight_20260917/config.yaml').read_text())
    source['systems']['miltat']['physics']['temperature_K'] = 12.5
    state = torch.load(path,map_location='cpu',weights_only=False)['spins'].to(torch.float64)
    full, reduced, scales = build(source,'miltat')
    rows = []
    with torch.inference_mode():
        positions = full.positions(state)*reduced['a_nm']
        center = positions[state[0,...,2] >= .5].mean(0)
        field_full = full.field(state)
        for shape in [(32,18),(64,38),(96,56),(128,74),(200,116)]:
            crop = centered_crop(full.shape,shape)
            index = (slice(None),*crop,slice(None),slice(None))
            s = state[index].clone(); ref = field_full[index]
            p = positions[crop]
            model, _, _ = build(source,'miltat',shape=shape)
            field = model.field(s)
            distance = (p-center).norm(dim=-1)
            regions = {'core_3nm': distance <= 3., 'central_5nm': distance <= 5.}
            row = dict(cell_shape=list(shape),spins=int(s.numel()/3),regions={})
            lo, hi = p.reshape(-1,3).amin(0), p.reshape(-1,3).amax(0)
            row['xy_extent_nm'] = (hi-lo)[:2].tolist()
            row['core_center_to_box_edge_nm'] = float(torch.minimum(center-lo,hi-center)[:2].min())
            for name, mask in regions.items():
                diff = (field-ref)[0][mask]; f = ref[0][mask]
                torque = torch.linalg.cross(s, field)[0][mask]
                ref_torque = torch.linalg.cross(s,ref)[0][mask]
                row['regions'][name] = dict(sites=int(mask.sum()),
                    field_rms_error=float(diff.square().mean().sqrt()),
                    field_relative_rms=float(diff.square().mean().sqrt()/f.square().mean().sqrt().clamp_min(1e-15)),
                    torque_rms_error=float((torque-ref_torque).square().mean().sqrt()),
                    reference_torque_rms=float(ref_torque.square().mean().sqrt()),
                    whole_region_fits=bool(row['core_center_to_box_edge_nm'] >= (3. if name=='core_3nm' else 5.)))
            rows.append(row)
    result = dict(source=str(path.relative_to(ROOT)), initial_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_protocol='50ps_zero_temperature_preparation_from_strongly_perturbed_texture',
                  reference_shape=list(full.shape), rows=rows, scales=scales, reduced=reduced,
                  scientific_certification=False,
                  notes=['Truncation diagnostic only: cropped states are NOT released as independent training data.',
                         'Long-range dipoles retained in every candidate, but outside spins removed.',
                         'Small instantaneous core field errors cannot exclude later boundary forces or edge escape.',
                         'Missing part of a requested region is explicitly marked whole_region_fits=false.',
                         'Need size-dependent preparations, radius/energy stability, and finite-temperature window controls.'])
    dump(HERE/'results'/'boundary_fields.json',result)
    for row in rows: print(json.dumps(row),flush=True)


if __name__ == '__main__': main()
