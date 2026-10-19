"""Open triangular-lattice solid-angle diagnostic; no periodic closure or rounding Q.

Geometric charge: Berg and Luscher, Nucl. Phys. B 190, 412 (1981),
doi:10.1016/0550-3213(81)90568-X. Here CCW xy triangles define positive orientation.
"""
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'miltat_1ns_20260919/results'


def triangles(positions, spacing=.251):
    p = np.asarray(positions).reshape(-1, 3)[:, :2]
    tri = Delaunay(p).simplices.copy()
    q = p[tri]
    edges = np.linalg.norm(q-np.roll(q, 1, axis=1), axis=-1)
    # Delaunay's convex hull bridges zig-zag free boundaries. Do not add those faces.
    tri = tri[np.all(np.isclose(edges, spacing, rtol=1e-5, atol=1e-7), axis=1)]
    q = p[tri]; u, v = q[:, 1]-q[:, 0], q[:, 2]-q[:, 0]
    negative = u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0] < 0
    tri[negative] = tri[negative][:, [0, 2, 1]]
    if not len(tri):
        raise ValueError('No elementary triangular faces found')
    return tri


def charge(spins, tri):
    s = np.asarray(spins, dtype=float).reshape(-1, 3)
    norms = np.linalg.norm(s, axis=-1)
    if not np.isfinite(s).all() or np.max(abs(norms-1)) > 1e-5:
        raise ValueError('Expected finite unit spins')
    s = s/norms[:, None]
    a, b, c = np.moveaxis(s[tri], 1, 0)
    numerator = np.sum(a*np.cross(b, c), axis=-1)
    denominator = 1 + np.sum(a*b+b*c+c*a, axis=-1)
    ambiguous = np.hypot(numerator, denominator) < 1e-10
    omega = 2*np.arctan2(numerator, denominator)
    return dict(Q=float(omega.sum()/(4*np.pi)), ambiguous_faces=int(ambiguous.sum()),
                max_face_solid_angle=float(abs(omega).max()), stored_norm_error=float(abs(norms-1).max()))


def main():
    out = HERE/'results'; out.mkdir(exist_ok=True)
    report = dict(scientific_certification=False, boundary='open, no artificial closure',
                  sign='positive CCW xy solid angles',
                  source='https://doi.org/10.1016/0550-3213(81)90568-X', phases={})
    for phase in ['preparation', 'thermal']:
        rows = []
        with h5py.File(SOURCE/f'{phase}.h5') as h:
            assert h.attrs['complete']
            tri = triangles(h['positions_nm'][:])
            for j, t in enumerate(h['time_ps'][:]):
                rows.append(dict(time_ps=float(t), **charge(h['spins'][j, 0], tri)))
            assert len(rows) == h['spins'].shape[0]
        with (out/f'{phase}_topology.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        qs = np.array([r['Q'] for r in rows])
        report['phases'][phase] = dict(frames=len(rows), triangles=len(tri), Q_min=float(qs.min()),
            Q_max=float(qs.max()), Q_initial=float(qs[0]), Q_final=float(qs[-1]),
            max_saved_frame_Q_jump=float(abs(np.diff(qs)).max()),
            ambiguous_faces=sum(r['ambiguous_faces'] for r in rows),
            max_stored_norm_error=max(r['stored_norm_error'] for r in rows))
        print(phase, report['phases'][phase], flush=True)
    report['remaining_requirements'] = ['Independent time-step sensitivity and initial preparations',
        'DMI source sign convention and gamma convention provenance',
        'Finite fixed-window versus published moving-window control',
        'Independent parent-level train/dev/test splits; one path is not a training library']
    report['caution'] = 'Open-boundary Q need not be integer; saved-frame diagnostic cannot exclude unsaved events.'
    (out/'topology.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
