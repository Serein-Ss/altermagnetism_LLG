"""Reclassify only complete frozen boundary runs, without modifying their artifacts."""
import json
from pathlib import Path
import h5py
from tests.bauer_two_stage_20260919.common import dump
from .geometry import CoreGeometry

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'miltat_source_audit_20260920/relaxation'


def main():
    report=[]
    for index in range(5):
        folder=SOURCE/f'size_{index}'
        # Completion JSON is written only after the trajectory is closed.
        if not (folder/'metrics.json').exists():
            report.append(dict(index=index,status='AWAITING_COMPLETE_RESULT'));continue
        with h5py.File(folder/'trajectory.h5') as h:
            assert h.attrs['complete'] and h.attrs['frames_written']==len(h['time_ps'])
            geometry=CoreGeometry(h['positions_nm'][:])
            rows=[dict(time_ps=float(t),**geometry.measure(h['spins'][j])) for j,t in enumerate(h['time_ps'][:])]
        original=json.loads((folder/'metrics.json').read_text())
        item=dict(index=index,shape=original['shape'],rows=rows,
                  last_torque_rms=original['rows'][-1]['torque_rms'],
                  last_window_energy_span_per_spin=original['last_window_energy_span_per_spin'])
        report.append(item);print(index,rows[-1],flush=True)
    dump(HERE/'results'/'boundary_core_status.json',dict(sizes=report,scientific_certification=False,
        note='Geometry classification for a positive core in negative background, not a topological proof or basin criterion.'))


if __name__=='__main__':main()
