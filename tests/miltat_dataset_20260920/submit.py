import argparse
import hashlib
import json
import subprocess
from tests.bauer_two_stage_20260919.common import dump
from .convergence import HERE, ROOT


def report():
    rows = []
    for index in range(4):
        p = HERE/'convergence'/f'branch_{index}'/'metrics.json'
        if not p.exists():
            rows.append(dict(branch=index, status='NOT_COMPLETED')); continue
        value = json.loads(p.read_text()); last = value['rows'][-1]
        rows.append(dict(branch=index, status='COMPLETED', final=last, seconds=value['seconds']))
    dump(HERE/'results'/'convergence_status.json', dict(branches=rows, scientific_certification=False,
        note='One parent, four coupled-noise replicas. Inspect all times in branch metrics; no automated release.'))


def main():
    target = HERE/'submission.json'
    if target.exists(): raise RuntimeError('Submission already recorded')
    (HERE/'logs').mkdir(exist_ok=True)
    sources = list(HERE.glob('*.py'))+list(HERE.glob('*.yaml'))+[HERE/'job.sh']
    sources += [ROOT/p for p in ['tests/literature_preflight_20260917/models.py',
        'tests/literature_preflight_20260917/config.yaml', 'scripts/core/reduced_llg.py',
        'scripts/core/cell_hamiltonian.py','scripts/core/literature_config.py',
        'tests/bauer_two_stage_20260919/common.py',
        'tests/miltat_1ns_20260919/results/preparation_final.pt']]
    (HERE/'submission.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n'
                                              for p in sorted(set(sources))))
    jobs = {}
    for phase, partition, wall, array, deps in [
        ('smoke','rtx4090','01:00:00',None,[]),
        ('convergence','rtx4090','08:00:00','0-3',['smoke']),
        ('report','fat','01:00:00',None,['smoke','convergence'])]:
        cmd = ['sbatch','--parsable','--job-name=zrs-mag',f'--partition={partition}',
            '--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=16G',f'--time={wall}',
            '--kill-on-invalid-dep=yes',f'--chdir={ROOT}',
            f'--output={HERE}/logs/{phase}_%A_%a.out',f'--error={HERE}/logs/{phase}_%A_%a.err']
        if partition=='rtx4090': cmd += ['--gres=gpu:1']
        if array: cmd += [f'--array={array}']
        if deps: cmd += ['--dependency='+('afterany:' if phase=='report' else 'afterok:')+':'.join(jobs[k] for k in deps)]
        cmd += [str(HERE/'job.sh'), phase]
        jobs[phase] = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(';')[0]
        dump(target, dict(jobs=jobs, monitor=False, scientific_certification=False))
        print(phase, jobs[phase], flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--report',action='store_true'); a=p.parse_args()
    report() if a.report else main()
