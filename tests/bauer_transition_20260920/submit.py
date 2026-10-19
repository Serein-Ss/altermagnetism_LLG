"""Submit frozen dependent jobs once; do not monitor training."""
import hashlib
import subprocess
from tests.bauer_two_stage_20260919.common import dump
from .run import HERE, ROOT


def main():
    target = HERE/'submission.json'
    if target.exists():
        raise RuntimeError('Submission already recorded; inspect before resubmitting')
    (HERE/'logs').mkdir(exist_ok=True)
    sources = list(HERE.glob('*.py'))+list(HERE.glob('*.yaml'))+[HERE/'job.sh']
    for folder in ['bauer_closed_loop_20260919', 'bauer_two_stage_20260919', 'bauer_publication_20260920',
                   'mdgen_path_refactor_20260918', 'bauer_active_data_20260918', 'bauer_fig2_10ns_20260917']:
        sources += list((ROOT/'tests'/folder).glob('*.py'))+list((ROOT/'tests'/folder).glob('*.yaml'))
    sources += [ROOT/'scripts/model/sphere.py', ROOT/'scripts/model/gate_f.py']
    sources += [ROOT/f'tests/bauer_two_stage_20260919/runs/anchored_{s}/best.pt' for s in [11,22,33]]
    (HERE/'submission.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n'
                                              for p in sorted(set(sources))))
    ids = {}
    phases = [('smoke','rtx4090','01:00:00',None,[]),
              ('train','rtx4090','12:00:00','0-5',['smoke']),
              ('evaluate','rtx4090','12:00:00','0-5',['train']),
              ('report','fat','01:00:00',None,['smoke','train','evaluate'])]
    for phase, partition, wall, array, dependencies in phases:
        cmd = ['sbatch','--parsable','--job-name=zrs-mag',f'--partition={partition}',
               '--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=16G',f'--time={wall}',
               '--kill-on-invalid-dep=yes',f'--chdir={ROOT}',
               f'--output={HERE}/logs/{phase}_%A_%a.out',f'--error={HERE}/logs/{phase}_%A_%a.err']
        if partition=='rtx4090': cmd += ['--gres=gpu:1']
        if array: cmd += [f'--array={array}']
        if dependencies:
            cmd += ['--dependency='+('afterany:' if phase=='report' else 'afterok:')+':'.join(ids[d] for d in dependencies)]
        cmd += [str(HERE/'job.sh'), phase]
        ids[phase] = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(';')[0]
        dump(target, dict(jobs=ids, monitor=False, array=['path_11','path_22','path_33','transition_11','transition_22','transition_33'],
                          scientific_certification=False))
        print(phase, ids[phase], flush=True)


if __name__ == '__main__': main()
