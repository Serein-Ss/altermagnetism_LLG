import hashlib
import subprocess
from tests.bauer_two_stage_20260919.common import dump
from .run import HERE,ROOT,OLD


def main():
    target=HERE/'submission.json'
    if target.exists():raise RuntimeError('Existing submission record')
    (HERE/'logs').mkdir(exist_ok=True)
    sources=list(HERE.glob('*.py'))+[HERE/'job.sh',OLD/'convergence.yaml',OLD/'topology.py']
    sources+=[ROOT/p for p in ['scripts/core/reduced_llg.py','scripts/core/cell_hamiltonian.py',
        'scripts/core/literature_config.py','tests/literature_preflight_20260917/models.py',
        'tests/literature_preflight_20260917/config.yaml','tests/bauer_two_stage_20260919/common.py',
        'tests/miltat_1ns_20260919/results/preparation_final.pt']]
    (HERE/'submission.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(set(sources))))
    command=['sbatch','--parsable','--job-name=zrs-mag','--partition=rtx4090','--gres=gpu:1',
        '--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=16G','--time=08:00:00','--array=0-3',
        f'--chdir={ROOT}',f'--output={HERE}/logs/refine_%A_%a.out',f'--error={HERE}/logs/refine_%A_%a.err',str(HERE/'job.sh')]
    job=subprocess.run(command,check=True,capture_output=True,text=True).stdout.strip().split(';')[0]
    dump(target,dict(job=job,monitor=False,new_dt_fs=.3125,reuses_frozen_noise=True,scientific_certification=False))
    print(job,flush=True)


if __name__=='__main__':main()
