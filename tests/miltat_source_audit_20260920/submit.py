import hashlib
import subprocess
from tests.bauer_two_stage_20260919.common import dump
from .boundary import HERE,ROOT


def main():
    target=HERE/'submission.json'
    if target.exists():raise RuntimeError('Submission already recorded')
    (HERE/'logs').mkdir(exist_ok=True)
    sources=list(HERE.glob('*.py'))+list(HERE.glob('*.yaml'))+[HERE/'job.sh']
    sources += [ROOT/p for p in ['tests/literature_preflight_20260917/models.py',
        'tests/literature_preflight_20260917/config.yaml','scripts/core/reduced_llg.py',
        'scripts/core/cell_hamiltonian.py','scripts/core/literature_config.py',
        'tests/physical_audit_20260917/cold.py','tests/miltat_dataset_20260920/topology.py',
        'tests/bauer_two_stage_20260919/common.py','tests/miltat_1ns_20260919/results/preparation_final.pt']]
    (HERE/'submission.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(set(sources))))
    cmd=['sbatch','--parsable','--job-name=zrs-mag','--partition=rtx4090','--gres=gpu:1',
         '--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=16G','--time=08:00:00','--array=0-4',
         f'--chdir={ROOT}',f'--output={HERE}/logs/relax_%A_%a.out',f'--error={HERE}/logs/relax_%A_%a.err',str(HERE/'job.sh')]
    job=subprocess.run(cmd,check=True,capture_output=True,text=True).stdout.strip().split(';')[0]
    dump(target,dict(job=job,array_shapes=[[32,18],[64,38],[96,56],[128,74],[200,116]],monitor=False,
                     scope='zero_temperature_boundary_control_not_training_data'))
    print(job,flush=True)


if __name__=='__main__':main()
