import hashlib
import json
import subprocess
from tests.bauer_two_stage_20260919.common import dump
from .generate import HERE,ROOT


def main():
    target=HERE/'submission.json'
    if target.exists():raise RuntimeError('Submission already recorded')
    if not json.loads((HERE/'release.json').read_text())['approved_for_generation']:
        raise RuntimeError('Dataset scope has not been released')
    (HERE/'logs').mkdir(exist_ok=True)
    sources=list(HERE.glob('*.py'))+list(HERE.glob('*.yaml'))+[HERE/'layout.json',HERE/'release.json',HERE/'job.sh']
    sources+=[ROOT/p for p in ['scripts/core/reduced_llg.py','scripts/core/cell_hamiltonian.py',
        'scripts/core/literature_config.py','tests/literature_preflight_20260917/models.py',
        'tests/literature_preflight_20260917/config.yaml','tests/physical_audit_20260917/cold.py',
        'tests/bauer_two_stage_20260919/common.py']]
    (HERE/'submission.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(set(sources))))
    jobs={}
    phases=[('smoke','rtx4090','01:00:00',None,[]),
        ('cold','rtx4090','08:00:00','0-2',['smoke']),
        ('cold_gate','fat','01:00:00',None,['cold']),
        ('parent','rtx4090','12:00:00','0-59',['cold_gate']),
        ('finalize','fat','04:00:00',None,['parent']),
        ('status','fat','01:00:00',None,['smoke','cold','cold_gate','parent','finalize'])]
    for phase,partition,wall,array,deps in phases:
        cmd=['sbatch','--parsable','--job-name=zrs-mag',f'--partition={partition}',
             '--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=16G',f'--time={wall}',
             f'--chdir={ROOT}','--kill-on-invalid-dep=yes',
             f'--output={HERE}/logs/{phase}_%A_%a.out',f'--error={HERE}/logs/{phase}_%A_%a.err']
        if partition=='rtx4090':cmd+=['--gres=gpu:1']
        if array:cmd+=[f'--array={array}']
        if deps:cmd+=['--dependency='+('afterany:' if phase=='status' else 'afterok:')+':'.join(jobs[k] for k in deps)]
        cmd+=[str(HERE/'job.sh'),phase]
        jobs[phase]=subprocess.run(cmd,check=True,capture_output=True,text=True).stdout.strip().split(';')[0]
        dump(target,dict(jobs=jobs,monitor=False,parents=60,branches=480,cold_gate_required=True,
                        scientific_certification=False))
        print(phase,jobs[phase],flush=True)


if __name__=='__main__':main()
