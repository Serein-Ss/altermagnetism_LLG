"""Freeze a bounded pilot manifest, CPU I/O check, submit exactly three GPU tasks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]


def main():
    if (BASE/'submission.json').exists():
        raise RuntimeError('Submission record already exists; inspect before any resubmission')
    files=[BASE/x for x in ('config.yaml','models.py','run.py','test_models.py','pilot.sbatch','submit.py')]
    files += [ROOT/x for x in ('scripts/core/cell_hamiltonian.py','scripts/core/reduced_llg.py',
              'scripts/core/literature_config.py','scripts/literature/hirst_mn2au_2022/model.py')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (BASE/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    (BASE/'logs').mkdir(exist_ok=True)
    for name in ('bauer','miltat','hirst'):
        with (BASE/'logs'/f'{name}_cpu_smoke.log').open('x') as log:
            subprocess.run([sys.executable,'-m','tests.literature_preflight_20260917.run',
                            '--system',name,'--device','cpu','--smoke'],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['bash','-n',str(BASE/'pilot.sbatch')],check=True)
    result=subprocess.run(['sbatch','--parsable','--partition=rtx4090',str(BASE/'pilot.sbatch')],
                           cwd=ROOT,text=True,capture_output=True,check=True)
    job=result.stdout.strip().split(';')[0]
    record={'job_id':job,'submitted_unix':time.time(),'name':'zrs-mag','partition':'rtx4090',
            'tasks':{'0':'bauer','1':'miltat','2':'hirst'},'state':'SUBMITTED_NOT_MONITORED',
            'scientific_status':'NOT_CERTIFIED','production_enabled':False,
            'training_jobs_submitted':0,'source_hashes':hashes}
    (BASE/'submission.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
