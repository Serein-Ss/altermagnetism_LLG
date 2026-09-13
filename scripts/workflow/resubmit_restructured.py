"""One-shot resubmission of the eight explicitly stopped jobs; no monitoring."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT/'logs/restructure'


def main():
    record = LOGS/'resubmission_20260913.json'
    if record.exists():
        raise FileExistsError('submission record exists; inspect instead of duplicating')
    check = subprocess.run([sys.executable, str(ROOT/'scripts/workflow/resume_after_restructure.py'), '--check'],
                           check=True, capture_output=True, text=True)
    validated = json.loads(check.stdout)
    state = dict(status='submitting', monitor=False, jobs=[], preflight=validated,
                 adapter_sha256=hashlib.sha256((ROOT/'scripts/workflow/resume_after_restructure.py').read_bytes()).hexdigest())
    with record.open('x') as stream: json.dump(state,stream,indent=2)
    def save():
        temp=record.with_suffix('.json.tmp')
        with temp.open('w') as stream: json.dump(state,stream,indent=2)
        temp.replace(record)
    def submit(index, old, minutes, dependency=None):
        cmd=['sbatch','--parsable','--time='+str(minutes),
             '--kill-on-invalid-dep=yes','--output='+str(LOGS/f'resumed_{index}_%j.log')]
        if dependency:cmd+=['--dependency=afterany:'+dependency]
        cmd += [str(ROOT/'slurm/resume_after_restructure.sbatch'),str(ROOT),str(index)]
        response=subprocess.run(cmd,capture_output=True,text=True,check=True)
        job=response.stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('unexpected submission response: '+response.stdout)
        state['jobs'].append(dict(old_job=old,job_id=job,index=index,command=cmd));save()
        return job
    a=submit(0,'669197_14',11742)
    b=submit(1,'669337',2506)
    c=submit(2,'669338',4951)
    d=submit(3,'669214_31',15,a)
    e=submit(4,'669218_35',15,d)
    f=submit(5,'669222_39',1194,e)
    submit(6,'669226_43',2052,f)
    old=ROOT/'conf/archive/frozen/bauer_revision4/20260913_023043/manifest.json'
    m=json.loads(old.read_text())
    for row in m['reference_runs']:
        row['config']=row['config'].replace('/conf/frozen/','/conf/archive/frozen/')
    m['output']=str(ROOT/'assets/literature_reproduction/bauer_2011/runs/bauer_revision4/20260913_023043')
    review_manifest=ROOT/'conf/literature_reproduction/campaigns/bauer/review_after_restructure.json'
    with review_manifest.open('x') as stream:json.dump(m,stream,indent=2)
    cmd=['sbatch','--parsable','--job-name=zrs-mag','--partition=fat','--nodes=1','--ntasks=1',
         '--cpus-per-task=1','--mem=16G','--time=120','--kill-on-invalid-dep=yes',
         '--dependency=afterany:'+':'.join([a,b,c]),'--output='+str(LOGS/'review_%j.log'),
         str(ROOT/'slurm/review_after_restructure.sbatch'),str(ROOT),str(review_manifest)]
    response=subprocess.run(cmd,capture_output=True,text=True,check=True)
    job=response.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError(response.stdout)
    state['jobs'].append(dict(old_job='669339',job_id=job,index='review',command=cmd))
    state['status']='submitted_not_monitored';save()
    print(json.dumps([dict(old_job=r['old_job'],job_id=r['job_id']) for r in state['jobs']],indent=2))


if __name__=='__main__':main()
