"""Submit the bounded dependency graph once; never poll jobs."""
import json
import subprocess
from tests.paper_work.completion.run import ROOT, load
from scripts.workflow.bauer_campaign import save, digest


def run():
    m=load();path=ROOT/'submission.json'
    if path.exists():raise FileExistsError('Submission record exists; inspect before any resubmission')
    report=dict(status='SUBMITTING',jobs=[],manifest_sha256=digest(ROOT/'manifest.json'))
    save(path,report)
    def submit(label,resource,stage,count=1,dependencies=()):
        cmd=['sbatch','--parsable']
        if count>1:cmd.append(f'--array=0-{count-1}')
        if dependencies:cmd.append('--dependency=afterok:'+':'.join(dependencies))
        cmd += [str(ROOT/f'{resource}.sbatch'),stage]
        result=subprocess.run(cmd,check=True,text=True,capture_output=True)
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError(result.stdout)
        report['jobs'].append(dict(label=label,job_id=job,array_tasks=count,command=cmd))
        save(path,report);print(label,job,count,flush=True)
        return job
    try:
        smoke=submit('GPU preflight','gpu','smoke')
        refs=submit('MT/Heun and core calibration','cpu','reference',len(m['reference']))
        baseline=submit('Gaussian baseline','cpu','baseline',22)
        training=submit('Data fractions and deterministic training','gpu','train',len(m['training']),(smoke,))
        generation=submit('Size/time conditional paths','gpu','generate',len(m['generation']),(smoke,training))
        bench=submit('Same GPU timing','gpu','benchmark',dependencies=(smoke,))
        submit('Six-figure analysis','cpu','analysis',dependencies=(refs,baseline,training,generation,bench))
        report['status']='SUBMITTED_NOT_MONITORED'
    except Exception:
        report['status']='PARTIAL_SUBMISSION_REQUIRES_INSPECTION';save(path,report);raise
    save(path,report)


if __name__=='__main__':run()
