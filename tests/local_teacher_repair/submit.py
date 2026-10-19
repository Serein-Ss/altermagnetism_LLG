"""Submit only preflight or verified formal chain; never poll long jobs."""
import argparse,json,subprocess
from .run import HERE,ROOT,dump,verify


def submit(smoke):
    verify();(HERE/'logs').mkdir(exist_ok=True)
    target=HERE/('smoke_submission.json' if smoke else 'submission.json')
    assert not target.exists(),'submission already recorded'
    if not smoke:
        summary=json.loads((HERE/'smoke_assets/summary.json').read_text())
        assert summary['pairing_errors']==[] and len(summary['runs'])==2
        for run in summary['runs']:assert run['training_updates']==16 and run['auxiliary_calls']==2 and not run['scientific_certification']
    def sbatch(*args):
        result=subprocess.run(['sbatch','--parsable',*args],cwd=ROOT,text=True,capture_output=True,check=True)
        value=result.stdout.strip().split(';')[0];assert value.isdigit(),result.stdout;return value
    jobs={}
    jobs['gpu']=sbatch('--array=0-1' if smoke else '--array=0-5',
        *(['--time=00:10:00'] if smoke else []),str(HERE/'job.sh'),'smoke' if smoke else 'train')
    dump(target,dict(smoke=smoke,jobs=jobs,status='gpu accepted; downstream submission pending'))
    extra=['--smoke'] if smoke else []
    jobs['statistics']=sbatch('--array=0-1' if smoke else '--array=0-5',
        f'--dependency=afterok:{jobs["gpu"]}',str(HERE/'report.sh'),'analyze',*extra)
    dump(target,dict(smoke=smoke,jobs=jobs,status='gpu/statistics accepted; aggregate pending'))
    jobs['aggregate']=sbatch(f'--dependency=afterok:{jobs["statistics"]}',str(HERE/'report.sh'),'aggregate',*extra)
    dump(target,dict(smoke=smoke,jobs=jobs,status='accepted; no formal monitoring',estimated_gpu_hours=[3,4] if not smoke else None))
    print(json.dumps(jobs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();submit(a.smoke)
