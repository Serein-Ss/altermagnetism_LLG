"""References -> audited data -> paired GPU repair -> physical stats -> page."""
import argparse,json,subprocess
from .data import HERE,ROOT,verify,dump


def submit(smoke):
    verify();target=HERE/('smoke_submission.json' if smoke else 'submission.json')
    assert not target.exists(),'already submitted'
    if not smoke:
        s=json.loads((HERE/'smoke_assets/summary.json').read_text())
        assert len(s['runs'])==2 and s['pairing_errors']==[]
        assert all(r['training_updates']==16 and r['auxiliary_calls']==2 for r in s['runs'])
        assert json.loads((HERE/'smoke_data_audit.json').read_text())['passed']
    jobs={}
    def sb(*args):
        result=subprocess.run(['sbatch','--parsable',*args],cwd=ROOT,text=True,capture_output=True,check=True)
        value=result.stdout.strip().split(';')[0];assert value.isdigit(),result.stdout;return value
    def save():dump(target,dict(smoke=smoke,jobs=jobs,status='accepted so far; chain may be partial'))
    jobs['references']=sb('--array=0' if smoke else '--array=1-23',str(HERE/'reference.sh'),'reference');save()
    extra=['--smoke'] if smoke else []
    jobs['data_audit']=sb(f'--dependency=afterok:{jobs["references"]}',str(HERE/'reference.sh'),'audit',*extra);save()
    jobs['gpu']=sb('--array=0-1' if smoke else '--array=0-5',f'--dependency=afterok:{jobs["data_audit"]}',
        *(['--time=00:10:00'] if smoke else []),str(HERE/'job.sh'),'smoke' if smoke else 'train');save()
    jobs['statistics']=sb('--array=0-1' if smoke else '--array=0-5',f'--dependency=afterok:{jobs["gpu"]}',str(HERE/'report.sh'),'analyze',*extra);save()
    jobs['aggregate']=sb(f'--dependency=afterok:{jobs["statistics"]}',str(HERE/'report.sh'),'aggregate',*extra)
    dump(target,dict(smoke=smoke,jobs=jobs,status='all accepted; no formal monitoring',estimated_gpu_hours=[3,4] if not smoke else None))
    print(json.dumps(jobs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();submit(a.smoke)
