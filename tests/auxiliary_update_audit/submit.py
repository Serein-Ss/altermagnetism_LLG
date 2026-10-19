import argparse,json,subprocess
from .run import HERE,ROOT,dump,verify


def main(smoke):
    verify();path=HERE/('smoke_submission.json' if smoke else 'submission.json');assert not path.exists()
    if not smoke:
        s=json.loads((HERE/'smoke_assets/summary.json').read_text());assert s['pairing_errors']==[] and s['actual_model_updates']==0
    def sbatch(*args):
        r=subprocess.run(['sbatch','--parsable',*args],cwd=ROOT,capture_output=True,text=True,check=True)
        value=r.stdout.strip().split(';')[0];assert value.isdigit();return value
    gpu=sbatch('--array=0' if smoke else '--array=0-8',*( ['--time=00:05:00'] if smoke else []),str(HERE/'job.sh'),*( ['--smoke'] if smoke else []))
    dump(path,dict(gpu=gpu,status='GPU accepted, report pending'))
    report=sbatch(f'--dependency=afterok:{gpu}',str(HERE/'report.sh'),*( ['--smoke'] if smoke else []))
    dump(path,dict(gpu=gpu,report=report,estimated_minutes=[8,15] if not smoke else [1,3],
        status='accepted; stop monitoring formal jobs',actual_training_updates=0))
    print(json.dumps(dict(gpu=gpu,report=report)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
