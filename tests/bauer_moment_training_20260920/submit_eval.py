import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    record = HERE/'evaluation_submission.json'
    if record.exists(): raise RuntimeError('Evaluation already submitted')
    train = json.loads((HERE/'submission.json').read_text())['train_array']
    jobs = {}
    def submit(stage, extra):
        args = ['sbatch','--parsable','--job-name=zrs-mag','--partition=rtx4090',
                '--gres=gpu:1','--cpus-per-task=1','--mem=8G','--time=01:00:00',
                '--output='+str(HERE/'logs'/f'eval_{stage}_%A_%a.out'),
                '--error='+str(HERE/'logs'/f'eval_{stage}_%A_%a.err')]
        job = subprocess.check_output(args+extra+[str(HERE/'evaluate.sh'),stage],text=True).strip().split(';')[0]
        jobs[stage] = job; record.write_text(json.dumps(jobs,indent=2)+'\n'); print(stage,job,flush=True)
        return job
    smoke = submit('smoke', [])
    submit('formal', ['--array=0-5','--dependency=afterok:'+train+':'+smoke,'--kill-on-invalid-dep=yes'])


if __name__ == '__main__': main()
