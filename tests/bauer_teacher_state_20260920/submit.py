"""Submit diagnostics with smoke dependencies; refuse duplicate submission."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    record = HERE/'submission.json'
    if record.exists():
        raise RuntimeError('Submission recorded; inspect jobs before retry')
    (HERE/'logs').mkdir(exist_ok=True)
    jobs = {}
    def submit(stage, gpu=False, dependency=None, array=None):
        args = ['sbatch', '--parsable', '--job-name=zrs-mag',
                '--partition='+('rtx4090' if gpu else 'fat'), '--cpus-per-task=1',
                '--mem=8G', '--time=01:00:00',
                '--output='+str(HERE/'logs'/f'{stage}_%A_%a.out'),
                '--error='+str(HERE/'logs'/f'{stage}_%A_%a.err')]
        if gpu: args += ['--gres=gpu:1']
        if dependency: args += ['--dependency=afterok:'+dependency, '--kill-on-invalid-dep=yes']
        if array: args += ['--array='+array]
        job = subprocess.check_output(args+[str(HERE/'job.sh'), stage], text=True).strip().split(';')[0]
        jobs[stage] = job
        record.write_text(json.dumps(jobs, indent=2)+'\n')
        print(stage, job, flush=True)
        return job
    r = submit('reference_smoke')
    m = submit('smoke', gpu=True)
    submit('reference', dependency=r)
    submit('model', gpu=True, dependency=m, array='0-2')


if __name__ == '__main__':
    main()
