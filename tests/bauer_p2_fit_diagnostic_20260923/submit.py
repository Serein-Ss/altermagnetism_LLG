import subprocess
from .run import HERE, ROOT, dump

if (HERE/'submission.json').exists():
    raise RuntimeError('Submission already recorded; do not duplicate jobs')
(HERE/'logs').mkdir(exist_ok=True)
jobs={}

def submit(name,*args):
    value=subprocess.check_output(['sbatch','--parsable',*args],cwd=ROOT,text=True).strip().split(';')[0]
    assert value.isdigit(),value
    jobs[name]=value
    dump(HERE/'submission.json',jobs)
    print(name,value,flush=True)
    return value

cpu=submit('decomposition','tests/bauer_p2_fit_diagnostic_20260923/cpu.sh','analyze')
gpu=submit('fixed_sample_fit','--array=0-2','tests/bauer_p2_fit_diagnostic_20260923/job.sh')
submit('report',f'--dependency=afterany:{cpu}:{gpu}','tests/bauer_p2_fit_diagnostic_20260923/report.sh')
