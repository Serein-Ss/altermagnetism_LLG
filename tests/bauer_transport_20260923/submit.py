import subprocess
from .run import HERE,ROOT,dump

if (HERE/'submission.json').exists():raise RuntimeError('Already submitted')
gpu=subprocess.check_output(['sbatch','--parsable','--array=0-2','tests/bauer_transport_20260923/job.sh'],cwd=ROOT,text=True).strip().split(';')[0]
assert gpu.isdigit()
jobs={'transport_diagnostic':gpu};dump(HERE/'submission.json',jobs);print(jobs,flush=True)
report=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{gpu}','tests/bauer_transport_20260923/report.sh'],cwd=ROOT,text=True).strip().split(';')[0]
assert report.isdigit()
jobs['report']=report;dump(HERE/'submission.json',jobs);print(jobs,flush=True)
