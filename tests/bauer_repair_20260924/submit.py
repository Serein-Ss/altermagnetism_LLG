"""Freeze receipt and leave the long campaign unmonitored."""
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from .run import HERE, ROOT, dump, verify


def main():
    if (HERE/'submission.json').exists():raise RuntimeError('submission already recorded')
    verify()
    suites=ET.parse(HERE/'software_tests.xml').getroot().findall('.//testsuite')
    assert suites and sum(int(s.get('tests','0')) for s in suites)>=14
    assert all(int(s.get('failures','0'))+int(s.get('errors','0'))==0 for s in suites)
    benchmark=json.loads((HERE/'benchmark.json').read_text());speeds={r['arm']:r for r in benchmark['rows']}
    for arm,row in speeds.items():
        if arm!='baseline':assert row['residual_rms']>1e-5 and all(v>0 for v in row['residual_gradient_norms'].values())
    protocol=json.loads((HERE/'protocol.json').read_text());estimates=[]
    for index,task in enumerate(protocol['tasks']):
        speed=speeds[task['arm']];parents=198 if task['domain']=='bauer' else 84
        seconds=benchmark['cache_load_seconds']+speed['seconds_per_update']*task['updates']+parents*speed['parent_eval_seconds']
        estimates.append(dict(index=index,task=task,minutes_estimate=seconds/60,minutes_with_margin=seconds*1.8/60))
    limit=max(30,math.ceil(max(r['minutes_with_margin'] for r in estimates)*2/5)*5)
    plan=dict(tasks=estimates,aggregate_gpu_hours=sum(r['minutes_estimate'] for r in estimates)/60,
              aggregate_gpu_hours_with_margin=sum(r['minutes_with_margin'] for r in estimates)/60,
              time_limit_minutes=limit,report_minutes=[3,8],gpu=benchmark['gpu'],
              warning='estimates exclude queue; no array concurrency cap; no physical accuracy or speedup claim')
    dump(HERE/'runtime_plan.json',plan);print({k:v for k,v in plan.items() if k!='tasks'},flush=True)
    command=['sbatch','--parsable',f'--array=0-{len(estimates)-1}',f'--time={limit}:00','--partition=rtx4090','tests/bauer_repair_20260924/job.sh']
    job=subprocess.check_output(command,cwd=ROOT,text=True).strip().split(';')[0];assert job.isdigit()
    receipt=dict(training_array=job,tasks=len(estimates),command=command)
    dump(HERE/'submission.json',receipt);print(receipt,flush=True)
    report=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{job}','tests/bauer_repair_20260924/report.sh'],cwd=ROOT,text=True).strip().split(';')[0]
    assert report.isdigit();receipt['report']=report;dump(HERE/'submission.json',receipt);print(receipt,flush=True)


if __name__=='__main__':main()
