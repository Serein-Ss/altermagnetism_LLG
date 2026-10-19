"""Submit the frozen campaign once; record receipt, never poll long tasks."""
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from .run import HERE, ROOT, dump, verify


def main():
    if (HERE/'submission.json').exists():raise RuntimeError('Already submitted; inspect receipts before any resubmission')
    verify()
    suites=ET.parse(HERE/'software_tests.xml').getroot().findall('.//testsuite')
    assert suites and sum(int(s.get('tests','0')) for s in suites)>=19
    assert all(int(s.get('errors','0'))==0 and int(s.get('failures','0'))==0 for s in suites)
    assert (HERE/'reference_audit.json').exists()
    benchmark=json.loads((HERE/'benchmark.json').read_text())
    by_arm={r['arm']:r for r in benchmark['rows']}
    protocol=json.loads((HERE/'protocol.json').read_text());estimates=[]
    for index,task in enumerate(protocol['tasks']):
        speed=by_arm[task['arm'].removesuffix('_long')]
        parent_equivalents=(198 if task['updates']>6000 else 162) if task['domain']=='bauer' else 72
        estimate=benchmark['cache_load_seconds']+task['updates']*speed['seconds_per_update']+parent_equivalents*speed['per_parent_evaluation_seconds']
        estimates.append(dict(index=index,task=task,minutes_estimate=estimate/60,
                              minutes_with_margin=estimate*1.8/60))
    # Wall limit is a failure safety boundary, NOT early-stopping at that duration.
    limit=max(30,math.ceil(max(x['minutes_with_margin'] for x in estimates)*2/5)*5)
    plan=dict(tasks=estimates,gpu=benchmark['gpu'],time_limit_minutes=limit,
        aggregate_gpu_hours=sum(x['minutes_estimate'] for x in estimates)/60,
        aggregate_gpu_hours_with_margin=sum(x['minutes_with_margin'] for x in estimates)/60,
        report_estimate_minutes=[3,10],report_limit_minutes=30,
        qualification='benchmark-based estimate, not queue-inclusive promise; array has no user concurrency cap',
        monitoring='record sbatch receipts only; no post-submission long-task monitoring')
    dump(HERE/'runtime_plan.json',plan)
    print(json.dumps({k:v for k,v in plan.items() if k!='tasks'},ensure_ascii=False),flush=True)
    command=['sbatch','--parsable',f'--array=0-{len(estimates)-1}',f'--time={limit}:00',
             '--partition=rtx4090','tests/bauer_comprehensive_20260923/job.sh']
    gpu=subprocess.check_output(command,cwd=ROOT,text=True).strip().split(';')[0]
    assert gpu.isdigit();receipt=dict(training_array=gpu,tasks=len(estimates),command=command)
    dump(HERE/'submission.json',receipt);print(receipt,flush=True)
    report=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{gpu}',
        'tests/bauer_comprehensive_20260923/report.sh'],cwd=ROOT,text=True).strip().split(';')[0]
    assert report.isdigit();receipt['report']=report;dump(HERE/'submission.json',receipt);print(receipt,flush=True)


if __name__=='__main__':main()
