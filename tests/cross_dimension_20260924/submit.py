"""Estimate and submit once; intentionally no long-job polling."""
import argparse
import json
import subprocess
from tests.bauer_p2_20260923.common import dump
from .run import HERE,ROOT,tasks


def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args()
    bench=json.loads((HERE/'benchmark/complete.json').read_text())
    toy=json.loads((HERE/'toy_smoke/complete.json').read_text())
    if toy['fresh_interpolant_support_failures']!=0:raise RuntimeError('failed oracle preflight')
    # Per-step generation measured at full smallest geometry; larger shapes scale with sites.
    sample_step=bench['generation'][0]['seconds']/(16*2)
    two_d=bench['warm_step_seconds']*6000+sample_step*(16*(16+64+128+256)+4*16*128+8*128*(96*56+128*74)/(64*38))
    toy_seconds=toy['warm_step_seconds']*12000
    plan=dict(benchmark_job='670675',toy_smoke_job='670676',oracle_job='670677',
        actual_benchmark_gpu=bench['device'],partition='rtx4090',job_name='zrs-mag',
        two_d_measured_linear_seconds=two_d,two_d_conservative_minutes=[30,45],
        toy_training_seconds=toy_seconds,toy_total_estimated_minutes=[6,12],
        cpu_report_estimated_minutes=[5,20],
        gpu_hours_estimated=[9*.5+18*.1,9*.75+18*.2],
        wall_time='Queue-dependent; parallel jobs, do not sum elapsed times as wall-clock forecast.',
        limits='90min per 2D task,45min per toy task,60min report; safety limits are not planned run durations',
        monitor='Only short preflight monitored. No polling of submitted long tasks.',tasks=tasks())
    dump(HERE/'runtime_plan.json',plan);print(json.dumps(plan,indent=2))
    if not a.submit:return
    if (HERE/'submission.json').exists():raise RuntimeError('submission already recorded; do not duplicate')
    record=dict(jobs=[],oracle='670677',runtime_plan=str(HERE/'runtime_plan.json'))
    for domain,limit in [('miltat','01:30:00'),('toy','00:45:00')]:
        indices=','.join(str(i) for i,r in enumerate(tasks()) if r['domain']==domain)
        command=['sbatch','--parsable','--time='+limit,'--array='+indices,str(HERE/'job.sh')]
        job=subprocess.check_output(command,cwd=ROOT,text=True).strip().split(';')[0]
        record['jobs'].append(dict(domain=domain,id=job,array=indices,command=command))
        dump(HERE/'submission.json',record)
    dependency='afterany:'+':'.join([r['id'] for r in record['jobs']]+[record['oracle']])
    command=['sbatch','--parsable','--dependency='+dependency,str(HERE/'report.sh')]
    record['report']=subprocess.check_output(command,cwd=ROOT,text=True).strip().split(';')[0]
    dump(HERE/'submission.json',record);print('ACCEPTED',json.dumps(record))


if __name__=='__main__':main()
