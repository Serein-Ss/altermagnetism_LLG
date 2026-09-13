"""Resume R1 under one report-specific user acceptance; preserve all raw gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[2]
PAPER='nishino_miyashita_2015'
SOURCE_RUN='20260909_8df33ac_bb48dc64'
SOURCE_CAMPAIGN='r0_20260909_8df33ac_6fd7226f'


def accepted_report(report):
    if report['run_id']!=SOURCE_RUN or report['phase']!='convergence' or len(report['rows'])!=216:
        raise ValueError('acceptance applies only to the reviewed convergence report')
    failed=[]
    for row in report['rows']:
        if not row['mean_check']['passed'] or not row['cdf_check']['passed'] or row['norm_error']>1e-10:
            raise ValueError('non-waived hard check failed')
        if not row['stationarity_check']['passed']:
            failed.append(row)
            if row['case']!='A' or row['theta']!=2.:
                raise ValueError('unreviewed stationarity failure')
    if len(failed)!=9 or len(report['paired_convergence'])!=96 or not all(x['passed'] for x in report['paired_convergence']):
        raise ValueError('reviewed evidence changed')
    return 1  # Largest tested dt; paired dt and equilibrium checks passed.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decision',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm allocation required')
    decision=json.loads(a.decision.read_text())
    report_path=ROOT/decision['source_report']
    if hashlib.sha256(report_path.read_bytes()).hexdigest()!=decision['source_report_sha256']:
        raise ValueError('reviewed report changed')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=decision['resume_script_sha256']:
        raise ValueError('resume code changed after acceptance')
    report=json.loads(report_path.read_text());factor=accepted_report(report)
    archive=ROOT/'scripts/archive'/SOURCE_CAMPAIGN
    workflow=archive/'code/altermagnetism_LLG/scripts/workflow/run_reduced_literature_campaign.py'
    api=runpy.run_path(str(workflow),run_name='frozen_r0_workflow')
    snapshot=json.loads((archive/'snapshot.json').read_text());api['verify_snapshot'](snapshot)
    base=api['read_document'](archive/'input.yaml')
    base['numerics']['acceptance_record']={'path':str(a.decision.relative_to(ROOT)),
        'sha256':api['sha256'](a.decision),'scope':'reviewed convergence stationarity only; original fail preserved'}
    campaign=decision['continuation_id']
    config_dir=ROOT/'conf/frozen'/PAPER/campaign;config_dir.mkdir(parents=True)
    output=ROOT/'output/literature_reproduction'/PAPER/campaign;output.mkdir(parents=True,exist_ok=True)
    workers=int(os.environ.get('SLURM_CPUS_PER_TASK','1'))
    try:
        result=api['run_phase'](ROOT,base,'primary',config_dir,snapshot,workers,factor)
        status={'status':'pass','primary_run':result['run_id']}
    except Exception as error:
        api['write_json'](output/'continuation_report.json',{'status':'fail','reason':str(error),
            'original_convergence_status':'fail','user_disposition':'accepted_with_deviation'})
        raise
    status.update(original_convergence_status='fail',user_disposition='accepted_with_deviation',
        source_run=SOURCE_RUN,training_dataset_production_enabled=False,
        next_stage='R2 Bauer: paper-specific implementation and gates required; not automatically submitted')
    api['write_json'](output/'continuation_report.json',status)


if __name__=='__main__': main()
