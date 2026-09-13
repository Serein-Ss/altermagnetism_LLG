"""Freeze and submit bounded R2-R5 numerical validation, never production.

Default prepares only. --submit submits once; run the frozen worker via Slurm.
All changes to protocol sizes/durations below are validation choices, not paper values.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.core.literature_config import read_document,sha256

PAPERS=('bauer_2011','hirst_mn2au_2022','gomonay_2024','laliena_crnb3s6_2020')
TESTS=('test_reduced_literature.py','test_reduced_workflow.py','test_bauer_reduced.py',
       'test_remaining_literature.py','test_remaining_pipeline.py',
       'test_hirst_llb.py','test_bauer_weak_rk.py','test_validation_submission.py')


def task_specs():
    tasks=[]
    def add(paper,protocol,pool,**overrides):
        tasks.append(dict(paper=PAPERS[paper],protocol=protocol,pool=pool,overrides=overrides))
    # Weak-RK draws are not Wiener increments: compare independent ensembles,
    # not fictitious pathwise coupling across these step sizes.
    for seed in (20260910,20260911,20260912):
        for dt in (.02,.01,.005):
            add(0,'trajectory','cpu',batch=32,dt=dt,steps=round(200/dt),save_every=round(1/dt),seed=seed)
    for theta in ('theta_300','theta_1000','theta_1200'):
        for dt in (.002,.001,.0005):
            add(1,'llb_afmr','cpu',theta_key=theta,dt=dt,steps=round(200/dt),save_every=round(.1/dt))
    for extent,points,ds in ((20,501,.1),(25,501,.1),(40,1001,.1),(25,501,.05),(25,1001,.025)):
        add(3,'branch','cpu',extent=extent,points=points,ds=ds,continuation_steps=math.ceil(12/ds))
    # Non-winding-fixed relaxation: same mesh, different periodic lengths.
    for sites in (384,480,576):
        add(3,'helix','cpu',sites=sites,batch=4)
    for dt in (.00005,.000025,.0000125):
        add(3,'current','cpu',dt=dt,steps=round(100/dt),save_every=round(.1/dt))
    for u in (0.,.5,1.1):
        add(3,'current','cpu',u=u)
    # Hirst spatial and dt scans are separately varied, not confounded.
    for size,dt in ((8,.002),(12,.002),(16,.002),(8,.001),(8,.0005)):
        add(1,'equilibrium','gpu',shape=[size]*3,dt=dt,steps=round(200/dt),save_every=round(1/dt))
    for theta in ('theta_300','theta_1000','theta_1200'):
        add(1,'afmr','gpu',shape=[8]*3,theta_key=theta,equilibration_steps=100000,save_every=500)
    for dt in (.002,.001,.0005):
        add(1,'wall','gpu',shape=[512,4,4],dt=dt,steps=round(400/dt),save_every=round(4/dt))
    vectors=[[.5,0],[0,.5],[.5,.5],[-.5,.5],[.5,.25],[-.5,.25]]
    for protocol in ('spinwave','control'):
        for vector in vectors:
            add(2,protocol,'gpu',shape=[64,64],wavevectors=[vector],save_every=20)
    for size,dt in ((32,.005),(96,.005),(64,.0025),(64,.00125)):
        add(2,'spinwave','gpu',shape=[size,size],wavevectors=[[.5,.5]],dt=dt,steps=round(200/dt),save_every=round(.1/dt))
    for protocol in ('wall','moving_wall'):
        for dt in (.005,.0025,.00125):
            add(2,protocol,'gpu',shape=[512,8],dt=dt,steps=round(200/dt),save_every=round(1/dt))
    return tasks


def prepare():
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    campaign=ROOT/'scripts/archive'/('r2_r5_validation_'+stamp)
    campaign.mkdir()
    snapshot=campaign/'code/altermagnetism_LLG'
    files=[]
    for folder in ('scripts/core','scripts/literature','scripts/tests'):
        files.extend((ROOT/folder).rglob('*.py'))
    files += [ROOT/'scripts/workflow/run_reduced_literature_campaign.py',Path(__file__),ROOT/'slurm/remaining_validation.sbatch',ROOT/'GUIDE/STRICT_LITERATURE_REPRODUCTION_PLAN.md']
    files += list((ROOT/'conf/literature').glob('*.yaml'))
    hashes={}
    for src in files:
        relative=src.relative_to(ROOT); dest=snapshot/relative
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest); dest.chmod(0o444)
        hashes[str(relative)]=sha256(dest)
    tasks=task_specs()
    raw_bytes=0
    for index,task in enumerate(tasks):
        doc=read_document(snapshot/'conf/literature'/f"{task['paper']}.yaml")
        p=copy.deepcopy(doc['numerics']['protocols'][task['protocol']]); p.update(task['overrides'])
        doc['numerics']['protocols']={task['protocol']:p}
        doc['numerics']['validation_extension'].update(production_enabled=False,validation_protocols=[task['protocol']],
            campaign_kind='bounded_numerical_validation_not_paper_reproduction')
        path=campaign/'configs'/f'{index:03d}.yaml';path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(doc,sort_keys=False));path.chmod(0o444);read_document(path)
        task.update(index=index,config=str(path),config_sha256=sha256(path),run_id=f'{stamp}_validation_{index:03d}',parameters=p)
        if 'steps' in p:
            sites=math.prod(p['shape'])*(4 if task['paper']==PAPERS[1] or p.get('orientation')=='110' else 2) if 'shape' in p else p.get('sites',p.get('length',2))
            frames=math.ceil(p['steps']/p['save_every'])+1
            if 'equilibration_steps' in p: frames+=math.ceil(p['equilibration_steps']/p['save_every'])+1
            raw_bytes+=frames*p.get('batch',1)*sites*3*8
    free=shutil.disk_usage(ROOT).free
    if raw_bytes*2+10*2**30>free: raise RuntimeError('Insufficient filesystem free space for uncompressed bound plus margin')
    manifest=dict(campaign=str(campaign),root=str(ROOT),snapshot=str(snapshot),files_sha256=hashes,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        claim='Numerical validation only; production flags remain false; no ML training.',
        storage_estimate_uncompressed_spins_bytes=raw_bytes,quota_status='quota_reports_no_user_quota_project_quota_unknown',tasks=tasks)
    (campaign/'manifest.json').write_text(json.dumps(manifest,indent=2));(campaign/'manifest.json').chmod(0o444)
    return campaign,manifest


def run_task(campaign,kind,index):
    m=json.loads((campaign/'manifest.json').read_text());snapshot=Path(m['snapshot']); root=Path(m['root'])
    for rel,digest in m['files_sha256'].items():
        if sha256(snapshot/rel)!=digest: raise RuntimeError('Frozen source hash mismatch: '+rel)
    os.chdir(snapshot)
    if kind=='tests':
        subprocess.run([sys.executable,'-m','pytest','-q',*[f'scripts/tests/{name}' for name in TESTS]],check=True)
        return
    if kind=='gpu_check':
        import torch
        if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('Allocated CUDA unavailable')
        for paper in PAPERS:
            subprocess.run([sys.executable,'-m',f'scripts.literature.{paper}.run','--config',str(snapshot/'conf/literature'/f'{paper}.yaml'),
                '--protocol','smoke','--run-id',campaign.name+'_cuda_smoke','--project-root',str(root),'--device','cuda'],check=True)
        return
    task=[t for t in m['tasks'] if t['pool']==kind][index]
    if sha256(task['config'])!=task['config_sha256']: raise RuntimeError('Frozen configuration hash mismatch')
    module='llb_run' if task['protocol'].startswith('llb_') else 'run'
    start=time.monotonic()
    subprocess.run([sys.executable,'-m',f"scripts.literature.{task['paper']}.{module}",'--config',task['config'],
        '--protocol',task['protocol'],'--run-id',task['run_id'],'--project-root',str(root),
        '--validation','--device','cuda' if kind=='gpu' else 'cpu'],check=True)
    raw=root/'data/literature_reproduction'/task['paper']/'raw'/task['run_id']
    for path in raw.glob('*.h5'):
        with h5py.File(path) as h:
            if not h.attrs.get('complete',False): raise RuntimeError('Incomplete output')
            for frame in h['spins']:
                if not np.isfinite(frame).all(): raise RuntimeError('Nonfinite saved spins')
    out=root/'output/literature_reproduction'/task['paper']/task['run_id']
    (out/'execution.json').write_text(json.dumps(dict(task=task,elapsed_seconds=time.monotonic()-start,
        campaign=str(campaign),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        result='simulation_complete_pending_scientific_analysis'),indent=2))


def submit(campaign,m):
    record=campaign/'submission.json'
    with record.open('x') as f: json.dump({'status':'submitting','jobs':{}},f)
    logs=ROOT/'logs/literature_reproduction'/campaign.name; logs.mkdir(parents=True)
    jobs={}
    def one(kind,pool,dependency=None,count=None):
        command=['sbatch','--parsable','--job-name=zrs-mag',f'--partition={"rtx4090" if pool=="gpu" else "fat"}',
            '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=16G','--time=2-00:00:00',
            f'--output={logs}/{kind}_%A_%a.log']
        if pool=='gpu':command+=['--gres=gpu:1']
        if dependency:command+=['--dependency=afterok:'+dependency]
        if count:command += [f'--array=0-{count-1}%{2 if pool=="gpu" else 4}']
        command += [str(Path(m['snapshot'])/'slurm/remaining_validation.sbatch'),str(ROOT),str(campaign),kind]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        if not job.isdigit(): raise RuntimeError('Unexpected sbatch response: '+job)
        jobs[kind]=dict(job_id=job,count=count or 1,command=command)
        record.write_text(json.dumps(dict(status='submitting',jobs=jobs),indent=2));print(kind,job,flush=True)
        return job
    try:
        tests=one('tests','cpu')
        gpu=one('gpu_check','gpu',tests)
        one('cpu','cpu',tests,sum(t['pool']=='cpu' for t in m['tasks']))
        one('gpu','gpu',gpu,sum(t['pool']=='gpu' for t in m['tasks']))
    except Exception:
        record.write_text(json.dumps(dict(status='partial_submission_do_not_resubmit_blindly',jobs=jobs),indent=2));raise
    record.write_text(json.dumps(dict(status='submitted_not_monitored',jobs=jobs),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true')
    parser.add_argument('--campaign',type=Path);parser.add_argument('--worker',choices=('tests','gpu_check','cpu','gpu'))
    args=parser.parse_args()
    if args.worker:
        run_task(args.campaign,args.worker,int(os.environ.get('SLURM_ARRAY_TASK_ID','0')));return
    if args.campaign:
        campaign=args.campaign;m=json.loads((campaign/'manifest.json').read_text())
    else:campaign,m=prepare()
    print(json.dumps(dict(campaign=str(campaign),tasks=len(m['tasks']),raw_GiB=m['storage_estimate_uncompressed_spins_bytes']/2**30)),flush=True)
    if args.submit:submit(campaign,m)


if __name__=='__main__':main()

