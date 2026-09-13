"""Freeze, time, budget and submit the next bounded evidence campaign.

Not a production switch. Independent literature diagnostics run separately;
wide zero-field equilibrium pilots require measured hash-bound kernel gates.
No automatic resubmission and no monitoring after formal submission.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.core.literature_config import read_document,sha256

PAPERS=('nishino_miyashita_2015','bauer_2011','hirst_mn2au_2022','gomonay_2024','laliena_crnb3s6_2020')


def specifications():
    tasks=[]
    def literature(paper,protocol,pool='cpu',**updates):
        tasks.append(dict(kind='literature',paper=PAPERS[paper],protocol=protocol,pool=pool,updates=updates))
    for index in range(8):
        tasks.append(dict(kind='nishino',paper=PAPERS[0],protocol='convergence',pool='cpu',task=index))
    for project in (False,True):
        for dt in (.02,.01,.005):
            literature(1,'trajectory',batch=32,dt=dt,steps=round(2000/dt),save_every=round(1/dt),project=project)
    literature(1,'trajectory',batch=8,dt=.005,steps=150000000,save_every=200)
    for theta in ('theta_300','theta_1000','theta_1200'):
        literature(2,'equilibrium','gpu',shape=[30,30,30],theta_key=theta,save_every=1000)
        literature(2,'afmr','gpu',shape=[8,8,8],theta_key=theta,steps=1000000,equilibration_steps=200000,save_every=100)
        literature(2,'llb_afmr',theta_key=theta,steps=1000000,save_every=100)
    literature(2,'wall','gpu',shape=[512,4,4],steps=8000000,save_every=5000)
    # Source S1 cell interpretation is explicit. These are finite-strip
    # diagnostics, not the unresolved original 10000x30 geometry certificate.
    speed=2*math.sqrt(1+2*1.88/11.1)
    for orientation in ('100','110'):
        literature(3,'wall','gpu',shape=[512,8],orientation=orientation,steps=200000,save_every=200)
        for fraction in (.4,.9):
            literature(3,'moving_wall','gpu',shape=[512,8],orientation=orientation,velocity=fraction*speed,
                       steps=400000,save_every=200)
    # Independently varied numerical axes; no material parameter retuning.
    for extent,points,ds in ((25,501,.025),(25,1001,.025),(25,2001,.025),
                             (20,801,.025),(40,1601,.025),(25,1001,.0125),(25,1001,.00625)):
        literature(4,'branch',extent=extent,points=points,ds=ds,continuation_steps=math.ceil(12/ds))
    for sites in (384,480,576):
        literature(4,'helix',sites=sites,batch=4,steps=20000000,save_every=10000)
    for factor in (1,2,4):
        dx=.12992957746478873/factor;dt=.00005/factor**2
        literature(4,'current',sites=480*factor,dx=dx,dt=dt,steps=round(100/dt),save_every=round(.1/dt))
    for theta in (.3,1.5):
        tasks.append(dict(kind='static_mc',pool='cpu',theta=theta))
        for method,dt in [('heun_projected_predictor',.005),('heun_projected_predictor',.0025),
                          ('heun_projected_predictor',.00125),('geometric_midpoint',.005)]:
            tasks.append(dict(kind='kernel_equilibrium',pool='cpu',theta=theta,shape=[4,4],method=method,dt=dt))
    for size in (16,32,48,64,96,128):
        for theta in (.1,.3,.6,1.,1.5,2.5):
            tasks.append(dict(kind='temperature_pilot',pool='gpu',theta=theta,shape=[size,size],method='heun_projected_predictor',dt=.005))
    return tasks


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False))


def prepare():
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    campaign=ROOT/'conf/frozen/zero_field_gomonay'/stamp;campaign.mkdir(parents=True)
    snapshot=ROOT/'scripts/archive'/('zero_field_'+stamp)/'code/altermagnetism_LLG'
    files=[]
    for folder in ('core','literature','tests','config','generation','datasets','analysis','validation','workflow','model','training','visualization','inference'):
        files.extend((ROOT/'scripts'/folder).rglob('*.py'))
    files+=list((ROOT/'conf/literature').glob('*.yaml'))+list((ROOT/'conf/zero_field_gomonay').glob('*'))
    files+=list((ROOT/'slurm').glob('zero_field*.sbatch'))+list((ROOT/'GUIDE').glob('*.md'))
    files += [ROOT/'LFS_SPLIT_MANIFEST.json',ROOT/'output/literature_reproduction/nishino_miyashita_2015/20260909_8df33ac_bb48dc64/report.json']
    files += list((ROOT/'data/literature_reproduction').glob('*/reference/target_registry.json'))
    hashes={}
    for src in files:
        if not src.is_file() or '__pycache__' in src.parts: continue
        rel=src.relative_to(ROOT);dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest);dest.chmod(0o444);hashes[str(rel)]=sha256(dest)
    out=ROOT/'output/zero_field_program'/stamp;out.mkdir(parents=True)
    log=ROOT/'logs/zero_field_gomonay'/stamp;log.mkdir(parents=True)
    tasks=specifications();benchmarks={}
    for index,task in enumerate(tasks):
        task.update(index=index,run_id=f'{stamp}_{index:03d}')
        config=campaign/f'{index:03d}.yaml'
        if task['kind'] in ('literature','nishino'):
            doc=read_document(snapshot/'conf/literature'/f'{task["paper"]}.yaml')
            ext=doc['numerics']['validation_extension']
            ext.update(production_enabled=False,campaign_kind='independent_diagnostic_not_literature_certificate')
            if task['kind']=='nishino':
                doc['numerics']['paper_settings'].update(total_steps=320000,burn_steps=160000)
                doc['numerics']['numerical_choice'].update(theta_grid=[2.],seed=2026091100,save_every=1000)
                ext['convergence_repetitions']=4
                params=doc['numerics']['paper_settings']
            else:
                params=copy.deepcopy(doc['numerics']['protocols'][task['protocol']]);params.update(task['updates'])
                if 'seed' in params: params['seed']=2026091100+100*index
                if 'steps' in params and not task['protocol'].startswith('llb'):
                    params.update(resumable=True,checkpoint_every=10000)
                doc['numerics']['protocols']={task['protocol']:params}
                ext['validation_protocols']=[task['protocol']]
            raw=ROOT/'data/literature_reproduction'/task['paper']/'raw'/task['run_id']
            task['raw']=str(raw)
        elif task['kind']=='static_mc':
            doc=dict(material_config=str(snapshot/'conf/literature/gomonay_2024.yaml'),length=4,theta=task['theta'],
                     sweeps=200000,stride=20,seed=902000+index*10)
            params=doc;task['raw']=str(ROOT/'data/research/zero_field_gomonay'/task['run_id']/'static_mc.npz')
        else:
            doc=yaml.safe_load((snapshot/'conf/zero_field_gomonay/software_validation.yaml').read_text())
            doc.update(role='equilibrium_diagnostic',run_id=task['run_id'],
                material_config=str(snapshot/'conf/literature/gomonay_2024.yaml'),theta=task['theta'],shape=task['shape'],
                method=task['method'],dt=task['dt'],steps=round((2000 if task['kind']=='kernel_equilibrium' else 1000)/task['dt']),
                save_every=round(.5/task['dt']),checkpoint_every=10000,
                noise_seeds=[903000+10*index+i for i in range(4)],seed=903000+10*index,initial_seed=904000+10*index)
            params=doc;task['raw']=str(ROOT/'data/research/zero_field_gomonay'/task['run_id']/'trajectory.h5')
        config.write_text(yaml.safe_dump(doc,sort_keys=False));config.chmod(0o444)
        task.update(config=str(config),config_sha256=sha256(config),parameters=params)
        # Include actual shape, batch, integrator, output cadence and projection
        # in timing groups; theta affects geometric iteration so retain it.
        key=json.dumps(dict(kind=task['kind'],paper=task.get('paper'),protocol=task.get('protocol'),pool=task['pool'],
            shape=params.get('shape',params.get('sites',params.get('moments'))),batch=params.get('batch'),
            method=params.get('method'),project=params.get('project'),theta=task.get('theta') if task.get('method')=='geometric_midpoint' else None,
            save_every=params.get('save_every'),orientation=params.get('orientation')),sort_keys=True)
        if key not in benchmarks: benchmarks[key]=index
        task['benchmark']=benchmarks[key]
    bindings=dict(source_tree=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                  material=sha256(snapshot/'conf/literature/gomonay_2024.yaml'),
                  acceptance=sha256(snapshot/'conf/zero_field_gomonay/acceptance.yaml'))
    m=dict(root=str(ROOT),campaign=str(campaign),snapshot=str(snapshot),output=str(out),logs=str(log),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files_sha256=hashes,tasks=tasks,benchmarks=list(benchmarks.values()),bindings=bindings,
        acceptance=str(snapshot/'conf/zero_field_gomonay/acceptance.yaml'),
        budget=dict(max_allocations=192,cpu_hours=512,gpu_hours=256,storage_gib=256,walltime_margin=1.5,max_retries=0),
        production_enabled=False)
    dump(campaign/'manifest.json',m);(campaign/'manifest.json').chmod(0o444)
    dump(out/'campaign.json',dict(campaign=str(campaign),status='prepared_not_submitted'))
    print(str(campaign),flush=True)
    return campaign,m


def command_for(task,config,run_id,m,*,timing=False):
    root=Path(m['root']);kind=task['kind'];device='cuda' if task['pool']=='gpu' else 'cpu'
    if kind=='nishino':
        return [sys.executable,'-m',f'scripts.literature.{task["paper"]}.run','--config',str(config),
            '--project-root',str(root),'--run-id',run_id,'--phase','convergence','--task',str(task['task']),'--device',device]
    if kind=='literature':
        module='llb_run' if task['protocol'].startswith('llb') else 'run'
        return [sys.executable,'-m',f'scripts.literature.{task["paper"]}.{module}','--config',str(config),
            '--project-root',str(root),'--run-id',run_id,'--protocol',task['protocol'],'--validation','--device',device]
    path=root/'data/research/zero_field_gomonay'/run_id/('static_mc.npz' if kind=='static_mc' else 'trajectory.h5')
    module='scripts.validation.gomonay_static_reference' if kind=='static_mc' else 'scripts.generation.generate_zero_field_gomonay'
    cmd=[sys.executable,'-m',module,'--config',str(config),'--output',str(path)]
    if kind!='static_mc': cmd+=['--device',device]
    return cmd


def worker(campaign,mode,index):
    m=json.loads((campaign/'manifest.json').read_text());snapshot=Path(m['snapshot']);out=Path(m['output'])
    for rel,digest in m['files_sha256'].items():
        if sha256(snapshot/rel)!=digest: raise RuntimeError('frozen source changed: '+rel)
    os.chdir(snapshot)
    if mode in ('checks','gpu_checks'):
        device='cuda' if mode=='gpu_checks' else 'cpu'
        if mode=='checks':
            subprocess.run([sys.executable,'scripts/config/validate_literature_configs.py'],check=True)
            subprocess.run([sys.executable,'-m','pytest','scripts/tests','-q'],check=True)
            subprocess.run([sys.executable,'-m','scripts.workflow.zero_field_inventory','--campaign',str(campaign)],check=True)
        else:
            subprocess.run([sys.executable,'-m','pytest','scripts/tests/test_zero_field_program.py','-q'],check=True)
        subprocess.run([sys.executable,'-m','scripts.validation.zero_field_bath','--output',str(out/f'bath_{device}.json'),'--device',device],check=True)
        return
    if mode=='gate':
        subprocess.run([sys.executable,'-m','scripts.analysis.zero_field_equilibrium','--campaign',str(campaign)],check=True);return
    task=m['tasks'][index]
    if sha256(task['config'])!=task['config_sha256']: raise RuntimeError('frozen config mismatch')
    if mode=='run' and task['kind']=='temperature_pilot':
        from scripts.validation.zero_field_certificates import verify
        try: verify(campaign/'equilibrium_gate.json','small_periodic_gomonay_static_kernel',m['bindings'])
        except (OSError,KeyError,ValueError,TypeError) as e:
            dump(out/'tasks'/f'{index:03d}.json',dict(status='blocked_kernel_certificate',reason=str(e)));return
    config=Path(task['config']);run_id=task['run_id'];work=task['parameters']
    count=work.get('steps',work.get('total_steps',work.get('sweeps',work.get('continuation_steps',1))))
    count+=work.get('equilibration_steps',0)
    if mode=='timing':
        doc=yaml.safe_load(config.read_text());run_id+='_timing'
        if task['kind']=='nishino':
            doc['numerics']['paper_settings'].update(total_steps=10000,burn_steps=5000);count=10000
        elif task['kind']=='literature':
            p=doc['numerics']['protocols'][task['protocol']]
            if 'steps' in p:
                p['steps']=10000;count=10000
                if 'equilibration_steps' in p: p['equilibration_steps']=10000;count+=10000
            else:
                p['continuation_steps']=8;count=8
        elif task['kind']=='static_mc': doc['sweeps']=10000;count=10000
        else: doc.update(steps=10000,run_id=run_id);count=10000
        config=campaign/'timing_configs'/f'{index:03d}.yaml';config.parent.mkdir(exist_ok=True)
        with config.open('x') as f: yaml.safe_dump(doc,f,sort_keys=False)
        config.chmod(0o444)
    command=command_for(task,config,run_id,m,timing=mode=='timing')
    start=time.monotonic()
    process=subprocess.Popen(command)
    previous=signal.signal(signal.SIGUSR1,lambda signum,stack: process.send_signal(signal.SIGUSR1))
    try:
        code=process.wait()
    finally:
        signal.signal(signal.SIGUSR1,previous)
    result=subprocess.CompletedProcess(command,code)
    seconds=time.monotonic()-start
    if task['kind'] in ('nishino','literature'):
        raw=Path(m['root'])/'data/literature_reproduction'/task['paper']/'raw'/run_id
        if not raw.exists(): raw=raw.with_name(raw.name+'.partial')
    else: raw=Path(m['root'])/'data/research/zero_field_gomonay'/run_id
    files=list(raw.rglob('*')) if raw.exists() else []
    record=dict(index=index,command=command,returncode=result.returncode,elapsed_seconds=seconds,work_units=count,
        maxrss_children_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        raw_bytes=sum(f.stat().st_size for f in files if f.is_file()),
        raw_files_sha256={str(f):sha256(f) for f in files if f.is_file() and not f.name.endswith('.tmp')},
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),status='complete_pending_scientific_analysis' if result.returncode==0 else
        'checkpointed_not_complete' if result.returncode==85 else 'failed')
    dump(out/('timing' if mode=='timing' else 'tasks')/f'{index:03d}.json',record)
    if result.returncode: raise SystemExit(result.returncode)


def sbatch(m,mode,*,indices=None,pool='cpu',dependency=None,minutes=30):
    command=['sbatch','--parsable','--job-name=zrs-mag','--partition='+('rtx4090' if pool=='gpu' else 'fat'),
        '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=12G',f'--time={minutes}',
        '--signal=B:USR1@180',f'--output={m["logs"]}/{mode}_%A_%a.log']
    if pool=='gpu': command+=['--gres=gpu:1']
    if dependency: command+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
    if indices: command+=['--array='+','.join(map(str,indices))+('%2' if pool=='gpu' else '%4')]
    command+=[str(Path(m['snapshot'])/'slurm/zero_field_campaign.sbatch'),m['root'],m['campaign'],mode]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    if not job.isdigit(): raise RuntimeError('unexpected sbatch response')
    print(mode,pool,job,indices,flush=True)
    return dict(job_id=job,mode=mode,pool=pool,indices=indices or [],command=command,minutes=minutes)


def submit_preflight(campaign,m):
    path=campaign/'preflight_submission.json'
    with path.open('x') as f: json.dump(dict(status='submitting',jobs=[]),f)
    jobs=[]
    def add(*args,**kwargs):
        row=sbatch(m,*args,**kwargs);jobs.append(row);dump(path,dict(status='submitting',jobs=jobs));return row['job_id']
    cpu=add('checks',minutes=30)
    gpu=add('gpu_checks',pool='gpu',dependency='afterok:'+cpu,minutes=30)
    for pool,dep in [('cpu',cpu),('gpu',gpu)]:
        ids=[i for i in m['benchmarks'] if m['tasks'][i]['pool']==pool]
        add('timing',pool=pool,indices=ids,dependency='afterok:'+dep,minutes=30)
    dump(path,dict(status='submitted_preflight_only',jobs=jobs))


def submit_runs(campaign,m):
    out=Path(m['output']);pre=json.loads((campaign/'preflight_submission.json').read_text())
    for row in pre['jobs']:
        states=subprocess.check_output(['sacct','-X','-j',row['job_id'],'-n','-P','-o','State'],text=True).splitlines()
        if not states or any(s.strip()!='COMPLETED' for s in states): raise RuntimeError('preflight not all completed: '+str(row))
    elapsed={'cpu':0.,'gpu':0.};bytes_bound=0;allocations=sum(len(r['indices']) or 1 for r in pre['jobs'])
    estimates=[]
    for task in m['tasks']:
        timing=json.loads((out/'timing'/f'{task["benchmark"]:03d}.json').read_text())
        if timing['returncode']!=0 or timing['work_units']<10000 and task.get('protocol')!='branch':
            raise RuntimeError('invalid timing evidence')
        p=task['parameters'];units=p.get('steps',p.get('total_steps',p.get('sweeps',p.get('continuation_steps',1))))+p.get('equilibration_steps',0)
        ratio=units/timing['work_units'];minutes=max(15,math.ceil((timing['elapsed_seconds']*ratio*m['budget']['walltime_margin']+600)/60))
        estimates.append(dict(index=task['index'],minutes=minutes,estimated_bytes=math.ceil(timing['raw_bytes']*ratio*1.5)))
        if 'steps' in p:
            sites=math.prod(p['shape'])*(4 if task.get('paper')==PAPERS[2] or p.get('orientation')=='110' else 2) if 'shape' in p else p.get('sites',p.get('length',2))
            frames=math.ceil(p['steps']/p['save_every'])+2
            frames+=math.ceil(p.get('equilibration_steps',0)/p['save_every'])
            uncompressed=frames*p.get('batch',1)*sites*3*8
            estimates[-1]['estimated_bytes']=max(estimates[-1]['estimated_bytes'],math.ceil(uncompressed*1.5))
        elapsed[task['pool']]+=minutes/60;bytes_bound+=estimates[-1]['estimated_bytes'];allocations+=1
    # No silent smaller ensemble/time window if this bounded campaign won't fit.
    budget=dict(hours=elapsed,storage_gib=bytes_bound/2**30,allocations=allocations+1,estimates=estimates)
    dump(out/'resource_estimate.json',budget)
    if elapsed['cpu']>m['budget']['cpu_hours'] or elapsed['gpu']>m['budget']['gpu_hours'] or bytes_bound>m['budget']['storage_gib']*2**30 or allocations+1>m['budget']['max_allocations']:
        raise RuntimeError('campaign budget exceeded; revise scheduling/scope explicitly, not physics')
    if bytes_bound+10*2**30>shutil.disk_usage(m['root']).free: raise RuntimeError('insufficient disk free space')
    path=campaign/'submission.json'
    with path.open('x') as f: json.dump(dict(status='submitting',jobs=[]),f)
    jobs=[];kernel=[];lanes={'cpu':[None]*4,'gpu':[None]*2};used={'cpu':0,'gpu':0}
    ordered=sorted(zip(m['tasks'],estimates),key=lambda pair: 0 if pair[0]['kind'] in ('kernel_equilibrium','static_mc') else 1)
    for task,estimate in ordered:
        if task['kind']=='temperature_pilot': continue
        pool=task['pool'];slot=used[pool]%len(lanes[pool]);used[pool]+=1
        dep='afterany:'+lanes[pool][slot] if lanes[pool][slot] else None
        row=sbatch(m,'run',indices=[task['index']],pool=pool,minutes=estimate['minutes'],dependency=dep)
        lanes[pool][slot]=row['job_id']
        jobs.append(row);dump(path,dict(status='submitting',jobs=jobs))
        if task['kind'] in ('static_mc','kernel_equilibrium'): kernel.append(row['job_id'])
    row=sbatch(m,'gate',dependency='afterok:'+':'.join(kernel),minutes=30);jobs.append(row);gate=row['job_id']
    dump(path,dict(status='submitting',jobs=jobs))
    for task,estimate in zip(m['tasks'],estimates):
        if task['kind']!='temperature_pilot': continue
        slot=used['gpu']%2;used['gpu']+=1
        dependencies=[gate]+([lanes['gpu'][slot]] if lanes['gpu'][slot] else [])
        row=sbatch(m,'run',indices=[task['index']],pool='gpu',dependency='afterany:'+':'.join(dependencies),minutes=estimate['minutes'])
        lanes['gpu'][slot]=row['job_id']
        jobs.append(row);dump(path,dict(status='submitting',jobs=jobs))
    dump(path,dict(status='submitted_not_monitored',jobs=jobs))
    dump(out/'campaign.json',dict(campaign=str(campaign),status='submitted_not_monitored',submission=str(path)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--submit',action='store_true')
    p.add_argument('--worker',choices=('checks','gpu_checks','timing','run','gate'))
    a=p.parse_args()
    if a.worker: worker(a.campaign,a.worker,int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    else:
        if a.prepare: campaign,m=prepare()
        else: campaign=a.campaign;m=json.loads((campaign/'manifest.json').read_text())
        if a.preflight: submit_preflight(campaign,m)
        if a.submit: submit_runs(campaign,m)
