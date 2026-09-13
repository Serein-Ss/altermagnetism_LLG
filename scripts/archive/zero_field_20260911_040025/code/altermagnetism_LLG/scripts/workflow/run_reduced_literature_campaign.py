"""R0 only: frozen code, convergence gate, then conditional Nishino Fig.1."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from altermagnetism_LLG.scripts.core.literature_config import load_runtime,read_document,sha256
from altermagnetism_LLG.scripts.literature.nishino_miyashita_2015.run import PAPER
from altermagnetism_LLG.scripts.literature.nishino_miyashita_2015.analyze import collect


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream: json.dump(value,stream,indent=2)


def verify_snapshot(snapshot):
    for relative,digest in snapshot['files'].items():
        if sha256(ROOT/relative)!=digest: raise ValueError('frozen source changed: '+relative)


def freeze(base,phase,config_dir,commit,code_hash,selected_factor=None):
    doc=yaml.safe_load(yaml.safe_dump(base))
    doc['numerics'].update(code_commit=commit,code_sha256=code_hash,phase=phase)
    if selected_factor is not None: doc['numerics']['selected_factor']=selected_factor
    encoded=yaml.safe_dump(doc,sort_keys=True)
    import hashlib
    digest=hashlib.sha256(encoded.encode()).hexdigest()
    run_id=f"{datetime.now():%Y%m%d}_{commit[:7]}_{digest[:8]}"
    path=config_dir/(run_id+'.yaml')
    with path.open('x') as stream: stream.write(encoded)
    path.chmod(0o444)
    read_document(path)
    return run_id,path


def markdown_report(report,run_id):
    passed=sum(r['passed'] for r in report['rows'])
    return f"""# Nishino reduced LLG reproduction

_Run {run_id}; phase {report['phase']}; status {report['status']}_

---

## 📋 Scope and outcome

This is Fig. 1 case A/B only, not a five-system certificate. Passed numerical
conditions: {passed}/{len(report['rows'])}. Selected step factor: {report['selected_factor']}.
The primary method is the explicit midpoint of Appendix B5/B6 with a separately
declared final projection; raw predictor/final norm errors are reported.[^1]

## 📊 Acceptance

See [report.json](report.json) for every temperature, method, time-step,
stationarity CI, CDF effect size, Holm-adjusted p-value and paired convergence CI.
The mean criterion is abs(mean-exact)+95% CI half-width <= 0.01. CI samples are
independent magnetic moments, not correlated frames. Non-rejection of a CDF test
does not prove equality; the additional effect-size limit is required.
Temperature grid and repeat counts are project validation extensions.

## 🔐 Provenance and limits

Config SHA-256: {report['config_sha256']}.
Code commit: {report['code_commit']}; working-code SHA-256: {report['code_sha256']}.
The manifest links immutable raw files, derived results, frozen config and plots.
No failed trajectories are discarded. A failed gate prohibits primary production.
R2–R5 and training-dataset generation remain blocked.
The 2018 erratum repairs Fokker–Planck/precession and definition terms;
Fig. 1 settings and the Appendix B integration target are unchanged.[^2]

## 🔗 References

[^1]: Nishino and Miyashita (2015), Phys. Rev. B 91, 134411. https://doi.org/10.1103/PhysRevB.91.134411
[^2]: Nishino and Miyashita (2018), Erratum, Phys. Rev. B 97, 019904. https://doi.org/10.1103/PhysRevB.97.019904
"""


def run_phase(root,base,phase,config_dir,snapshot,workers,selected_factor=None):
    run_id,config=freeze(base,phase,config_dir,snapshot['git_commit'],snapshot['code_sha256'],selected_factor)
    runtime=load_runtime(config);extension=runtime.numerics['validation_extension']
    count=2*len(runtime.numerics['numerical_choice']['theta_grid'])*(extension['convergence_repetitions'] if phase=='convergence' else extension['repetitions'])
    raw_parent=root/'data/literature_reproduction'/PAPER/'raw'
    raw_parent.mkdir(parents=True,exist_ok=True);partial=raw_parent/(run_id+'.partial');final=raw_parent/run_id
    if final.exists(): raise FileExistsError(final)
    partial.mkdir()  # An exclusive campaign reservation, never reuse partial runs.
    output=root/'output/literature_reproduction'/PAPER/run_id;output.mkdir(parents=True)
    logs=root/'logs/literature_reproduction'/PAPER/run_id;logs.mkdir(parents=True)
    (output/'logs').symlink_to(os.path.relpath(logs,output),target_is_directory=True)
    write_json(output/'started.json',dict(run_id=run_id,phase=phase,config=str(config),
        config_sha256=sha256(config),tasks=count,workers=workers,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        snapshot=snapshot,finite_temperature_dataset_production_enabled=False))
    stopped=threading.Event()
    def task(index):
        if stopped.is_set(): return {'task':index,'status':'not_started_after_failure'}
        command=[sys.executable,str(ROOT/'scripts/literature'/PAPER/'run.py'),
            '--config',str(config),'--project-root',str(root),'--run-id',run_id,
            '--phase',phase,'--task',str(index),'--device','cpu']
        with (logs/f'task_{index:04d}.log').open('x') as stream:
            code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
        if code: stopped.set()
        print(json.dumps(dict(phase=phase,task=index,exit_code=code)),flush=True)
        return dict(task=index,status='complete' if not code else 'failed',exit_code=code)
    with ThreadPoolExecutor(max_workers=workers) as executor: tasks=list(executor.map(task,range(count)))
    write_json(output/'task_exit_codes.json',tasks)
    if any(t['status']!='complete' for t in tasks):
        write_json(output/'report.json',dict(status='fail',reason='incomplete task set',run_id=run_id))
        raise RuntimeError('task failure; raw remains partial and production is blocked')
    verify_snapshot(snapshot)
    if {p.name for p in partial.iterdir()}!={f'task_{i:04d}.h5' for i in range(count)}:
        raise ValueError('unexpected/missing raw files')
    _,raw_hashes=collect(partial,runtime,phase,sha256(config))
    partial.rename(final)
    analysis=[sys.executable,str(ROOT/'scripts/literature'/PAPER/'analyze.py'),
        '--config',str(config),'--project-root',str(root),'--run-id',run_id,'--phase',phase]
    result=subprocess.run(analysis)
    report=json.loads((output/'report.json').read_text())
    if result.returncode not in (0,2): raise RuntimeError('analysis crashed')
    derived=root/'data/literature_reproduction'/PAPER/'derived'/run_id
    assets=root/'assets/literature_reproduction'/PAPER/run_id
    subprocess.run([sys.executable,str(ROOT/'scripts/literature'/PAPER/'plot.py'),
        '--derived',str(derived/'summary.json'),'--reference',str(ROOT/'data/literature_reproduction'/PAPER/'reference'),
        '--output',str(assets)],check=True)
    (output/'report.md').write_text(markdown_report(report,run_id))
    manifest=dict(run_id=run_id,phase=phase,status=report['status'],config=str(config),
        config_sha256=sha256(config),code_commit=snapshot['git_commit'],code_sha256=snapshot['code_sha256'],
        raw_sha256=raw_hashes,raw_directory=str(final),
        artifact_sha256={str(p.relative_to(root)):sha256(p) for directory in [derived,assets,output]
            for p in directory.rglob('*') if p.is_file() and 'logs' not in p.relative_to(directory).parts},
        slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(output/'manifest.json',manifest)
    if report['status']!='pass': raise RuntimeError(f'{phase} statistical/convergence gate failed; no primary production')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--workers',type=int,required=True)
    a=p.parse_args();root=a.project_root.resolve();campaign=a.campaign.resolve()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('formal work requires a Slurm allocation')
    if not 1<=a.workers<=int(os.environ.get('SLURM_CPUS_PER_TASK','1')): raise ValueError('workers exceed allocation')
    snapshot=json.loads((campaign/'snapshot.json').read_text());verify_snapshot(snapshot)
    config_dir=root/'conf/frozen'/PAPER/campaign.name;config_dir.mkdir(parents=True)
    base=read_document(campaign/'input.yaml')
    preflight=json.loads((campaign/'preflight.json').read_text())
    if preflight['status']!='pass' or preflight['input_sha256']!=sha256(campaign/'input.yaml'):
        raise ValueError('missing or mismatched preflight')
    if base['numerics']['validation_extension']['production_enabled']:
        raise ValueError('cross-system production must stay disabled')
    destination=root/'output/literature_reproduction'/PAPER/campaign.name
    destination.mkdir(parents=True,exist_ok=True)
    try:
        convergence=run_phase(root,base,'convergence',config_dir,snapshot,a.workers)
        primary=run_phase(root,base,'primary',config_dir,snapshot,a.workers,convergence['selected_factor'])
        write_json(destination/'campaign_report.json',dict(status='pass',scope='R0/Nishino only',
            convergence_run=convergence['run_id'],primary_run=primary['run_id'],
            finite_temperature_dataset_production_enabled=False))
    except Exception as error:
        write_json(destination/'campaign_report.json',dict(status='fail',reason=str(error),
            traceback=traceback.format_exc(),finite_temperature_dataset_production_enabled=False))
        raise


if __name__=='__main__': main()
