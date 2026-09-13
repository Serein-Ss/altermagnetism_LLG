"""Validate, freeze and estimate R0; no job submission or deletion in this tool."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import h5py
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from altermagnetism_LLG.scripts.core.literature_config import read_document,sha256

PAPER='nishino_miyashita_2015'


def main():
    config=ROOT/'conf/literature'/f'{PAPER}.yaml';base=read_document(config)
    for file in sorted((ROOT/'conf/literature').glob('*.yaml')): read_document(file)
    paper=base['numerics']['paper_settings']
    if paper!={'moments':1000,'dt':.005,'total_steps':80000,'burn_steps':40000}:
        raise ValueError('strict Fig.1 paper protocol changed')
    timing_files=[ROOT/'data/literature_reproduction'/PAPER/'raw'/directory/file for directory,file in [
        ('preflight_20260909_timing.partial','task_0000.h5'),
        ('preflight_20260909_timing_b.partial','task_0048.h5')]]
    timing=[]
    for file in timing_files:
        with h5py.File(file,'r') as h5:
            meta=json.loads(h5.attrs['metadata'])
            if not h5.attrs['complete'] or meta['total_steps']!=10000 or meta['moments']!=1000 or meta['formal_settings']:
                raise ValueError('invalid 10000-step timing record')
            if meta['config_sha256']!=sha256(config): raise ValueError('timed configuration changed')
        timing.append(dict(path=str(file.relative_to(ROOT)),sha256=sha256(file),metadata=meta))
    noise=ROOT/'output/literature_reproduction'/PAPER/'preflight_20260909/noise_gate.json'
    noise_result=json.loads(noise.read_text())
    if noise_result['status']!='pass' or noise_result['config_sha256']!=sha256(config):
        raise ValueError('noise gate failed or config changed')
    # Real regression tests, not a manually supplied "pass" flag.
    test=subprocess.run([sys.executable,'-m','pytest','scripts/tests','-q','--disable-warnings'],
                        cwd=ROOT,capture_output=True,text=True)
    print(test.stdout,flush=True)
    if test.returncode: print(test.stderr);raise RuntimeError('regression tests failed')
    source_files=[
        'scripts/core/literature_config.py','scripts/core/reduced_llg.py',
        'scripts/config/validate_literature_configs.py',
        'scripts/config/build_literature_reference_manifest.py',
        'scripts/workflow/clean_obsolete_nishino.py',
        'scripts/literature/nishino_miyashita_2015/model.py',
        'scripts/literature/nishino_miyashita_2015/run.py',
        'scripts/literature/nishino_miyashita_2015/analyze.py',
        'scripts/literature/nishino_miyashita_2015/plot.py',
        'scripts/workflow/run_reduced_literature_campaign.py',
        'scripts/workflow/prepare_reduced_campaign.py',
        'scripts/validation/reduced_noise_gate.py',
        'scripts/tests/conftest.py','scripts/tests/test_reduced_literature.py',
        'scripts/tests/test_reduced_workflow.py','slurm/reduced_literature_r0.sbatch',
        'GUIDE/STRICT_LITERATURE_REPRODUCTION_PLAN.md','requirements.txt']
    source_files+=['conf/reduced_llg.yaml']+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'conf/literature').glob('*.yaml'))]
    source_files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'data/literature_reproduction'/PAPER/'reference').iterdir()) if p.is_file()]
    files={file:sha256(ROOT/file) for file in source_files}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    campaign_id=f"r0_{datetime.now():%Y%m%d}_{commit[:7]}_{digest[:8]}"
    campaign=ROOT/'scripts/archive'/campaign_id
    campaign.mkdir()
    snapshot=campaign/'code/altermagnetism_LLG'
    for file in source_files:
        dest=snapshot/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,dest)
        dest.chmod(0o444)
    shutil.copy2(config,campaign/'input.yaml');(campaign/'input.yaml').chmod(0o444)
    record=dict(git_commit=commit,code_sha256=digest,files=files,
        note='Commit plus working-tree content hash; does not claim uncommitted code is in the commit.',
        source_root=str(snapshot))
    (campaign/'snapshot.json').write_text(json.dumps(record,indent=2))
    workers=20
    # Conservative bound: charge all nine-variant timing cost even for primary
    # (which has only one selected variant). Margin also covers I/O and plotting.
    seconds=max(row['metadata']['elapsed_seconds'] for row in timing)
    convergence_waves=math.ceil(96/workers);primary_waves=math.ceil(240/workers)
    estimate=seconds*8*(convergence_waves+primary_waves)*base['numerics']['validation_extension']['walltime_margin']+1800
    minutes=math.ceil(estimate/60)
    preflight=dict(status='pass',input_sha256=sha256(campaign/'input.yaml'),
        regression_output=test.stdout,regression_stderr=test.stderr,
        timing=timing,noise_gate_path=str(noise.relative_to(ROOT)),noise_gate_sha256=sha256(noise),
        resource_estimate=dict(workers=workers,memory_GiB=64,walltime_minutes=minutes,
            timing_seconds_worst=seconds,convergence_tasks=96,conditional_primary_tasks=240,
            assumption='10000 coarse steps of all three methods and all three dt factors; primary deliberately overestimated',
            node_partition='fat'))
    (campaign/'preflight.json').write_text(json.dumps(preflight,indent=2))
    versions=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (campaign/'environment.txt').write_text(versions)
    result=dict(campaign=str(campaign),snapshot=record,preflight=preflight,
        submit_command=['sbatch','--hold',f'--time={minutes}',str(ROOT/'slurm/reduced_literature_r0.sbatch'),str(ROOT),str(campaign)])
    output=ROOT/'output/literature_reproduction'/PAPER/campaign_id;output.mkdir(parents=True)
    (output/'preparation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(campaign=str(campaign),walltime_minutes=minutes,submit_command=result['submit_command'])),flush=True)


if __name__=='__main__': main()
