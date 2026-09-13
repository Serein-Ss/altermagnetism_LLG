"""Revision-4 preparation and submission; no job monitoring or production release."""
import argparse
from datetime import datetime, timezone
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'scripts/archive/zero_field_20260911_040025/code/altermagnetism_LLG'
OLD_CONFIG = ROOT/'conf/frozen/zero_field_gomonay/20260911_040025/014.yaml'
FILES = ['scripts/workflow/bauer_revision4.py', 'scripts/analysis/bauer_revision4_audit.py',
         'scripts/analysis/bauer_features.py', 'scripts/model/bauer_gate_f.py',
         'scripts/model/gate_f.py', 'scripts/model/sphere.py',
         'scripts/literature/bauer_2011/model.py', 'scripts/validation/bauer_contract.py',
         'scripts/tests/test_bauer_gate_f.py', 'scripts/tests/test_bauer_revision4_audit.py',
         'scripts/tests/conftest.py', 'conf/bauer_path/draft.yaml', 'slurm/bauer_revision4.sbatch']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(obj, f, indent=2)


def prepare():
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out = ROOT/'output/bauer_revision4'/run_id
    frozen = ROOT/'conf/frozen/bauer_revision4'/run_id
    snapshot = ROOT/'scripts/archive'/('bauer_revision4_'+run_id)/'code'
    for folder in (out, frozen, snapshot): folder.mkdir(parents=True, exist_ok=False)
    files = FILES+[str(p.relative_to(ROOT)) for p in (ROOT/'GUIDE').glob('*.md')]
    hashes = {}
    for name in files:
        target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target); hashes[name] = digest(target)
    old_hashes = {str(p): digest(p) for p in OLD.rglob('*.py')}
    runs, timings = [], []
    for index, dt in enumerate((.02, .01)):
        c = yaml.safe_load(OLD_CONFIG.read_text())
        name = run_id+f'_bauer_dt_{index}'
        p = c['numerics']['protocols']['trajectory']
        p.update(dt=dt, steps=round(750000/dt), save_every=round(1./dt), seed=2026091300+index*100)
        c['numerics']['validation_extension']['campaign_kind'] = 'original_condition_timestep_diagnostic_not_certificate'
        c['numerics']['implementation_notes']['trajectory'] = 'duration=750000, save_dt=1 reduced; saving convergence still pending'
        path = frozen/f'dt_{index}.yaml'
        with path.open('x') as f: yaml.safe_dump(c, f)
        short = yaml.safe_load(path.read_text())
        short['numerics']['protocols']['trajectory'].update(steps=256, save_every=64)
        timing_config = frozen/f'timing_{index}.yaml'
        with timing_config.open('x') as f: yaml.safe_dump(short, f)
        command = [sys.executable, '-m', 'scripts.literature.bauer_2011.run', '--config', str(timing_config),
                   '--protocol', 'trajectory', '--run-id', name+'_timing', '--validation',
                   '--project-root', str(ROOT), '--device', 'cpu']
        started = time.monotonic()
        env = dict(os.environ, MKL_THREADING_LAYER='GNU', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        result = subprocess.run(command, cwd=OLD, capture_output=True, text=True, env=env)
        if result.returncode: raise RuntimeError(result.stderr)
        timings.append(dict(dt=dt, steps=256, elapsed_seconds=time.monotonic()-started,
                            scope='bounded_software_and_IO_timing_not_convergence'))
        runs.append(dict(run_id=name, dt=dt, config=str(path), config_sha256=digest(path),
                         path=str(ROOT/'data/literature_reproduction/bauer_2011/raw'/name/'trajectory.h5')))
    # Long-run observed rate includes real checkpoint and I/O overhead, unlike tiny timings.
    checkpoint = ROOT/'data/literature_reproduction/bauer_2011/raw/20260911_040025_014.partial/trajectory.h5.checkpoint.h5'
    with h5py.File(checkpoint, 'r') as h: completed = int(h.attrs['step'])
    state = subprocess.check_output(['sacct', '-X', '-n', '-j', '669197_14', '--format=ElapsedRaw', '-P'], text=True)
    elapsed = int(state.strip().split('|')[0])
    if completed <= 0: raise RuntimeError('missing measured throughput')
    rate = elapsed/completed
    requests = [math.ceil((rate*round(750000/r['dt'])*1.5+3600)/60) for r in runs]
    storage_gib = 2*(750001*8*100*3*8)*2/2**30  # two new runs, 2x reserve
    cpu_hours = sum(requests)/60+2
    if max(requests) > 7*24*60 or cpu_hours > 256 or shutil.disk_usage(ROOT).free < (storage_gib+10)*2**30:
        raise RuntimeError('resource envelope exceeded; no tasks submitted')
    runs.append(dict(run_id='20260911_040025_014', dt=.005, config=str(OLD_CONFIG),
                     config_sha256=digest(OLD_CONFIG),
                     path=str(ROOT/'data/literature_reproduction/bauer_2011/raw/20260911_040025_014/trajectory.h5')))
    m = dict(run_id=run_id, output=str(out), snapshot=str(snapshot), project_root=str(ROOT),
             wrapper=str(ROOT.parent/'run_zrs_mag.sh'), files_sha256=hashes,
             reference_snapshot=str(OLD), reference_code_sha256=old_hashes,
             reference_runs=runs, existing_reference_job='669197_14',
             protocol_commit=subprocess.check_output(['git','rev-parse','FETCH_HEAD'], text=True).strip(),
             resources=dict(measured_old_seconds_per_step=rate, reference_steps_observed=completed,
                  reference_elapsed_seconds=elapsed, short_timings=timings, requested_minutes=requests,
                  new_cpu_hours_requested=cpu_hours, storage_bound_gib=storage_gib,
                  concurrency_limit=None, retries=0, software_gpu_hours_upper=.5),
             production_enabled=False, training_enabled=False,
             design=dict(conditions='Bauer original L100,K/J=.1,theta=.11,lambda=.1',
                  replicas_per_dt=8, initial='same all-positive full spin state',
                  noise='independent weak discrete draws across dt; no Wiener strong-coupling claim',
                  purpose='complete mandated .02/.01/.005 timestep references; not event power certification'),
             pending=['Original lifetime/length/temperature reference acceptance and >=500 event requirement',
                      'P2 measured basins, correlation length, dwell/time resolution and mechanism',
                      'P2 sample sizes, event tolerances and exact size holdouts not yet frozen',
                      'Certificate/data-bound Bauer training/inference and complete statistical acceptance entry',
                      'No automatic P2 extension, training, or AM production'])
    write(frozen/'manifest.json', m)
    write(out/'inventory.json', dict(status='P0_software_and_reference_preparation',
          protocol_commit=m['protocol_commit'], historical_gomonay_status='inconclusive_preserved_not_mainline_release',
          resources=m['resources'], design=m['design'], pending=m['pending']))
    print(json.dumps(dict(manifest=str(frozen/'manifest.json'), resources=m['resources']), indent=2))


def worker(path, stage, index):
    m = json.loads(Path(path).read_text())
    for name, sha in m['files_sha256'].items():
        if digest(Path(m['snapshot'])/name) != sha: raise ValueError('snapshot changed: '+name)
    if stage in ('cpu', 'cuda'):
        if stage == 'cuda':
            import torch
            if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
                raise RuntimeError('Slurm CUDA allocation required')
        folder = Path(m['output'])/('software_'+stage); folder.mkdir(exist_ok=False)
        env = dict(os.environ, GATE_F_TEST_DEVICE=stage)
        command = [sys.executable, '-m', 'pytest', '-q', '-s', '-p', 'no:cacheprovider',
                   'scripts/tests/test_bauer_gate_f.py', 'scripts/tests/test_bauer_revision4_audit.py',
                   '--junitxml='+str(folder/'junit.xml')]
        with (folder/'pytest.log').open('x') as f:
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT)
        write(folder/'result.json', dict(status='pass' if result.returncode==0 else 'fail',
              scope='synthetic_software_only_no_physics_certificate', training_started=False,
              device=stage, returncode=result.returncode))
        raise SystemExit(result.returncode)
    if stage == 'reference':
        row = m['reference_runs'][index]
        if index not in (0, 1): raise ValueError('do not duplicate existing reference')
        for name, sha in m['reference_code_sha256'].items():
            if digest(name) != sha: raise ValueError('reference source changed: '+name)
        if digest(row['config']) != row['config_sha256']: raise ValueError('reference configuration changed')
        command = [sys.executable, '-m', 'scripts.literature.bauer_2011.run', '--config', row['config'],
                   '--protocol', 'trajectory', '--run-id', row['run_id'], '--validation',
                   '--project-root', m['project_root'], '--device', 'cpu']
        os.chdir(m['reference_snapshot'])
        os.execv(sys.executable, command)
    if stage == 'review':
        subprocess.run([sys.executable, '-m', 'scripts.analysis.bauer_revision4_audit', '--manifest', str(path)], check=True)


def submit(path):
    path = Path(path).resolve(); m = json.loads(path.read_text())
    record = Path(m['output'])/'submission.json'
    write(record, dict(status='submitting', jobs=[]))
    logs = Path(m['project_root'])/'logs/bauer_revision4'/m['run_id']; logs.mkdir(parents=True, exist_ok=False)
    jobs = []
    def launch(stage, minutes, index=0, dependency=None):
        command = ['sbatch', '--parsable', '--partition='+('rtx4090' if stage=='cuda' else 'fat'),
                   '--time='+str(minutes), '--kill-on-invalid-dep=yes',
                   '--output='+str(logs/(f'{stage}_{index}_%j.log'))]
        if stage == 'cuda': command += ['--gres=gpu:1']
        if dependency: command += ['--dependency='+dependency]
        command += [str(Path(m['snapshot'])/'slurm/bauer_revision4.sbatch'), m['snapshot'], m['wrapper'], str(path), stage, str(index)]
        response = subprocess.run(command, capture_output=True, text=True)
        if response.returncode: raise RuntimeError(response.stderr)
        job = response.stdout.strip().split(';')[0]
        if not job.isdigit(): raise RuntimeError('unexpected submission response; inspect before retry')
        jobs.append(dict(stage=stage,index=index,job_id=job,command=command))
        temporary = record.with_suffix('.json.partial')
        temporary.write_text(json.dumps(dict(status='partial_submission', jobs=jobs, monitor=False), indent=2)); temporary.replace(record)
        return job
    launch('cpu', 30)
    launch('cuda', 30)
    references = [launch('reference', minutes, i) for i, minutes in enumerate(m['resources']['requested_minutes'])]
    launch('review', 120, dependency='afterany:'+':'.join([m['existing_reference_job'], *references]))
    temporary = record.with_suffix('.json.partial')
    temporary.write_text(json.dumps(dict(status='submitted_not_monitored',jobs=jobs,
         production_enabled=False,training_enabled=False),indent=2)); temporary.replace(record)
    print(json.dumps(dict(submission=str(record), jobs=[{k:j[k] for k in ('stage','index','job_id')} for j in jobs]), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true'); p.add_argument('--submit', action='store_true')
    p.add_argument('--manifest'); p.add_argument('--stage', choices=['cpu','cuda','reference','review'])
    p.add_argument('--index', type=int, default=0); args = p.parse_args()
    if args.prepare: prepare()
    elif args.submit: submit(args.manifest)
    else: worker(args.manifest, args.stage, args.index)


if __name__ == '__main__': main()
