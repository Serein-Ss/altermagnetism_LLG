"""Freeze source, submit two bounded software jobs, then exit without polling."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from scripts.validation.gate_f_contract import digest

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    'scripts/model/gate_f.py', 'scripts/model/sphere.py',
    'scripts/core/cell_hamiltonian.py', 'scripts/core/literature_config.py',
    'scripts/literature/gomonay_2024/model.py',
    'scripts/validation/gate_f_contract.py', 'scripts/validation/zero_field_certificates.py',
    'scripts/validation/run_gate_f_software.py', 'scripts/datasets/gate_f_paths.py',
    'scripts/analysis/gate_f_metrics.py', 'scripts/training/train_gate_f.py',
    'scripts/inference/sample_gate_f.py', 'scripts/tests/test_gate_f.py',
    'scripts/tests/conftest.py', 'conf/gate_f/draft.yaml', 'slurm/gate_f_software.sbatch',
    'scripts/workflow/submit_gate_f_software.py',
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    if not args.submit:
        print(json.dumps(dict(plan='two software jobs: fat CPU and one rtx4090 GPU',
                              minutes_each=30, production_enabled=False, files=FILES), indent=2))
        return
    run = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    base = ROOT/'output/gate_f/software'/run
    snapshot = ROOT/'scripts/archive'/('gate_f_software_'+run)/'code'
    base.mkdir(parents=True, exist_ok=False)
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in FILES:
        destination = snapshot/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
        hashes[name] = digest(destination)
    logs = ROOT/'logs/gate_f'/run
    logs.mkdir(parents=True, exist_ok=False)
    wrapper = ROOT.parent/'run_zrs_mag.sh'
    manifest = dict(scope='software_preparation_only', run_id=run, snapshot=str(snapshot),
                    files_sha256=hashes, wrapper_sha256=digest(wrapper), jobs=[],
                    production_enabled=False, monitor=False)
    path = base/'submission.json'
    def record():
        temporary = path.with_suffix('.json.partial')
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(path)
    record()
    for device, partition in [('cpu', 'fat'), ('cuda', 'rtx4090')]:
        command = ['sbatch', '--parsable', '--partition='+partition,
                   '--output='+str(logs/(device+'_%j.log'))]
        if device == 'cuda':
            command += ['--gres=gpu:1']
        command += [str(snapshot/'slurm/gate_f_software.sbatch'), str(snapshot),
                    str(wrapper), device, str(base/device)]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            manifest['status'] = 'partial_submission_failed_no_auto_retry'
            manifest['submission_error'] = result.stderr
            record()
            raise RuntimeError('submission failed; earlier accepted jobs retained: '+str(path))
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise RuntimeError('unexpected scheduler response; do not blindly resubmit')
        manifest['jobs'].append(dict(job_id=job_id, device=device, command=command))
        record()
    manifest['status'] = 'submitted_not_monitored'
    record()
    print(json.dumps(dict(submission=str(path), jobs=manifest['jobs'], status=manifest['status']), indent=2))


if __name__ == '__main__':
    main()
