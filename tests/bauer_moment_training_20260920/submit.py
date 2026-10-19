"""Freeze current controlled experiment and submit once after verified smoke."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    record = HERE/'submission.json'
    if record.exists(): raise RuntimeError('Already submitted; inspect recorded job')
    for arm in ['w0_11', 'w1_11']:
        completed = json.loads((HERE/'train_smoke'/arm/'complete.json').read_text())
        if not completed['smoke'] or completed['updates'] != 2:
            raise RuntimeError('Full pipeline smoke not verified')
    paths = [HERE/name for name in ['config.yaml', 'train.py', 'objective.py', 'data.py', 'train.sh']]
    paths += [HERE.parent/'bauer_transition_20260920/runs'/f'transition_{seed}'/'best.pt' for seed in [11, 22, 33]]
    lines = [hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT)) for p in paths]
    (HERE/'submission.sha256').write_text('\n'.join(lines)+'\n')
    job = subprocess.check_output(['sbatch', '--parsable', str(HERE/'train.sh')], cwd=ROOT, text=True).strip().split(';')[0]
    record.write_text(json.dumps(dict(train_array=job, arms=2, seeds=[11,22,33], updates=1500,
        scope='Controlled diagnostic training; no publication certification'), indent=2)+'\n')
    print(job)


if __name__ == '__main__':
    main()
