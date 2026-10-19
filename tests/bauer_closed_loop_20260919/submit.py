"""Submit once, preserve IDs immediately, then leave the scheduler to execute."""
import json
import subprocess
from .run import HERE, ROOT, config, dump


def main():
    target = HERE/'submission.json'
    if target.exists():
        raise RuntimeError('Submission record already exists; inspect it before any resubmission')
    ids = {}
    phases = [('audit', 'fat', '00:30:00'), ('smoke', 'rtx4090', '00:30:00'),
              ('train', 'rtx4090', '12:00:00'), ('report', 'fat', '01:00:00')]
    for phase, partition, wall in phases:
        command = ['sbatch', '--parsable', '--job-name=zrs-mag', f'--partition={partition}',
                   '--nodes=1', '--ntasks=1', '--cpus-per-task=2', '--mem=8G', f'--time={wall}',
                   f'--chdir={ROOT}', f'--output={HERE}/logs/{phase}_%A_%a.out',
                   f'--error={HERE}/logs/{phase}_%A_%a.err', '--kill-on-invalid-dep=yes']
        if partition == 'rtx4090':
            command.append('--gres=gpu:1')
        if phase == 'smoke': command.append(f'--dependency=afterok:{ids["audit"]}')
        if phase == 'train':
            cfg = config()
            command.extend([f'--dependency=afterok:{ids["smoke"]}',
                            f'--array=0-{len(cfg["models"])*len(cfg["seeds"])-1}'])
        if phase == 'report': command.append(f'--dependency=afterany:{ids["train"]}')
        command.extend([str(HERE/'job.sh'), phase])
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        ids[phase] = result.stdout.strip().split(';')[0]
        dump(target, dict(jobs=ids, monitor=False, array_mapping=[f'{k}_{s}' for k in config()['models'] for s in config()['seeds']]))
        print(phase, ids[phase], flush=True)


if __name__ == '__main__':
    main()
