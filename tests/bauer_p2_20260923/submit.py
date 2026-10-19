"""Freeze dependencies, submit 3 GPU seeds and one afterany report; never poll."""
import hashlib
import json
import subprocess
from .common import HERE, ROOT, cfg, dump


def main():
    if (HERE/'submission.json').exists():
        raise RuntimeError('Already submitted: inspect recorded IDs, do not duplicate')
    assert json.loads((HERE/'audit.json').read_text())['integrity_pass']
    log=(HERE/'logs/tests.log').read_text()
    if 'failed' in log or 'passed' not in log:
        raise RuntimeError('regression suite not passing')
    benchmark=json.loads((HERE/'benchmark/benchmark.json').read_text())
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.yaml'))+list(HERE.glob('*.sh'))
    files += [ROOT/p for p in ('tests/spin_graph_20260920/model.py',
        'tests/mdgen_path_refactor_20260918/data.py','tests/bauer_closed_loop_20260919/run.py',
        'scripts/model/gate_f.py','scripts/model/sphere.py','scripts/core/cell_hamiltonian.py')]
    files += list(HERE.glob('manifest_*.json'))+[HERE/'audit.json',HERE/'numerical_reference.json']
    dump(HERE/'source_lock.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    gpu=subprocess.check_output(['sbatch','--parsable','--array=0-2',str(HERE/'job.sh')],cwd=ROOT,text=True).strip().split(';')[0]
    submission=dict(training_array=gpu,seeds=cfg()['seeds'],report_job=None,
        estimated_per_seed_seconds=benchmark['estimated_train_eval_seconds'],
        estimate_excludes_queue=True,partition='rtx4090',job_name='zrs-mag',
        status='SUBMITTED_NOT_COMPLETED',monitor_after_submission=False)
    dump(HERE/'submission.json',submission)
    report=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{gpu}',str(HERE/'report_job.sh')],cwd=ROOT,text=True).strip().split(';')[0]
    submission['report_job']=report; dump(HERE/'submission.json',submission)
    print(json.dumps(submission,ensure_ascii=False))


if __name__=='__main__':
    main()
