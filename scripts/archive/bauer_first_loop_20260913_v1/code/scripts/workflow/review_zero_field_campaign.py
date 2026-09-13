"""One-shot post-completion inventory, not a watcher or physics certifier."""
import argparse
import json
from pathlib import Path
import subprocess


def review(campaign):
    campaign=Path(campaign);m=json.loads((campaign/'manifest.json').read_text())
    submission=json.loads((campaign/'submission.json').read_text())
    ids=','.join(row['job_id'] for row in submission['jobs'])
    accounting=subprocess.check_output(['sacct','-j',ids,'-n','-P','-o',
        'JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS'],text=True)
    out=Path(m['output']);rows=[]
    for task in m['tasks']:
        record=out/'tasks'/f'{task["index"]:03d}.json'
        row=dict(index=task['index'],kind=task['kind'],paper=task.get('paper'),
                 run_id=task['run_id'],configuration=task['config'],raw=task['raw'])
        row['execution']=json.loads(record.read_text()) if record.exists() else dict(status='no_worker_completion_record_check_accounting')
        rows.append(row)
    gate=campaign/'equilibrium_gate.json'
    return dict(campaign=str(campaign),scheduler_accounting=accounting,tasks=rows,
        equilibrium_gate=json.loads(gate.read_text()) if gate.exists() else None,
        production_enabled=False,next_action='Independent scientific analysis; do not infer pass from COMPLETED. Preserve failed/partial runs.',
        warning='Nishino and legacy LLB have no resumable trajectory checkpoint; use a new run if interrupted. Other new reduced trajectory protocols retain exact RNG checkpoints.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=review(a.campaign)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2)
    print(str(a.output))
