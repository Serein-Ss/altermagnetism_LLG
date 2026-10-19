"""Full-training-set whole-path versus short-window conditional learning."""
import argparse
import json
from pathlib import Path
import torch
from scripts.workflow.bauer_campaign import save
from tests.experiments.bauer_budget_comparison import run


def main(index,plan_path='tests/conf/window_learning.json'):
    plan=json.loads(Path(plan_path).read_text())
    experiment=plan.get('experiment','window_learning')
    arms=list(plan['arms']);n=len(plan['seeds'])
    if not 0<=index<len(arms)*n:raise ValueError('unknown arm')
    arm=arms[index//n];seed=plan['seeds'][index%n]
    cfg=json.loads(Path(plan['base_experiment']).read_text())
    cfg.update(model='cartesian',window_frames=plan['arms'][arm],compare_legacy=False,
        budgets=plan['budgets'],scope=plan['scope'],source_scale=plan.get('source_scale'),optimization=plan.get('optimization'),
        controls='Same Cartesian width64 blocks4 and initial weights per seed; Adam lr.001, batch8, uniform flow matching. Same update budget, unequal frame/compute budget.',
        interpretation='Compare development full-path distributions, not losses across unequal windows; no promotion or scale-up based only on loss.',
        output=f'tests/output/{experiment}/{arm}',raw=f'tests/data/{experiment}/{arm}',
        analysis=f'tests/results/{experiment}/{arm}')
    if plan.get('optimization'):
        cfg['controls']='Same Cartesian backbone, data and batch8; longer-budget cosine/EMA optimization recipe, not an equal-compute or single-factor comparison.'
    path=Path(f'tests/conf/{experiment}_runs/{arm}_{seed}.json')
    if path.exists():raise FileExistsError(path)
    save(path,cfg);run(path,seed,'cuda')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True)
    p.add_argument('--plan',default='tests/conf/window_learning.json')
    args=p.parse_args();torch.set_num_threads(1);main(args.index,args.plan)
