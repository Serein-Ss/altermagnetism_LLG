import argparse
import json
import os
from pathlib import Path
import torch
from tests.experiments.path_learning_audit import fixed_pair


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True)
    p.add_argument('--device',choices=['cuda'],default='cuda');a=p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    cfg=json.loads(Path('tests/conf/velocity_capacity.json').read_text())
    if not 0<=a.index<9:raise ValueError('unknown experiment')
    variant=cfg['variants'][a.index//3];seed=cfg['seeds'][a.index%3]
    torch.set_num_threads(1)
    fixed_pair(seed,iterations=cfg['iterations'],device=a.device,variant=variant,
        root=f'tests/results/velocity_capacity/{variant}',output_root=f'tests/output/velocity_capacity/{variant}')
