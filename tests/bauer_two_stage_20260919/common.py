import json
import os
from pathlib import Path
import numpy as np
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def cfg():
    return yaml.safe_load((HERE/'config.yaml').read_text())


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def seed_stream(seed,stream):
    return int(np.random.SeedSequence([int(seed),int(stream)]).generate_state(1)[0])


def latent_seed(seed,stream):
    return int(np.random.SeedSequence([int(seed),int(stream)]).generate_state(1,dtype=np.uint64)[0]) & ((1<<63)-1)


def allocation(device=None):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Formal work requires Slurm')
    if device=='cuda' and os.environ.get('SLURM_JOB_PARTITION')!='rtx4090':
        raise RuntimeError('GPU work requires rtx4090 partition')


def old_cfg():
    from tests.bauer_closed_loop_20260919.run import config
    return config()
