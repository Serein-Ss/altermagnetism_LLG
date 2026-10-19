"""Freeze stratified parent assignments before any thermal parent is generated."""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import yaml
from tests.bauer_two_stage_20260919.common import dump

HERE=Path(__file__).resolve().parent
RANDOMIZATION=Path('/share/home/xlzou/WORKSPACE/rszhong/.codex/skills/experimental-design/scripts/randomization.py')


def build(cfg):
    spec=importlib.util.spec_from_file_location('research_randomization',RANDOMIZATION)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    strata={f'{a}x{b}':cfg['parents_per_size'] for a,b in cfg['cell_shapes']}
    schedule=module.stratified_block_randomization(strata,arms=['train','dev','test'],
        block_size=5,ratio=cfg['split_ratio'],seed=cfg['layout_seed'])
    rows=[];seeds=set()
    for index,row in schedule.iterrows():
        shape=[int(n) for n in row['stratum'].split('x')]
        streams=np.random.SeedSequence([cfg['noise_seed_root'],int(index)]).generate_state(
            cfg['branches_per_parent']+1,dtype=np.uint64)
        streams=[int(n)&((1<<63)-1) for n in streams]
        if len(set(streams))!=len(streams) or seeds.intersection(streams):raise RuntimeError('Seed collision')
        seeds.update(streams)
        name=f'parent_{index:03d}'
        rows.append(dict(parent_id=name,split=str(row['arm']),cell_shape=shape,
            preparation_group=f'thermal_preparation_{streams[0]}',preparation_seed=streams[0],
            noise_seeds=streams[1:],file=f'{name}.h5',allocation_block=int(row['block'])))
    return dict(schema='miltat_parent_v1',parents=rows,
        design='Stratified permuted blocks by size, ratio train:dev:test=3:1:1',
        randomization_source_sha256=hashlib.sha256(RANDOMIZATION.read_bytes()).hexdigest(),
        scope=cfg['scope'],scientific_certification=False)


if __name__=='__main__':
    target=HERE/'layout.json'
    if target.exists():raise RuntimeError('Layout already frozen')
    dump(target,build(yaml.safe_load((HERE/'config.yaml').read_text())))
