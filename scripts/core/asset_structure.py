"""Canonical result destinations; live September 13 tasks retain their paths."""
from pathlib import Path

PREFIX = 'assets/literature_reproduction/'
LIVE = (
    PREFIX+'bauer_2011/20260911_040025_014/',
    PREFIX+'bauer_2011/20260913_023043_bauer_dt_0/',
    PREFIX+'bauer_2011/20260913_023043_bauer_dt_1/',
    PREFIX+'bauer_2011/runs/bauer_revision4/20260913_023043/',
)
BOOKKEEPING = {'manifest.json', 'submission.json', 'completed.json', 'artifact_manifest.json',
               'inventory.json', 'resource_budget.json', 'resource_estimate.json',
               'cpu_software_preflight.json', 'implementation_status.json', 'delivery_manifest.json'}


def destination(value):
    p = Path(value); s = p.as_posix()
    if not s.startswith('assets/') or s.startswith(LIVE):
        return s
    if 'numba_cache' in p.parts:
        return 'logs/cache/asset_migration_20260913/' + s.removeprefix('assets/')
    if p.suffix == '.sbatch':
        return 'slurm/archive/assets_20260913/' + s.removeprefix('assets/')
    if p.suffix == '.py':
        return 'scripts/archive/assets_20260913/' + s.removeprefix('assets/')
    if p.name in BOOKKEEPING or 'runner_status' in p.parts:
        return 'logs/assets_20260913/' + s.removeprefix('assets/')
    first = 'assets/research/runs/bauer_first_loop/'
    if s.startswith(first):
        run, *tail = Path(s[len(first):]).parts
        base = f'assets/research/bauer_L25_first_loop/{run}/'
        if tail[0] == 'figures':
            return base+'model_comparison/'+p.name
        if tail[0] == 'reference_visualization':
            return base+'reference_data/'+p.name
        return base+'analysis/'+p.name
    if s.startswith('assets/research/runs/zero_field_program/'):
        return s.replace('assets/research/runs/', 'logs/research/', 1)
    if s.startswith('assets/research/runs/gate_f/'):
        return s.replace('assets/research/runs/', 'logs/research/', 1)
    if not s.startswith(PREFIX):
        return s
    paper, *tail = Path(s[len(PREFIX):]).parts
    if not tail or tail[0] in ('reference', 'experiments'):
        return s
    if paper == 'r2_r5_analysis_20260910':
        return PREFIX+'cross_paper/experiments/reproduction_validation/20260910_v1/analysis/'+p.name
    if tail[0] == 'runs':
        tail = tail[1:]
    run = tail[0]
    experiment = 'reproduction_validation'
    version = run
    if run == 'r2_r5_analysis_20260910':
        version = '20260910_v1'
        if paper == 'gomonay_2024':
            experiment = 'spinwave_validation'
    elif paper == 'nishino_miyashita_2015':
        experiment = 'figure1_validation'
        if run.startswith('20260909_'):
            version = '20260909_v1'
        elif run.startswith('20260910_'):
            version = '20260910_v1'
        else:
            experiment = {'preflight_20260909': 'noise_validation',
                          'r0_20260909_8df33ac_6fd7226f': 'r0_summary',
                          'r1_20260910_user_accepted_bb48dc64': 'r1_summary'}.get(run, 'figure1_validation')
            if experiment != 'figure1_validation':
                version = '20260909_v1' if experiment != 'r1_summary' else '20260910_v1'
    elif paper == 'bauer_2011' and run == 'bauer_campaign':
        version = tail[1]
        if 'software' in tail:
            return 'logs/assets_20260913/'+s.removeprefix('assets/')
        if 'physics' in tail:
            experiment = 'physics_checks'
        elif 'reference' in tail:
            experiment = 'reference_dynamics'
            version += '_'+tail[tail.index('reference')+1]
        else:
            experiment = 'pilot_switching'
    elif paper == 'bauer_2011' and run == '20260910_8df33ac_060e74d3':
        experiment, version = 'preflight', '20260910_v1'
    category = 'figures' if p.suffix.lower() == '.png' else 'animations' if p.suffix.lower() in ('.gif', '.mp4', '.webm') else 'analysis'
    return f'{PREFIX}{paper}/experiments/{experiment}/{version}/{category}/{p.name}'
