"""Physical destinations for the 2026-09-13 workspace migration (no aliases)."""
from pathlib import Path

PREFIXES = {
    'data/datasets': 'data/research/datasets',
    'data/generated': 'data/research/generated',
    'data/literature': 'data/literature_reproduction/legacy',
    'data/audit': 'assets/research/audit_data',
    'assets/literature': 'assets/literature_reproduction/legacy',
    'conf/literature': 'conf/literature_reproduction/physics',
    'conf/bauer_campaign': 'conf/literature_reproduction/campaigns/bauer',
    'conf/bauer_path': 'conf/research/models/bauer_path',
    'conf/gate_f': 'conf/research/experiments/gate_f',
    'conf/zero_field_gomonay': 'conf/research/experiments/zero_field_gomonay',
    'conf/frozen': 'conf/archive/frozen',
}


def destination(value):
    """Map a repository-relative file; do not follow symlinks or modify payloads."""
    p = Path(value); s = p.as_posix(); parts = p.parts
    if not parts or parts[0] in ('.git', 'GUIDE', 'docs') or s == 'README.md':
        return s
    if p.suffix.lower() == '.md':
        category = 'history' if s.startswith('scripts/archive/') else 'reports' if parts[0] == 'output' else 'reference'
        return f'docs/{category}/{s}'
    if s.startswith('scripts/archive/') and p.suffix not in ('.py', '.pyc'):
        category = 'conf/archive/extracted' if p.suffix in ('.yaml', '.yml', '.json', '.sha256') else 'slurm/archive/extracted' if p.suffix in ('.sbatch', '.sh') else 'docs/attachments'
        return f'{category}/{s}'
    if p.suffix in ('.log', '.err', '.out', '.xml') and parts[0] not in ('logs', 'GUIDE'):
        return 'logs/local/' + s
    if len(parts) == 1 and p.suffix == '.json':
        return 'logs/restructure/history/' + s
    if parts[0] == 'output':
        tail = '/'.join(parts[1:])
        paper = 'bauer_2011' if parts[1] in ('bauer_campaign', 'bauer_revision4') else None
        if parts[1] == 'literature_reproduction':
            scope = 'literature_reproduction/' + parts[2]
            tail = '/'.join(parts[3:])
        elif paper:
            scope = 'literature_reproduction/' + paper
        else:
            scope = 'research'
        if p.suffix in ('.yaml', '.yml') or p.name in ('config.json', 'training_config.json'):
            return f'conf/{scope}/runs/{tail}'
        is_data = '.h5' in p.name or p.suffix == '.npy' or p.name in ('development.npz', 'checkpoint.npz')
        if parts[1] == 'bauer_campaign' and 'reference' in parts and p.name in ('LFS_REASSEMBLY.txt', 'LFS_SPLIT_MANIFEST.json'):
            is_data = True
        if is_data:
            return f'data/{scope}/runs/{tail}'
        if scope == 'research' and (p.suffix in ('.pt', '.pth', '.ckpt') or p.name in ('history.json', 'history.csv', 'training_history.json', 'training.json', 'training_metrics.json')):
            return s
        return f'assets/{scope}/runs/{tail}'
    for old, new in sorted(PREFIXES.items(), key=lambda x: -len(x[0])):
        if s == old or s.startswith(old + '/'):
            return new + s[len(old):]
    return s
