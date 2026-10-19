"""One-time recoverable relocation of active tests and diagnostic artifacts."""
from pathlib import Path
import difflib
import shutil
import subprocess
import json
from scripts.workflow.bauer_campaign import save, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    mapping = {'scripts/tests': 'tests/unit',
        'scripts/analysis/bauer_checkpoint_diagnostics.py': 'tests/experiments/bauer_checkpoint_diagnostics.py',
        'scripts/training/bauer_budget_comparison.py': 'tests/experiments/bauer_budget_comparison.py',
        'scripts/inference/bauer_window.py': 'tests/experiments/bauer_window.py',
        'output/bauer_training_budget': 'tests/output/bauer_training_budget',
        'output/smoke': 'tests/output/smoke',
        'assets/research/bauer_training_budget': 'tests/results/bauer_training_budget',
        'data/research/bauer_training_budget': 'tests/data/bauer_training_budget',
        'data/research/generated/diagnostics': 'tests/data/legacy_diagnostics'}
    for version in ('20260913', '20260913_v2'):
        mapping[f'conf/research/bauer_diagnostics_{version}.json'] = f'tests/conf/bauer_diagnostics_{version}.json'
        mapping[f'data/research/diagnostics/bauer_{version}'] = f'tests/data/bauer_{version}'
        mapping[f'assets/research/bauer_L25_first_loop/20260913_v1/analysis/checkpoint_diagnostics_{version}'] = f'tests/results/checkpoint_diagnostics_{version}'
        mapping[f'logs/research/bauer_diagnostics_{version}'] = f'tests/logs/bauer_diagnostics_{version}'
    mapping['conf/research/bauer_training_budget_20260913.json'] = 'tests/conf/bauer_training_budget_20260913.json'
    for name in ('bauer_training_budget', 'bauer_checkpoint_diagnostics'):
        mapping[f'slurm/{name}.sbatch'] = f'tests/slurm/{name}.sbatch'
    for name in ('bauer_training_budget_20260913', 'gpu_partition_policy'):
        mapping[f'logs/research/{name}'] = f'tests/logs/{name}'
    manifest = ROOT/'tests/relocation.json'
    existing = json.loads(manifest.read_text()) if manifest.exists() else None
    if existing and existing['status'] != 'MOVING':
        raise FileExistsError('migration already complete')
    inventory = []
    for old, new in mapping.items():
        src, dst = ROOT/old, ROOT/new
        if not src.exists():
            continue
        if dst.exists():
            raise FileExistsError(dst)
        for p in ([src] if src.is_file() else src.rglob('*')):
            if p.is_file():
                inventory.append(dict(old=str(p.relative_to(ROOT)),
                    new=str(Path(new)/(p.relative_to(src))) if src.is_dir() else new, sha256=digest(p)))
    inventory = existing['files'] if existing else inventory
    save(manifest, dict(status='MOVING', mapping=mapping, files=inventory))
    for old, new in mapping.items():
        src, dst = ROOT/old, ROOT/new
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(src, dst)
    for row in inventory:
        assert digest(ROOT/row['new']) == row['sha256'], row['new']
    replacements = dict(mapping)
    replacements.update({'scripts.tests': 'tests.unit',
        'scripts.analysis.bauer_checkpoint_diagnostics': 'tests.experiments.bauer_checkpoint_diagnostics',
        'scripts.training.bauer_budget_comparison': 'tests.experiments.bauer_budget_comparison',
        'scripts.inference.bauer_window': 'tests.experiments.bauer_window'})
    candidates = [ROOT/'pytest.ini']
    for folder in ('scripts', 'slurm', 'tests'):
        candidates += [p for p in (ROOT/folder).rglob('*') if p.suffix in ('.py', '.sbatch', '.sh')
                       and 'archive' not in p.parts and p != Path(__file__).resolve()]
    candidates += list((ROOT/'tests/conf').glob('*.json'))
    changed = []
    for p in candidates:
        old = p.read_text(); new = old
        for a, b in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
            new = new.replace(a, b)
        if new != old:
            diff = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True)))[2:]
            diff = ['@@\n' if line.startswith('@@') else line for line in diff]
            patch = '*** Begin Patch\n*** Update File: '+str(p)+'\n'+''.join(diff)+'*** End Patch\n'
            subprocess.run(['apply_patch'], input=patch, text=True, check=True, capture_output=True)
            changed.append(str(p.relative_to(ROOT)))
    save(manifest, dict(status='COMPLETE_BYTES_VERIFIED_BEFORE_PATH_EDITS', mapping=mapping,
                        files=inventory, path_edits=changed, frozen_archives='preserved unchanged'))


if __name__ == '__main__':
    main()
