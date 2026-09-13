"""One-off, recoverable cleanup of pre-Bauer visualization clutter (2026-09-13)."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def inventory(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    candidates = {}
    for name in ('audit', 'audit_data', 'unbiased_smoke', 'multimodality_validation',
                 'training_benchmark', 'size_convergence', 'standard_v2',
                 'models/diagnostics', 'models/production', 'runs/diagnostics',
                 'runs/production', 'runs/smoke'):
        candidates[ROOT / 'assets/research' / name] = 'Historical pre-Bauer research outputs; archived, not declared scientifically invalid'
    literature = ROOT / 'assets/literature_reproduction'
    candidates[literature / 'cross_paper_legacy'] = 'Legacy cross-paper outputs'
    for paper in ('bauer_2011', 'gomonay_2024', 'hirst_mn2au_2022',
                  'laliena_crnb3s6_2020', 'nishino_miyashita_2015'):
        for parent in (literature / paper, literature / paper / 'runs'):
            if not parent.is_dir():
                continue
            for p in parent.iterdir():
                if p.is_dir() and (p.name == 'legacy_before_reduced_20260909'
                        or p.name.startswith('implementation_')
                        or p.name.startswith('20260910_110834_validation_')
                        or p.name.startswith('r2_r5_validation_20260910_110834_cuda_smoke')
                        or p.name.startswith('smoke_20260910')):
                    candidates[p] = 'Early legacy/smoke/per-case validation outputs; aggregate analyses retained'
    candidates = {p: reason for p, reason in candidates.items() if p.is_dir()}
    print(json.dumps({'directories': len(candidates), 'bytes': sum(
        f.stat().st_size for p in candidates for f in p.rglob('*') if f.is_file()),
        'targets': [str(p.relative_to(ROOT)) for p in sorted(candidates)]}, indent=2))
    if not args.apply:
        return
    backup = Path(tempfile.mkdtemp(prefix='assets_history_backup_20260913_', dir=ROOT.parent))
    record = {'backup': str(backup), 'status': 'in_progress', 'moved': []}
    log = ROOT / 'logs/restructure/assets_cleanup_20260913.json'
    log.write_text(json.dumps(record, indent=2) + '\n')
    for source, reason in sorted(candidates.items()):
        assert source.resolve().is_relative_to((ROOT / 'assets').resolve())
        assert not source.is_symlink()
        files = inventory(source)
        destination = backup / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        source.rename(destination)
        record['moved'].append({'old': str(source.relative_to(ROOT)),
                              'new': str(destination), 'reason': reason, 'sha256': files})
        log.write_text(json.dumps(record, indent=2) + '\n')
        assert inventory(destination) == files, destination
    record['status'] = 'complete_verified'
    log.write_text(json.dumps(record, indent=2) + '\n')
    print('Verified recoverable archive:', backup)


if __name__ == '__main__':
    main()
