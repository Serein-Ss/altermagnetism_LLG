"""One-off literature asset housekeeping; preserve evidence and active outputs."""
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'assets/literature_reproduction'


def main():
    backup = Path(tempfile.mkdtemp(prefix='literature_assets_backup_20260913_', dir=ROOT.parent))
    entries = []
    log = ROOT / 'logs/restructure/literature_assets_cleanup_20260913.json'

    def move(source, destination, reason):
        assert source.is_file() and not source.is_symlink()
        assert not destination.exists(), destination
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        entries.append(dict(old=str(source.relative_to(ROOT)), new=str(destination),
                            reason=reason, sha256=sha))
        log.write_text(json.dumps(dict(backup=str(backup), files=entries), indent=2)+'\n')
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == sha

    metadata = {'manifest.json', 'submission.json', 'started.json', 'task_exit_codes.json',
                'preparation.json', 'obsolete_deletion_completed.json', 'obsolete_deletion_manifest.json'}
    for source in sorted(BASE.rglob('*')):
        if not source.is_file():
            continue
        relative = source.relative_to(BASE)
        # Root-level current run manifests and all current campaign records stay in place.
        if 'numba_cache' in relative.parts:
            move(source, backup / relative, 'Historical compiled cache, not a scientific result')
        elif 'runs' in relative.parts and not any(x in relative.parts for x in ('bauer_campaign', 'bauer_revision4')) and source.name in metadata:
            move(source, ROOT / 'logs/literature_reproduction/history_20260913' / relative,
                 'Execution bookkeeping belongs in logs; scientific reports retained')
    anomalous = BASE / 'remaining_submission_20260910.json/runs'
    if anomalous.is_file():
        move(anomalous, ROOT / 'logs/literature_reproduction/history_20260913/remaining_submission_20260910.json',
             'Correct misplaced submission record and inverted file/directory name')
    # Put reports beside their existing figures, without renaming experiments.
    for paper in ('bauer_2011', 'gomonay_2024', 'hirst_mn2au_2022', 'laliena_crnb3s6_2020'):
        source = BASE / paper / 'runs/r2_r5_analysis_20260910/report.json'
        if source.exists():
            move(source, BASE / paper / 'r2_r5_analysis_20260910/report.json', 'Co-locate report with its figures')
    paper = BASE / 'nishino_miyashita_2015'
    for run in ('20260909_8df33ac_bb48dc64', '20260910_8df33ac_d30ae957'):
        for source in sorted((paper / 'runs' / run).glob('*.json')):
            move(source, paper / run / source.name, 'Co-locate report/acceptance evidence with its figures')
    for directory in sorted(BASE.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    print(json.dumps(dict(moved_files=len(entries), backup=str(backup), manifest=str(log)), indent=2))


if __name__ == '__main__':
    main()
