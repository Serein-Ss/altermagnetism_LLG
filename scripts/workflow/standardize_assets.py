"""Apply the audited assets layout, preserving bytes and live-task directories."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.core.asset_structure import destination, LIVE

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT/'logs/restructure/assets_standardization_20260913.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    rows = []
    for source in sorted((ROOT/'assets').rglob('*')):
        if not source.is_file():
            continue
        assert not source.is_symlink()
        old = str(source.relative_to(ROOT)); new = destination(old)
        if old != new:
            rows.append(dict(old=old, new=new, sha256=sha(source)))
    assert len({r['new'] for r in rows}) == len(rows), 'Destination collision'
    for row in rows:
        assert not (ROOT/row['new']).exists(), row['new']
    print(json.dumps(dict(files=len(rows), live_exceptions=LIVE), indent=2))
    if not args.apply:
        return
    assert not LOG.exists(), 'Migration already recorded'
    record = dict(status='in_progress', files=rows, live_exceptions=LIVE)
    LOG.write_text(json.dumps(record, indent=2)+'\n')
    for row in rows:
        target = ROOT/row['new']; target.parent.mkdir(parents=True, exist_ok=True)
        (ROOT/row['old']).rename(target)
        assert sha(target) == row['sha256']
    for directory in sorted((ROOT/'assets').rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    record['status'] = 'complete_verified'
    LOG.write_text(json.dumps(record, indent=2)+'\n')
    print('All moved file hashes verified; no payload rewritten.')


if __name__ == '__main__':
    main()
