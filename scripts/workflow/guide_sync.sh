#!/usr/bin/env bash
# GUIDE is versioned; raw datasets remain local. No commit or push here.
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-}" in
  stage)
    if test -n "$(git ls-files -- data tests/data)"; then
      echo 'data/ or tests/data/ is tracked/staged. Remove it from the index without deleting local data before staging.' >&2; exit 1
    fi
    # Respect .gitignore; keep existing staged GUIDE edits and include new plans.
    git add -A
    if test -n "$(git ls-files -- data tests/data)"; then
      echo 'Dataset exclusion failed; inspect the index before committing.' >&2; exit 1
    fi
    ;;
  download)
    if ! git diff --quiet HEAD -- GUIDE || test -n "$(git ls-files --others --exclude-standard -- GUIDE)"; then
      echo 'GUIDE has local changes. Commit or preserve them before downloading; no overwrite performed.' >&2; exit 1
    fi
    git fetch origin main
    guide_backup=$(mktemp -d "$(git rev-parse --git-dir)/guide-backup.XXXXXX")
    cp -a GUIDE "$guide_backup/"
    # Compatible with the server's Git 1.8; leave the index unchanged.
    git archive origin/main GUIDE | tar -x
    echo "Previous GUIDE retained in $guide_backup; inspect changes before staging."
    ;;
  *) echo 'Usage: bash scripts/workflow/guide_sync.sh stage|download' >&2; exit 2 ;;
esac
