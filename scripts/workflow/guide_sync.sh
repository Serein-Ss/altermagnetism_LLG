#!/usr/bin/env bash
# Local GUIDE is download-only. No commit or push is performed here.
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-}" in
  stage)
    if test -n "$(git diff --cached --name-only -- GUIDE data)"; then
      echo 'GUIDE or data already staged; unstage them before proceeding.' >&2; exit 1
    fi
    # data is ignored; restore tracked GUIDE in the index after staging.
    # The guard above ensures this cannot discard pre-existing staged GUIDE edits.
    git add -A
    git restore --staged -- GUIDE
    ;;
  download)
    git fetch origin main
    backup=$(mktemp -d .git/guide-backup.XXXXXX)
    cp -a GUIDE "$backup/"
    git restore --source=origin/main --worktree -- GUIDE
    echo "Previous GUIDE retained in $backup"
    ;;
  *) echo 'Usage: bash scripts/workflow/guide_sync.sh stage|download' >&2; exit 2 ;;
esac
