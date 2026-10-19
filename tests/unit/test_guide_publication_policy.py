"""Exercise publication helpers in isolated repositories, not the user's index."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]


def git(repo,*args):
    return subprocess.run(['git',*args],cwd=repo,text=True,capture_output=True,check=True)


@pytest.fixture
def repo(tmp_path):
    # Keep temporary nested Git repositories out of the parent project's index.
    (tmp_path/'.gitignore').write_text('*\n')
    p=tmp_path/'repo';p.mkdir();git(p,'init')
    for folder in ['GUIDE','data','tests/data','scripts/workflow']:(p/folder).mkdir(parents=True,exist_ok=True)
    (p/'GUIDE/plan.md').write_text('old\n')
    shutil.copy(ROOT/'.gitignore',p/'.gitignore')
    shutil.copy(ROOT/'scripts/workflow/guide_sync.sh',p/'scripts/workflow/guide_sync.sh')
    shutil.copy(ROOT/'.git/hooks/pre-commit',p/'.git/hooks/pre-commit')
    git(p,'add','.')
    git(p,'-c','user.name=Policy Test','-c','user.email=policy-test@example.invalid','commit','-m','fixture')
    return p


def test_guide_stage_keeps_data_local(repo):
    (repo/'GUIDE/plan.md').write_text('new\n');(repo/'GUIDE/new.md').write_text('new plan\n')
    (repo/'data/raw.bin').write_bytes(b'raw');(repo/'tests/data/raw.bin').write_bytes(b'raw')
    subprocess.run(['bash','scripts/workflow/guide_sync.sh','stage'],cwd=repo,check=True,capture_output=True)
    changed=git(repo,'diff','--cached','--name-only').stdout.splitlines()
    assert changed==['GUIDE/new.md','GUIDE/plan.md']
    assert not git(repo,'ls-files','--','data','tests/data').stdout
    subprocess.run(['bash','.git/hooks/pre-commit'],cwd=repo,check=True,capture_output=True)


@pytest.mark.parametrize('path',['data/raw.bin','tests/data/raw.bin'])
def test_forced_data_is_blocked_without_deleting_it(repo,path):
    (repo/path).write_bytes(b'raw');git(repo,'add','-f',path)
    for cmd in [['bash','.git/hooks/pre-commit'],['bash','scripts/workflow/guide_sync.sh','stage']]:
        r=subprocess.run(cmd,cwd=repo,capture_output=True,text=True)
        assert r.returncode!=0
    assert (repo/path).read_bytes()==b'raw'


def test_download_refuses_local_changes_before_network(repo):
    (repo/'GUIDE/plan.md').write_text('keep my plan\n')
    r=subprocess.run(['bash','scripts/workflow/guide_sync.sh','download'],cwd=repo,capture_output=True,text=True)
    assert r.returncode!=0 and 'local changes' in r.stderr
    assert (repo/'GUIDE/plan.md').read_text()=='keep my plan\n'
