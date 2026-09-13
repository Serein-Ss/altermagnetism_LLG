"""Read-only scientific inventory. Run hashing on fat, not on the login node."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.core.literature_config import sha256


def command(args,root):
    p=subprocess.run(args,cwd=root,text=True,capture_output=True)
    return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)


def inventory(campaign):
    m=json.loads((Path(campaign)/'manifest.json').read_text());root=Path(m['root']);out=Path(m['output'])
    statuses={
        'nishino_miyashita_2015':('accepted_with_deviation','Primary 24/24 pass; old 207/216 convergence and nine accepted stationarity deviations retained. New independent stationarity pending.'),
        'bauer_2011':('inconclusive','288 short chains ended at reduced time 200; zero saved crossings. Original duration and 500 events not certified.'),
        'hirst_mn2au_2022':('inconclusive','Wall still evolving, old width +30.48%; external per-bond audit, temperature/AFMR calibration pending.'),
        'gomonay_2024':('inconclusive','Candidate discrete S1 implemented; original moment-count geometry, external curves and critical velocity scan unresolved.'),
        'laliena_crnb3s6_2020':('fail','Candidate corrected BVP fold 1.208970 versus 1.2405; numerical axis separation and external curve audit pending.')}
    papers={}
    for paper,(status,note) in statuses.items():
        paths=[]
        for directory in (root/'scripts/literature'/paper,root/'data/literature_reproduction'/paper,
                          root/'output/literature_reproduction'/paper):
            paths.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        paths.append(root/'conf/literature'/f'{paper}.yaml')
        papers[paper]=dict(status=status,note=note,historical_state_updated_by_new_data=False,
            files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in paths},
            next_tasks=[t['index'] for t in m['tasks'] if t.get('paper')==paper])
    disk=shutil.disk_usage(root)
    record=dict(program_id=Path(campaign).name,root=str(root),commit=m['commit'],
        worktree=command(['git','status','--short'],root),python=sys.executable,python_version=sys.version,
        host=platform.node(),cpu=command(['lscpu'],root),memory=command(['free','-h'],root),
        queues=command(['sinfo','-o','%P %a %l %D %G'],root),quota=command(['quota','-s'],root),
        disk=dict(total=disk.total,free=disk.free),project_quota='not_exposed_by_user_quota_command',
        versions=command([sys.executable,'-m','pip','list','--format=json'],root),papers=papers,
        old_scheduler=command(['sacct','-j','669029,669030,669031,669032,669101,669112','-n','-P',
                               '-o','JobID,JobName,State,ExitCode,Elapsed'],root),
        preserved_analysis_failure='669101 failed during finalization; reports were recovered separately, exit code unchanged',
        acceptance_yaml=m['acceptance'],production_enabled=False)
    (out/'inventory.json').write_text(json.dumps(record,indent=2))
    text='# 无场研究计划事实盘点\n\n_2026-09-11；代码、数据和调度状态分开记录_\n\n---\n\n## 📋 状态\n\n'
    text+='| paper_id | 科学状态 | 下一批任务索引 |\n|---|---|---|\n'
    for paper,row in papers.items(): text+=f'| {paper} | {row["status"]} | {row["next_tasks"]} |\n'
    text+='\n## 🔍 证据与边界\n\n'
    for paper,row in papers.items(): text+=f'- `{paper}`: {row["note"]}\n'
    text+='\n完整逐文件 SHA-256、Python 环境、队列、配额查询及历史退出码见 [inventory.json](inventory.json)。\n'
    text+='本记录不把任务成功退出当作科学通过；旧失败、用户接受记录和原始数据均保留。\n'
    (out/'inventory.md').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);a=p.parse_args();inventory(a.campaign)
