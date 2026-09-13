"""GUIDE §6.3 exact-target cleanup, only after tested replacement is held in Slurm."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from altermagnetism_LLG.scripts.core.literature_config import sha256

TARGETS=[
    'data/audit/20260909_3b3b824_guide_thermal/nishino',
    'data/audit/software_check_20260909/nishino',
    'assets/literature/audit/20260909_3b3b824_guide_thermal/nishino_decision.png']


def inventory():
    files=[]
    for relative in TARGETS:
        path=ROOT/relative
        if path.resolve()!=path or not path.exists(): raise ValueError('invalid exact target '+relative)
        contents=sorted(path.rglob('*')) if path.is_dir() else [path]
        if any(p.is_symlink() for p in contents): raise ValueError('symlinks are prohibited')
        files.extend(p for p in contents if p.is_file())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    records=[]
    for path in files:
        relative=str(path.relative_to(ROOT));digest=sha256(path)
        committed=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=ROOT)
        if committed.startswith(b'version https://git-lfs.github.com/spec/v1'):
            expected=re.search(rb'oid sha256:([0-9a-f]{64})',committed).group(1).decode()
        else: expected=hashlib.sha256(committed).hexdigest()
        if digest!=expected: raise ValueError('uncommitted target data: cannot certify recovery '+relative)
        records.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest))
    return dict(targets=TARGETS,files=records,bytes=sum(r['bytes'] for r in records),
        recovery_commit=commit,recovery='All payloads match HEAD blobs or LFS OIDs; restore from Git and existing LFS storage.',
        reason='GUIDE 6.3 obsolete Nishino implementation; not deleting other research data')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--held-job-id',required=True)
    p.add_argument('--apply',action='store_true');a=p.parse_args()
    if not a.held_job_id.isdigit(): raise ValueError('numeric job ID required')
    preflight=json.loads((a.campaign/'preflight.json').read_text())
    if preflight['status']!='pass': raise ValueError('preflight not passed')
    job=subprocess.check_output(['scontrol','show','job',a.held_job_id,'-o'],text=True)
    if not all(text in job for text in ['JobName=zrs-mag','JobState=PENDING','Reason=JobHeldUser',
        'reduced_literature_r0.sbatch',str(a.campaign)]):
        raise ValueError('replacement job is not the expected held R0 campaign')
    result=inventory();result.update(replacement_job_id=a.held_job_id,campaign=str(a.campaign))
    output=ROOT/'output/literature_reproduction/nishino_miyashita_2015'/a.campaign.name
    output.mkdir(parents=True,exist_ok=True);manifest=output/'obsolete_deletion_manifest.json'
    if a.apply:
        previous=json.loads(manifest.read_text())
        if previous!=result: raise ValueError('inventory changed since dry run')
        for target in TARGETS:
            path=ROOT/target
            if path.is_dir(): shutil.rmtree(path)
            else: path.unlink()
        (output/'obsolete_deletion_completed.json').write_text(json.dumps(result,indent=2))
    else:
        with manifest.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(applied=a.apply,files=len(result['files']),bytes=result['bytes'],
        manifest=str(manifest),recovery_commit=result['recovery_commit'])))


if __name__=='__main__': main()
