"""Snapshot and submit the prerequisite GUIDE thermal/integrator audit.

No polling, automatic resubmission, production generation or model training.
The interacting-system protocols depend on these results and remain unsubmitted.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    status = subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True)
    snapshot = ROOT / "scripts" / "archive" / output.name / "code"
    log_dir = ROOT / "logs" / "audit" / output.name
    if snapshot.exists() or log_dir.exists():
        raise FileExistsError("snapshot/log task name already exists")
    output.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    code = snapshot / "altermagnetism_LLG"
    code.mkdir(parents=True)
    for name in ("scripts","GUIDE"):
        shutil.copytree(ROOT/name,code/name,ignore=shutil.ignore_patterns("__pycache__","*.pyc",".pytest_cache","archive"))
    for name in ("registry.yaml","altermagnet_gomonay2024/protocol.candidate.yaml"):
        target=code/"data" / "datasets" / "standard_v3"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"data" / "datasets" / "standard_v3"/name,target)
    shutil.copy2(ROOT.parent/"run_zrs_mag.sh",snapshot/"run_zrs_mag.sh")
    batch=ROOT/"slurm"/"archive"/output.name/"guide_thermal_audit.sbatch"
    batch.parent.mkdir(parents=True)
    shutil.copy2(ROOT/"slurm"/batch.name,batch)
    manifest={"created":datetime.now().astimezone().isoformat(),"base_commit":commit,
              "source_worktree_status":status,"production_enabled":False,
              "scope":"GUIDE prerequisite thermal noise and free-moment integrator certification",
              "snapshot":str(snapshot),"log_dir":str(log_dir),"files":{},"jobs":{},"remaining":{
                  "interacting_dt_and_geometric_comparison":"requires thermal gate",
                  "certified_initial_pools":"candidate extraction added; requires interacting equilibrium and basin certificates",
                  "tn_scan_and_binder":"pilot scripts added; refined-grid bootstrap TN certification remains",
                  "zero_field_reversal_and_survival":"not implemented; requires TN/pool/threshold freeze",
                  "bauer_full_lifetime_campaign":"requires runtime and integrator certification",
                  "mn2au":"blocked_parameter_and_implementation_audit",
                  "production_datasets":"not submitted; prerequisites unverified"}}
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["files"][str(batch.relative_to(ROOT))]=hashlib.sha256(batch.read_bytes()).hexdigest()
    path=output/"submission.json"
    def save():
        path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    save()
    def submit(stage,dependency=None,array=None):
        cmd=["sbatch","--parsable","--output",str(log_dir/f"{stage}-%A_%a.out")]
        if dependency:
            cmd += ["--dependency",dependency,"--kill-on-invalid-dep=yes"]
        if array:
            cmd += ["--array",array]
        cmd += [str(batch),str(output),stage,str(snapshot),str(log_dir),str(ROOT)]
        value=subprocess.check_output(cmd,text=True).strip().split(";")[0]
        if not value.isdigit():
            raise RuntimeError(f"unexpected sbatch response: {value}")
        manifest["jobs"][stage]=value
        save()
        return value
    first_gate=submit("preflight")
    first=submit("first",f"afterok:{first_gate}")
    ensemble=submit("ensemble",f"afterok:{first}","1-6%1")
    submit("aggregate",f"afterany:{ensemble}")
    print(json.dumps({"output":str(output),"jobs":manifest["jobs"]},indent=2))


if __name__=="__main__":
    main()
