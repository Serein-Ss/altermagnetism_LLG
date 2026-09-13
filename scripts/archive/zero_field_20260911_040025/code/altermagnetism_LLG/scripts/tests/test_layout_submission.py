"""Exercise snapshot paths and sbatch arguments without submitting jobs."""
import json
import sys

from altermagnetism_LLG.scripts.workflow import submit_guide_thermal_audit as workflow


def test_submission_uses_categorized_snapshot_and_logs(tmp_path, monkeypatch):
    root = tmp_path / "altermagnetism_LLG"
    for name in ["scripts", "GUIDE", "slurm", "data/datasets/standard_v3/altermagnet_gomonay2024"]:
        (root / name).mkdir(parents=True)
    (root / "scripts/example.py").write_text("# snapshot example\n")
    (root / "GUIDE/plan.md").write_text("# plan\n")
    (root / "slurm/guide_thermal_audit.sbatch").write_text("#!/bin/bash\n")
    (root / "data/datasets/standard_v3/registry.yaml").write_text("{}")
    (root / "data/datasets/standard_v3/altermagnet_gomonay2024/protocol.candidate.yaml").write_text("{}")
    (tmp_path / "run_zrs_mag.sh").write_text("# fake runner; never executed\n")
    output = root / "data/audit/test_batch"
    monkeypatch.setattr(workflow, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["submit", "--output-root", str(output)])
    calls = []

    def fake_output(command, **kwargs):
        if command[0] == "git":
            return "test_commit\n" if command[1] == "rev-parse" else ""
        assert command[0] == "sbatch"
        calls.append(command)
        return str(1000 + len(calls)) + "\n"

    monkeypatch.setattr(workflow.subprocess, "check_output", fake_output)
    workflow.main()
    assert len(calls) == 4
    assert (root / "scripts/archive/test_batch/code/altermagnetism_LLG/scripts/example.py").exists()
    assert not (output / "code").exists()
    for command in calls:
        log = command[command.index("--output") + 1]
        assert str(root / "logs/audit/test_batch") in log
    record = json.loads((output / "submission.json").read_text())
    assert not record["production_enabled"]
    assert record["snapshot"] == str(root / "scripts/archive/test_batch/code")
