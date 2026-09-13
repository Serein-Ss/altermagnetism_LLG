"""Regression checks for the engineered layout and artifact routing."""
import json
from pathlib import Path
import subprocess

from altermagnetism_LLG.scripts.core import project_paths

ROOT = Path(__file__).resolve().parents[2]


def test_separate_artifact_destinations(tmp_path, monkeypatch):
    monkeypatch.setattr(project_paths, "ROOT", tmp_path)
    output = tmp_path / "output/production/run/evaluation/L16"
    assert project_paths.asset_path(output / "paths.png") == tmp_path / "assets/research/models/production/run/evaluation/L16/paths.png"
    assert project_paths.generated_path(output / "paths.h5") == tmp_path / "data/generated/production/run/evaluation/L16/paths.h5"
    assert project_paths.asset_path(tmp_path / "data/literature/task/plot.png") == tmp_path / "assets/literature/task/plot.png"
    assert project_paths.asset_path(tmp_path / "data/audit/task/plot.png") == tmp_path / "assets/research/audit/task/plot.png"
    explicit = tmp_path.parent / "explicit.png"
    assert project_paths.asset_path(explicit) == explicit


def test_historical_path_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(project_paths, "ROOT", tmp_path)
    (tmp_path / "DIRECTORY_MIGRATION.json").write_text(json.dumps({"directory_rules": {"data/old": "data/datasets/new"}}))
    assert project_paths.resolve_recorded_path("data/old/a.h5") == tmp_path / "data/datasets/new/a.h5"


def test_lfs_manifest_destinations_and_sizes():
    manifest = json.loads((ROOT / "LFS_SPLIT_MANIFEST.json").read_text())
    for entry in manifest["files"]:
        assert entry["source"].startswith("data/")
        # Original reassembled files may be absent on a fresh LFS checkout.
        source = ROOT / entry["source"]
        if source.exists():
            assert source.stat().st_size == entry["source_bytes"]
        assert sum(part["bytes"] for part in entry["parts"]) == entry["source_bytes"]


def test_slurm_syntax_and_log_destination():
    for batch in (ROOT / "slurm").glob("*.sbatch"):
        subprocess.run(["bash", "-n", str(batch)], check=True)
        for line in batch.read_text().splitlines():
            if line.startswith("#SBATCH --output="):
                assert "altermagnetism_LLG/logs/" in line


def test_old_model_and_output_directories_removed():
    assert not (ROOT / "model").exists()
    assert not (ROOT / "outputs").exists()
    assert (ROOT / "scripts/model/network.py").exists()
    assert (ROOT / "scripts/training/train.py").exists()
    assert (ROOT / "scripts/visualization/model_plots.py").exists()
