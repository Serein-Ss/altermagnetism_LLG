import json
import sys

import h5py

from scripts.validation import validate_nishino_full, audit_nishino_ensemble


def test_small_reference_run_writes_report_and_is_not_certified(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,"argv",["validate_nishino_full","--task","0","--output-dir",str(tmp_path),
        "--coarse-steps","8","--moments","4","--save-every","1"])
    validate_nishino_full.main()
    report=json.loads((tmp_path/"run_000.json").read_text())
    assert not report["formal_settings"]
    assert len(report["rows"])==6
    with h5py.File(tmp_path/"run_000.h5","r") as h5:
        assert h5.attrs["complete"]
        assert h5["heun/dt_4/cos_theta"].shape==(4,4)


def test_missing_runs_fail_ensemble_gate_and_leave_report(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,"argv",["audit_nishino_ensemble","--input-dir",str(tmp_path),
        "--output",str(tmp_path/"decision.json")])
    import pytest
    with pytest.raises(SystemExit) as error:
        audit_nishino_ensemble.main()
    assert error.value.code==2
    report=json.loads((tmp_path/"decision.json").read_text())
    assert not report["passed"]
    assert not report["production_enabled"]
