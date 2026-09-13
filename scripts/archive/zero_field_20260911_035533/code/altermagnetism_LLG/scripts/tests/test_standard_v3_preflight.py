import json
from pathlib import Path

from scripts.validation.preflight_standard_v3 import inspect_v3


def test_v3_candidate_is_valid_but_not_production_ready():
    report = inspect_v3()
    assert report["status"] == "preproduction_ready"
    assert not report["production_ready"]
    assert all(report["preparation_checks"].values())
    assert not report["production_checks"]["frozen_protocol_exists"]


def test_v3_frozen_protocol_requires_all_certification_files(tmp_path: Path):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        json.dumps(
            {
                "status": "frozen",
                "production_enabled": True,
                "certification_inputs": {
                    "equilibrium": str(tmp_path / "missing.json")
                },
            }
        ),
        encoding="utf-8",
    )
    report = inspect_v3(frozen_protocol=protocol)
    assert not report["production_ready"]
    assert not report["production_checks"][
        "all_certification_inputs_present"
    ]
