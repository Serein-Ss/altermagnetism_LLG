"""Validate the immutable inputs and scientific gates for Standard V3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from unified_llg import CrNb3S6Helimagnet  # noqa: E402


REGISTRY = ROOT / "data" / "datasets" / "standard_v3" / "registry.yaml"
CANDIDATE = (
    ROOT
    / "data" / "datasets" / "standard_v3"
    / "altermagnet_gomonay2024"
    / "protocol.candidate.yaml"
)


def load_json_yaml(path: Path) -> dict:
    """Load the JSON-compatible subset of YAML used by frozen protocols."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one mapping")
    return value


def inspect_v3(
    registry_path: Path = REGISTRY,
    candidate_path: Path = CANDIDATE,
    frozen_protocol: Path | None = None,
) -> dict:
    registry = load_json_yaml(registry_path)
    candidate = load_json_yaml(candidate_path)
    systems = registry.get("systems", {})
    gomonay = systems.get("gomonay2024", {})
    mn2au = systems.get("mn2au2022", {})
    crnb3s6 = systems.get("crnb3s6_2020", {})
    scans = candidate.get("candidate_scans", {})
    unfrozen = candidate.get("unfrozen_production_fields", {})

    checks = {
        "registry_schema_v3": registry.get("schema_version") == 3,
        "registry_preproduction": registry.get("status") == "preproduction",
        "generator_cannot_self_certify": bool(
            registry.get("global_rules", {}).get(
                "production_certification_is_separate_from_generation"
            )
        ),
        "crnb3s6_doi_corrected": (
            crnb3s6.get("doi") == "10.1038/s41598-020-76903-8"
            and CrNb3S6Helimagnet().metadata().get("doi")
            == "10.1038/s41598-020-76903-8"
        ),
        "mn2au_explicitly_blocked": (
            mn2au.get("stage") == "blocked_parameter_audit"
            and mn2au.get("production_enabled") is False
        ),
        "gomonay_not_prematurely_enabled": (
            gomonay.get("production_enabled") is False
            and candidate.get("status") == "candidate_not_frozen"
            and candidate.get("production_enabled") is False
        ),
        "dt_scan_preregistered": scans.get("integration_dt_fs")
        == [0.1, 0.05, 0.025],
        "save_scan_preregistered": scans.get("full_spin_saved_dt_fs")
        == [2.5, 5.0, 10.0],
        "duration_scan_preregistered": scans.get("duration_ps")
        == [1.0, 2.0, 4.0, 8.0],
        "size_scan_and_holdouts_preregistered": scans.get("lattice_L")
        == [32, 48, 64, 96, 128],
        "production_fields_remain_unset": bool(unfrozen)
        and all(value is None for value in unfrozen.values()),
    }
    preparation_ready = all(checks.values())

    frozen = None
    production_checks = {
        "frozen_protocol_exists": frozen_protocol is not None
        and frozen_protocol.is_file(),
        "frozen_protocol_status": False,
        "frozen_protocol_enabled": False,
        "all_certification_inputs_present": False,
    }
    if production_checks["frozen_protocol_exists"]:
        frozen = load_json_yaml(frozen_protocol)
        production_checks["frozen_protocol_status"] = (
            frozen.get("status") == "frozen"
        )
        production_checks["frozen_protocol_enabled"] = (
            frozen.get("production_enabled") is True
        )
        required = frozen.get("certification_inputs", {})
        production_checks["all_certification_inputs_present"] = bool(required) and all(
            Path(value).is_file() for value in required.values()
        )

    production_ready = preparation_ready and all(production_checks.values())
    return {
        "status": (
            "production_ready"
            if production_ready
            else "preproduction_ready" if preparation_ready else "invalid"
        ),
        "registry": str(registry_path.resolve()),
        "candidate_protocol": str(candidate_path.resolve()),
        "frozen_protocol": (
            str(frozen_protocol.resolve()) if frozen_protocol else None
        ),
        "preparation_checks": checks,
        "production_checks": production_checks,
        "production_ready": production_ready,
        "action": (
            "production generation may be submitted"
            if production_ready
            else "run and certify the registered pilots before freezing protocol.yaml"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--frozen-protocol", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-production-ready", action="store_true")
    args = parser.parse_args()
    report = inspect_v3(args.registry, args.candidate, args.frozen_protocol)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] == "invalid":
        raise SystemExit(1)
    if args.require_production_ready and not report["production_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
