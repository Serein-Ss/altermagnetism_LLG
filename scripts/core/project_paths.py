"""Project artifact destinations; historical payloads are never rewritten."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("LLG_ARTIFACT_ROOT", Path(__file__).resolve().parents[2]))


def asset_path(path):
    """Mirror project data/output plots into assets, retaining task hierarchy.

    Explicit paths outside the project (e.g. test temporary directories) and
    paths already in assets are left unchanged.
    """
    path = Path(path)
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return path
    parts = relative.parts
    if parts[0] == "output":
        path = ROOT / "assets/research/models" / Path(*parts[1:])
    elif parts[0] == "data":
        if parts[1] == "audit" and path.name.startswith("nishino"):
            path = ROOT / "assets/literature" / Path(*parts[1:])
        elif parts[1] == "literature":
            path = ROOT / "assets/literature" / Path(*parts[2:])
        elif parts[1] == "research":
            path = ROOT / "assets/research" / Path(*parts[2:])
        else:
            path = ROOT / "assets/research" / Path(*parts[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def generated_path(path):
    """Keep generated trajectory arrays out of model/checkpoint directories."""
    path = Path(path)
    try:
        relative = path.resolve().relative_to((ROOT / "output").resolve())
    except ValueError:
        return path
    path = ROOT / "data/generated" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_recorded_path(path):
    """Resolve old report/HDF5 paths through the explicit migration rules."""
    import json
    path = Path(path)
    if path.exists():
        return path
    value = str(path)
    marker = ROOT.name + "/"
    if marker in value:
        value = value.split(marker, 1)[1]
    manifest = ROOT / "DIRECTORY_MIGRATION.json"
    if manifest.exists():
        rules = json.loads(manifest.read_text())["directory_rules"]
        for old in sorted(rules, key=len, reverse=True):
            if value == old or value.startswith(old + "/"):
                return ROOT / (rules[old] + value[len(old):])
    return ROOT / value if not path.is_absolute() else path
