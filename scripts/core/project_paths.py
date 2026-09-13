"""Project artifact destinations; historical payloads are never rewritten."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("LLG_ARTIFACT_ROOT", Path(__file__).resolve().parents[2]))


def document_path(path):
    """Keep newly generated Markdown reports under docs, including local runs."""
    path = Path(path)
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return path
    if relative.parts[0] not in ('docs', 'GUIDE') and relative.as_posix() != 'README.md':
        path = ROOT / 'docs/reports' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
        if parts[1] == 'literature_reproduction':
            path = ROOT / 'assets/literature_reproduction' / Path(*parts[2:])
        elif parts[1] == "audit" and path.name.startswith("nishino"):
            path = ROOT / "assets/literature_reproduction/legacy" / Path(*parts[1:])
        elif parts[1] == "literature":
            path = ROOT / "assets/literature_reproduction/legacy" / Path(*parts[2:])
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
    path = ROOT / "data/research/generated" / relative
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
    latest = ROOT / 'logs/restructure/migration_20260913.json'
    latest_rules = {row['old']:row['new'] for row in json.loads(latest.read_text())['files']} if latest.exists() else {}
    supplement = ROOT / 'logs/restructure/migration_supplement_20260913.json'
    extra = {row['old']:row['new'] for row in json.loads(supplement.read_text())['files']} if supplement.exists() else {}
    latest_rules = {old:extra.get(new,new) for old,new in latest_rules.items()}
    latest_rules.update(extra)
    if value in latest_rules:
        return ROOT / latest_rules[value]
    literature_manifest = ROOT / "logs/restructure/history/LITERATURE_DIRECTORY_MIGRATION.json"
    literature_rules = json.loads(literature_manifest.read_text())["file_rules"] if literature_manifest.exists() else {}
    if value in literature_rules:
        mapped = literature_rules[value]
        return ROOT / latest_rules.get(mapped, mapped)
    manifest = ROOT / "logs/restructure/history/DIRECTORY_MIGRATION.json"
    if manifest.exists():
        rules = json.loads(manifest.read_text())["directory_rules"]
        for old in sorted(rules, key=len, reverse=True):
            if value == old or value.startswith(old + "/"):
                mapped = rules[old] + value[len(old):]
                mapped = literature_rules.get(mapped, mapped)
                return ROOT / latest_rules.get(mapped, mapped)
    return ROOT / value if not path.is_absolute() else path
