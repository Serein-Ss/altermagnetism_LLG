"""One-time content-preserving migration; legacy data are not recertified."""
from pathlib import Path
import hashlib
import json
import re
ROOT=Path(__file__).resolve().parents[2]
PAPERS=['nishino_miyashita_2015','bauer_2011','hirst_mn2au_2022','gomonay_2024','laliena_crnb3s6_2020']
LEGACY='legacy_before_reduced_20260909'


def main():
    manifest=ROOT/'LITERATURE_DIRECTORY_MIGRATION.json'
    if manifest.exists(): raise FileExistsError(manifest)
    moves={}
    def add(old,paper,kind,filename=None):
        name=filename or Path(old).name
        base=f'data/literature_reproduction/{paper}'
        dest=f'{base}/reference/{name}' if kind=='reference' else f'{base}/{kind}/{LEGACY}/{name}'
        moves[old]=dest
    add('data/literature/literature_validation/Gomonay_2024_main.pdf','gomonay_2024','reference')
    add('data/literature/literature_validation/bauer_2011_magnetic_chain.pdf','bauer_2011','reference')
    add('data/literature/literature_validation/spinwave_trajectory.npz','gomonay_2024','raw')
    add('data/literature/literature_validation/spinwave_validation.json','gomonay_2024','derived')
    for p in sorted((ROOT/'data/literature/path_literature_validation').iterdir()):
        kind='raw' if p.suffix=='.h5' or p.name.endswith('.manifest.json') else 'derived'
        add(str(p.relative_to(ROOT)),'bauer_2011',kind)
    for p in sorted((ROOT/'data/literature/noncollinear_validation/crnb3s6').iterdir()):
        add(str(p.relative_to(ROOT)),'laliena_crnb3s6_2020','raw' if p.suffix=='.npz' else 'derived')
    for p in sorted((ROOT/'assets/literature/literature_validation').rglob('*')):
        if not p.is_file(): continue
        old=str(p.relative_to(ROOT))
        if 'reference' in p.parts: add(old,'gomonay_2024','reference')
        else:
            kind='animations' if p.suffix in ('.gif','.mp4') else 'figures'
            moves[old]=f'assets/literature_reproduction/gomonay_2024/{LEGACY}/{kind}/{p.name}'
    composite='assets/literature/next_stage_validation/literature_path_validation.png'
    moves[composite]=f'assets/literature_reproduction/cross_paper_legacy/{LEGACY}/figures/literature_path_validation.png'
    code={
        'scripts/literature/validate_spinwave.py':('gomonay_2024','legacy_run'),
        'scripts/literature/validate_bauer2011_chain.py':('bauer_2011','legacy_static'),
        'scripts/literature/generate_bauer2011_paths.py':('bauer_2011','legacy_run'),
        'scripts/literature/validate_crnb3s6_helix.py':('laliena_crnb3s6_2020','legacy_run'),
        'scripts/literature/validate_nishino_full.py':('nishino_miyashita_2015','legacy_run'),
        'scripts/analysis/audit_nishino_ensemble.py':('nishino_miyashita_2015','legacy_analyze'),
        'scripts/visualization/plot_literature_validation.py':('gomonay_2024','legacy_plot'),
        'scripts/visualization/compose_literature_comparison.py':('gomonay_2024','legacy_comparison'),
        'scripts/visualization/extract_reference_figures.py':('gomonay_2024','legacy_extract_reference'),
        'scripts/visualization/animate_spinwave_trajectory.py':('gomonay_2024','legacy_animation'),
    }
    for old,(paper,module) in code.items(): moves[old]=f'scripts/literature/{paper}/{module}.py'
    records=[]
    for old,new in moves.items():
        source=ROOT/old;dest=ROOT/new
        if not source.is_file() or source.is_symlink() or dest.exists(): raise ValueError((old,new))
        stat=source.stat()
        records.append(dict(old=old,new=new,bytes=stat.st_size,inode=stat.st_ino))
    for row in records:
        source=ROOT/row['old'];dest=ROOT/row['new']
        dest.parent.mkdir(parents=True,exist_ok=True);source.rename(dest)
        assert dest.stat().st_ino==row['inode'] and dest.stat().st_size==row['bytes']
    # Mechanical path-depth correction on moved code; preserve legacy equations.
    for old,(paper,module) in code.items():
        path=ROOT/moves[old];text=path.read_text()
        text=text.replace('.parents[3]','.parents[4]').replace('.parents[2]','.parents[3]')
        # Literal path expressions are updated only when a precise migrated file
        # is known. Historical data payloads and archived code are never edited.
        for before,after in sorted(moves.items(),key=lambda x:len(x[0]),reverse=True):
            text=text.replace(before,after)
            pattern=r'ROOT\s*'+''.join(r'/\s*["\x27]'+re.escape(part)+r'["\x27]\s*' for part in before.split('/'))
            text=re.sub(pattern,lambda m:'ROOT / '+repr(after)+' ',text)
        path.write_text(text)
        wrapper=ROOT/old
        wrapper.write_text('"""Deprecated import/CLI compatibility; implementation is explicitly legacy."""\n'
            'from pathlib import Path\nimport sys\n'
            'sys.path.insert(0,str(Path(__file__).resolve().parents[3]))\n'
            f'from altermagnetism_LLG.scripts.literature.{paper}.{module} import *\n'
            "if __name__=='__main__':\n"
            "    import warnings\n"
            "    warnings.warn('Legacy SI/pilot entry: not the strict reduced reproduction',RuntimeWarning)\n"
            "    main()\n")
    for paper in PAPERS:
        for kind in ['reference','raw','derived']:
            (ROOT/'data/literature_reproduction'/paper/kind).mkdir(parents=True,exist_ok=True)
        status={'paper_id':paper,'legacy_run_id':LEGACY,'legacy_status':'not_strictly_certified',
                'strict_status':'R0_in_preparation' if paper==PAPERS[0] else 'blocked_until_R0_and_paper_model_audit',
                'legacy_paths':[r for r in records if f'/{paper}/' in r['new']],
                'config':f'conf/literature/{paper}.yaml'}
        output=ROOT/'output/literature_reproduction'/paper/LEGACY
        output.mkdir(parents=True,exist_ok=True)
        (output/'manifest.json').write_text(json.dumps(status,indent=2))
    manifest.write_text(json.dumps({'note':'Data moved without content changes; legacy code depth/path edits only. Compatibility wrappers retained. Old Nishino deletion is a separate gated operation.',
        'files':records,'file_rules':moves},indent=2))
    print(json.dumps({'moved':len(records),'manifest':str(manifest)}))


if __name__=='__main__': main()
