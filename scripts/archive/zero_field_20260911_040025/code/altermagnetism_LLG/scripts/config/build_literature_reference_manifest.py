"""Inventory reference files without altering legacy source metadata."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from altermagnetism_LLG.scripts.core.literature_config import read_document,sha256


def main():
    for path in sorted((ROOT/'conf/literature').glob('*.yaml')):
        doc=read_document(path);paper=doc['paper_id']
        reference=ROOT/'data/literature_reproduction'/paper/'reference'
        reference.mkdir(parents=True,exist_ok=True)
        records={p.name:dict(sha256=sha256(p),bytes=p.stat().st_size)
                 for p in sorted(reference.iterdir()) if p.is_file() and p.name!='reference_manifest.json'}
        report=dict(paper_id=paper,provenance=doc['provenance'],files=records,
            audit_status='candidate_only; full paper-specific audit remains required')
        if paper=='nishino_miyashita_2015':
            report.update(audit_status='published Fig.1 and Appendix B inspected; 2018 erratum inspected',
                published_doi='10.1103/PhysRevB.91.134411',
                erratum_doi='10.1103/PhysRevB.97.019904',
                erratum_scope='Fokker-Planck Eqs.7,A11,A13 and definitions15,16 corrected; Fig.1 settings and AppendixB target unchanged',
                extraction=dict(file='fig1_published_page3.png',figure='1',pdf_page_1based=3,
                    printed_page='134411-3',method='pdftoppm direct raster crop; no redrawing/digitization',
                    command='pdftoppm -f 3 -l 3 -singlefile -r 100 -x 430 -y 45 -W 395 -H 370 -png PhysRevB.91.134411.pdf fig1_published_page3',
                    content='Published plot plus partial caption. Full caption is in source PDF.',
                    sha256=records['fig1_published_page3.png']['sha256']),
                numerical_grid_note='The YAML grid is a project validation grid, not extracted paper simulation points.')
        with (reference/'reference_manifest.json').open('x') as stream: json.dump(report,stream,indent=2)
    print('Reference inventories written; candidate papers are not recertified.')


if __name__=='__main__': main()
