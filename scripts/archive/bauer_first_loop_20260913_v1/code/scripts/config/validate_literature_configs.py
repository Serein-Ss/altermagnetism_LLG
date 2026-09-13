"""Validate YAML structure and independently recompute reduced parameters."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from altermagnetism_LLG.scripts.core.literature_config import read_document, conversions, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT/'conf/literature')
    args=p.parse_args()
    rows=[]
    for path in sorted(args.directory.glob('*.yaml')):
        doc=read_document(path)
        rows.append({'paper_id':doc['paper_id'],'sha256':sha256(path),'recomputed':conversions(doc),'status':doc['provenance']['status']})
    if not rows: raise ValueError('no configurations found')
    print(json.dumps({'configuration_checks':'pass','papers':rows,'production_certified':False},indent=2))


if __name__=='__main__': main()
