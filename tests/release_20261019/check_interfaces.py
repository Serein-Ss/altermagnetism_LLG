"""Short CPU CLI smoke; payloads stay under tests, never physical evidence."""
import json,subprocess,sys
from pathlib import Path
import h5py,numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


if __name__=='__main__':
    manifest=ROOT/'data/research/datasets/bauer_open_chain/conditional/real.json'
    row=json.loads(manifest.read_text())[0];reference=manifest.parent/row['file']
    with h5py.File(reference) as h:initial=h['spins'][0,0]
    np.save(HERE/'initial.npy',initial)
    subprocess.run([sys.executable,'-m','scripts.inference.bauer_path',
        '--checkpoint','output/development/bauer_path/step130240/seed11/checkpoint.pt',
        '--initial',str(HERE/'initial.npy'),'--branches','4','--duration-ps','.1',
        '--flow-steps','4','--device','cpu','--output',str(HERE/'generated.h5')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','scripts.analysis.evaluate_bauer_path',
        '--generated',str(HERE/'generated.h5'),'--reference',str(reference),
        '--output',str(HERE/'metrics.json'),'--bootstrap','4'],cwd=ROOT,check=True)
    report=json.loads((HERE/'metrics.json').read_text())
    assert report['branches']==[4,64] and not report['scientific_certification']
    print('CPU generation/evaluation CLI passed; TRAIN reference and4 transport steps are software smoke only')
