"""Synthetic fixture only; no scientific performance claim."""
import json
import h5py
import numpy as np
from . import report


def test_dashboard_analysis_and_png_embedding(tmp_path,monkeypatch):
    here=tmp_path/'project/tests/repair';out=here/'run/00';out.mkdir(parents=True)
    root=tmp_path/'project';(root/'tests/index.html').write_text('<html><body>preserve-old</body></html>')
    task=dict(domain='toy',arm='hierarchical',repair='baseline',seed=11,updates=32,modes=2)
    info=dict(task=task,train_seconds=1.,device='fixture',early_tau_fraction=.1,evaluation=[
        dict(parent=0,steps=128,positive_fraction=.5,positive_wilson95=[.3,.7],direction_coherence=1.)])
    (out/'complete.json').write_text(json.dumps(info))
    rng=np.random.default_rng(12);x=rng.normal(size=(16,11,16,3));x/=np.linalg.norm(x,axis=-1,keepdims=True)
    x[:,0]=x[0,0]
    with h5py.File(out/'paths.h5','w') as h:
        h.attrs['complete']=True;g=h.create_group('parent0_steps128')
        g.create_dataset('generated',data=x);g.create_dataset('reference',data=x)
    monkeypatch.setattr(report,'HERE',here);monkeypatch.setattr(report,'ROOT',root)
    monkeypatch.setattr(report,'tasks',lambda:[task]);report.main()
    data=json.loads((here/'assets/summary.json').read_text())
    assert not data['missing'] and not data['errors'] and len(data['results'])==1
    assert 'data:image/png;base64,' in (here/'assets/index.html').read_text()
    assert 'preserve-old' in (root/'tests/index.html').read_text()
