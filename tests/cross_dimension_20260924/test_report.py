import json
import sys
import h5py
import numpy as np
from . import report
from .oracle import make_net


def test_uniform_source_metadata():
    net=make_net('fm','cpu')
    assert net.get_extra_state()['source']=='independent_uniform_spheres'
    assert net.get_extra_state()['source_shift']==0


def test_end_to_end_report_gif_and_main_page(tmp_path,monkeypatch):
    root=tmp_path/'project';here=root/'tests/campaign';out=here/'runs/00';out.mkdir(parents=True)
    (root/'tests/index.html').write_text('<main><p>preserved old report</p></main>')
    task=dict(domain='miltat',arm='global',seed=11,updates=8)
    (out/'complete.json').write_text(json.dumps(dict(task=task,total_seconds=1.)))
    rng=np.random.default_rng(52);x=rng.normal(size=(8,11,6,5,2,3));x/=np.linalg.norm(x,axis=-1,keepdims=True)
    x[:,0]=x[0,0]
    cell,_,_=report.build(report.physical_config(),'miltat',shape=(6,5))
    import torch
    pos=cell.positions(torch.tensor(x[0])).numpy()*.251
    with h5py.File(out/'paths.h5','w') as h:
        h.attrs['complete']=True;g=h.create_group('parent_000_steps128')
        g.attrs['split']='dev';g.attrs['scope']='same_size'
        for k,v in [('generated',x),('reference',x),('positions_nm',pos),('time_ps',np.arange(11)*.1)]:g.create_dataset(k,data=v)
    monkeypatch.setattr(report,'HERE',here);monkeypatch.setattr(report,'ROOT',root)
    monkeypatch.setattr(report,'tasks',lambda:[task]);monkeypatch.setattr(sys,'argv',['report'])
    report.main()
    text=(here/'assets/index.html').read_text()
    assert 'data:image/png;base64,' in text and 'data:image/gif;base64,' in text
    assert 'preserved old report' in (root/'tests/index.html').read_text()
    assert (here/'assets/run00.gif').read_bytes()[:3]==b'GIF'
    result=json.loads((here/'assets/summary.json').read_text())
    assert not result['missing'] and len(result['results'])==1
