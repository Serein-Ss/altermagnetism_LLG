import h5py
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from .model import SpinPathTransformer
from .run import make_net, tasks, HERE


def test_shared_weights_and_time_contract():
    torch.set_num_threads(1)
    nets=[make_net(dict(domain='bauer',arm=a,seed=11),'cpu') for a in ('local','latent','hierarchical')]
    base=dict(nets[0].named_parameters())
    for net in nets[1:]:
        for name,p in net.named_parameters():
            if name in base:assert torch.equal(p,base[name])
    x=F.normalize(torch.randn(2,11,7,3),dim=-1)
    with pytest.raises(ValueError):nets[0].velocity(x,torch.ones(2)*.5,x[:,0],torch.arange(11)*.2)
    rows=tasks();assert len(rows)==36
    assert len({tuple(sorted(r.items())) for r in rows})==36


def test_bauer_report_path(tmp_path):
    from tests.bauer_comprehensive_20260923.run import references
    from .report import analyze
    _,raw=next(references('bauer','dev'))
    out=tmp_path/'run';out.mkdir();assets=tmp_path/'assets';assets.mkdir()
    with h5py.File(out/'paths.h5','w') as h:
        h.attrs['complete']=True;g=h.create_group('parent000_steps128')
        g.create_dataset('generated',data=raw);g.create_dataset('reference',data=raw)
    result=analyze(0,dict(domain='bauer'),out,assets)
    assert result['parents'][0]['anchor_error']<1e-6
    assert (assets/'run00.png').stat().st_size>1000


def test_toy_report_path(tmp_path):
    from .report import analyze
    from tests.cross_dimension_20260924.oracle import targets
    initial=F.normalize(torch.randn(16,16,3),dim=-1)
    raw=targets(initial,1)[:,0].numpy()
    # All independent futures must share the same actual parent.
    raw[:]=raw[:1]
    out=tmp_path/'run';out.mkdir();assets=tmp_path/'assets';assets.mkdir()
    with h5py.File(out/'paths.h5','w') as h:
        h.attrs['complete']=True;g=h.create_group('parent0_steps128')
        g.create_dataset('generated',data=raw);g.create_dataset('reference',data=raw)
    result=analyze(5,dict(domain='toy'),out,assets)
    assert (assets/'run05.png').exists()
    assert result['parents'][0]['mixture']['0.1']['direction_coherence']['difference']==0
