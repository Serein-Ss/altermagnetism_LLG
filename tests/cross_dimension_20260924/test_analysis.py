import numpy as np
from tests.literature_path_campaign_20260920.train_miltat import physical_config
from tests.literature_preflight_20260917.models import build
from .analysis import observables,compare,increment_variance
from .report import plot_series


def test_metrics_identical_ensembles_and_png(tmp_path):
    rng=np.random.default_rng(4);x=rng.normal(size=(8,11,6,5,2,3))
    x/=np.linalg.norm(x,axis=-1,keepdims=True);x[:,0]=x[0,0]
    cell,_,_=build(physical_config(),'miltat',shape=(6,5))
    a=observables(x,cell);result=compare(a,a)
    assert all(v==0 for k,v in result['metrics'].items() if k.endswith('rmse'))
    assert result['metrics']['first_increment_variance_ratio']==1
    centered=a['increment']-a['increment'].mean(0)
    trace=np.trace(np.einsum('btnc,btnd->tncd',centered,centered)/7,axis1=-2,axis2=-1).mean(-1)
    assert np.allclose(increment_variance(a['increment']),trace)
    plot_series(result,tmp_path/'plot.png')
    assert (tmp_path/'plot.png').read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
