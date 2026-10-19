"""Synthetic temporary fixture only; never writes invented scientific results."""
import json
import numpy as np
from . import report
from .common import cfg, comparison, dump


def test_completed_report_and_all_embedded_png(tmp_path, monkeypatch):
    import tests.progress_dashboard.build as dashboard
    monkeypatch.setattr(dashboard,'main',lambda:None)
    monkeypatch.setattr(report,'HERE',tmp_path)
    config=cfg(); rng=np.random.default_rng(8); parents=[]; manifest=[]
    run=tmp_path/'runs'/'seed_11'
    for n in config['sizes']:
        ref=rng.normal(size=(16,11,n,3)); ref/=np.linalg.norm(ref,axis=-1,keepdims=True)
        pred=ref.copy(); parent=f'L{n}_synthetic'
        parents.append(dict(parent_id=parent,L=n,family='synthetic',generation_seconds=.1,
                            metrics=comparison(pred,ref,repeats=4)))
        manifest.append(dict(L=n,parent_id=parent,family='synthetic'))
        dest=run/'generated'/'dev_128'; dest.mkdir(parents=True,exist_ok=True)
        np.savez(dest/f'{parent}.npz',generated=pred,reference=ref,time_ps=np.arange(11)*.1)
    metric={k:0. for k in parents[0]['metrics']}
    dump(tmp_path/'audit.json',dict(max_norm_error=0.,max_stored_energy_error=0.))
    dump(tmp_path/'numerical_reference.json',dict(comparisons=parents))
    dump(tmp_path/'manifest_dev.json',manifest)
    dump(run/'complete.json',dict(status='SYNTHETIC_TEST_ONLY'))
    dump(run/'selection.json',dict(selected_step=100,gpu='synthetic',train_seconds=1,criterion=[0,0]))
    dump(run/'history.json',[dict(loss=1.) for _ in range(101)])
    for n in config['flow_steps']:
        dump(run/f'dev_{n}.json',dict(parents=parents,mean_parent_RMSE=metric))
    dump(run/'diagnostic_128.json',dict(parents=parents,mean_parent_RMSE=metric))
    report.main()
    page=(tmp_path/'assets'/'index.html').read_text()
    assert page.count('data:image/png;base64,')==9
    assert 'Seed 22' in page and '尚无完整报告' in page
    assert not list((tmp_path/'assets').glob('*.svg'))
    assert json.loads((tmp_path/'assets'/'report.json').read_text())['status']=='INCOMPLETE'
