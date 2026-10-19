import numpy as np
import torch
from .model import ARMS, make_model
from .run import tasks, config_for, train_step, next_batch, mixture_samples, mixture_comparison, dump
from tests.bauer_comprehensive_20260923.components import synthetic_parents, synthetic_paths


def test_paired_training_streams_and_losses():
    torch.set_num_threads(1);data=synthetic_parents('train');streams=[];counts=[]
    for arm in ARMS:
        task=dict(domain='mixture',arm=arm,seed=11);config=config_for(task)
        torch.manual_seed(11);net=make_model(config,arm);opt=torch.optim.AdamW(net.parameters(),lr=.001)
        rng=np.random.default_rng(11);g=torch.Generator().manual_seed(1010);target=torch.Generator().manual_seed(923641)
        y,t=next_batch(task,config,data,rng,target,'cpu')
        info,draws,tau=train_step(net,opt,y,t,g,arm)
        assert np.isfinite(info['loss']) and np.isfinite(info['gradient_norm'])
        if arm!='baseline':
            assert info['residual_gradient_norms']['residual_gate.weight']>1e-6
            counts.append(sum(p.numel() for p in net.parameters()))
        streams.append((y,draws,tau))
    assert len(set(counts))==1
    for stream in streams[1:]:
        for a,b in zip(streams[0],stream):torch.testing.assert_close(a,b,atol=0,rtol=0)
    rows=tasks();assert len(rows)==36
    for domain in ('bauer','mixture'):
        for arm in ARMS:
            assert {r['seed'] for r in rows if r['domain']==domain and r['arm']==arm}=={11,22,33}


def test_mixture_metrics_discriminate_incoherent_site_modes():
    torch.set_num_threads(1)
    initial=synthetic_parents('train')[:1].expand(256,-1,-1)
    reference=synthetic_paths(initial,'mixture',torch.Generator().manual_seed(6)).numpy()
    broken=reference.copy()
    # Break branch-level coherence while preserving each site's exact sample set.
    for site in range(reference.shape[2]):broken[:,:,site]=np.roll(reference[:,:,site],site*13,axis=0)
    a,b=mixture_samples(reference),mixture_samples(broken)
    assert a['direction_coherence'].mean()>.9 and b['direction_coherence'].mean()<.5
    assert a['rotation_amplitude'].mean()>b['rotation_amplitude'].mean()+.1
    result=mixture_comparison(reference,reference,repeats=5)
    assert all(row['difference']==0 for row in result['0.1'].values())
    assert len(result)==3


def test_report_and_png_rendering(tmp_path,monkeypatch):
    from . import report
    from tests.bauer_comprehensive_20260923.components import diagnostic_arrays
    from .run import mixture_comparison
    from tests.bauer_p2_20260923.common import comparison
    project=tmp_path/'project';here=project/'tests/repair';here.mkdir(parents=True)
    (here/'assets').mkdir();(project/'tests/index.html').write_text('<main>existing content</main>')
    monkeypatch.setattr(report,'HERE',here);monkeypatch.setattr(report,'ROOT',project)
    dump(here/'protocol.json',dict(tasks=tasks(),controls={'local_fixed':'active residual'}))
    report.report()
    page=(here/'assets/index.html').read_text()
    assert '旧对照勘误' in page and '0/36' in page
    assert 'existing content' in (project/'tests/index.html').read_text()
    # Render representative schema fixtures, not pretend training results.
    initial=synthetic_parents('dev')[:1].expand(16,-1,-1)
    paths=synthetic_paths(initial,'mixture',torch.Generator().manual_seed(2)).numpy()
    rows=[]
    for index,domain in enumerate(('bauer','mixture')):
        task=dict(domain=domain,arm='baseline',seed=11)
        directory=report.directory(index,task);(directory/'generated_dev_128').mkdir(parents=True)
        np.savez_compressed(directory/'generated_dev_128/example.npz',generated=paths,reference=paths)
        d=diagnostic_arrays(paths)
        dump(directory/'dev_final128.json',dict(mean_parent_RMSE={k:v['rmse'] for k,v in comparison(paths,paths).items()},parents=[dict(diagnostic={k:dict(generated=v.tolist(),reference=v.tolist()) for k,v in d.items()},mixture=mixture_comparison(paths,paths,0) if domain=='mixture' else None)]))
        dump(directory/'complete.json',dict(step=12000,seconds=2.,sample_stream_sha256='fixture'))
        dump(directory/'history.json',[dict(residual_rms=0.)])
        (directory/'generated_dev_256').mkdir()
        np.savez_compressed(directory/'generated_dev_256/example.npz',generated=paths,reference=paths)
        rows.append(dict(index=index,task=task))
    files=report.figures(rows)
    assert len(files)==2 and all(p.suffix=='.png' and p.stat().st_size>10000 for p in files)

    dump(here/'protocol.json',dict(tasks=[r['task'] for r in rows],controls={'baseline':'test fixture only'}))
    report.report()
    page=(here/'assets/index.html').read_text()
    assert '2/2' in page and 'data:image/png;base64,' in page
    assert (project/'tests/index.html').read_text().count('<!-- repair-20260924-start -->')==1
