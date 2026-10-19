import csv
import numpy as np
import torch
from tests.paper_work.completion.run import subset_indices, energy_distance, paths, deterministic_step
from tests.paper_work.completion.long_rollout import advance
from tests.experiments.director_transition import DirectorTransition


def model():
    torch.manual_seed(19)
    m=DirectorTransition(.12,.01,8).eval()
    return m,dict(config=dict(lag=.01),latent_scale=.015)


def test_subset_whole_paths_and_nested():
    with open('tests/paper_work/tables/T01_transition_index.csv') as f:rows=list(csv.DictReader(f))
    subsets=[set(subset_indices(rows,f,241915)) for f in (.25,.5,1.)]
    assert list(map(len,subsets))==[3072,6144,12288]
    assert subsets[0]<subsets[1]<subsets[2]
    for subset in subsets:
        counts={}
        for r in rows:
            if r['split']=='train' and int(r['sample']) in subset:
                key=(r['source'],r['path_index']);counts[key]=counts.get(key,0)+1
        assert set(counts.values())=={8}


def test_distribution_distance():
    x=np.zeros((8,25,3));y=x.copy();y[...,2]=1
    assert abs(energy_distance(x,x))<1e-12
    assert energy_distance(x,y)>0
    assert np.isclose(energy_distance(x,y),energy_distance(y,x))


def test_deterministic_and_variable_size():
    m,ck=model()
    for length in (25,50,100):
        x=np.zeros((length,3));x[:,2]=1
        p=paths(m,ck,x,.11,.02,3,123,kind='deterministic',device='cpu')
        assert p.shape==(3,3,length,3)
        np.testing.assert_array_equal(p[0],p[1])
        assert np.max(abs(np.linalg.norm(p,axis=-1)-1))<1e-5


def test_exact_chunk_resume():
    m,ck=model();x=torch.zeros(2,5,3);x[...,2]=1
    rng=torch.Generator().manual_seed(19)
    end,all_frames=advance(m,ck,x,.11,rng,4,1)
    rng=torch.Generator().manual_seed(19)
    middle,first=advance(m,ck,x,.11,rng,2,1)
    saved=rng.get_state().clone();rng2=torch.Generator();rng2.set_state(saved)
    replay,last=advance(m,ck,middle.clone(),.11,rng2,2,1)
    torch.testing.assert_close(end,replay,atol=0,rtol=0)
    np.testing.assert_array_equal(all_frames,np.concatenate((first,last),axis=1))


def test_reference_torch_matches_cpu_given_draws():
    from scripts.core.bauer_fast import mt_step
    from scripts.core.reduced_llg import BondHamiltonian
    from scripts.literature.bauer_2011.integrator import BauerWeakRK
    rng=np.random.default_rng(22);x=rng.normal(size=(5,3));x/=np.linalg.norm(x,axis=-1,keepdims=True)
    xi=rng.choice([-np.sqrt(3.),0.,np.sqrt(3.)],size=x.shape);zeta=rng.choice([-1.,1.],size=x.shape)
    y,_=mt_step(x,xi,zeta,.00125,.1,.1,.11)
    solver=BauerWeakRK(BondHamiltonian([(i,i+1) for i in range(4)],1.,anisotropy=.1),alpha=.1,theta=.11)
    actual,_=solver.step(torch.tensor(x)[None],.00125,torch.tensor(np.stack((xi,zeta),-1))[None])
    np.testing.assert_allclose(actual.numpy()[0],y,atol=2e-14,rtol=0)


def test_event_return_order_and_censoring():
    from tests.paper_work.completion.analyze import events
    p=np.zeros((2,4,1,3));p[0,:,0,2]=[1.,-.2,.9,.9];p[1,:,0,2]=[1.,.9,.3,.2]
    rows=events(p,.1,-.8,.8)
    r={x['event']:x for x in rows if x['dwell']==0.}
    assert r['crossed_zero']['count']==1
    assert r['reverse_core_residence']['count']==0
    assert r['crossing_return']['count']==1
    assert r['unresolved_endpoint']['count']==1


def test_six_figure_pipeline_on_synthetic_data(tmp_path, monkeypatch):
    import json
    from tests.paper_work.completion import analyze
    from tests.paper_work.completion.run import cfg
    c=cfg();states=[]
    for L in c['sites']:
        for theta in c['theta']:
            for kind in c['kinds']:
                states.append(dict(state=len(states),sites=L,theta=theta,kind=kind))
    refs=[dict(arm=arm,state=s['state'],duration=32.) for arm in ('mt','heun') for s in states]
    refs += [dict(arm=arm,state=s,duration=200.) for arm in ('mt','heun') for s in c['long_states']]
    refs += [dict(arm='calibration',state=s,duration=200.) for s in (0,3,6,9,12,15)]
    generated=[dict(arm=arm,state=s['state'],duration=32.) for arm in ('frozen','selected','deterministic') for s in states]
    generated += [dict(arm=arm,state=s,duration=200.) for arm in ('frozen','selected','deterministic') for s in c['long_states']]
    training=[dict(kind=k,seed=s,fraction=1.) for k in ('flow','deterministic') for s in c['seeds']]
    manifest=dict(config=c,states=states,reference=refs,generation=generated,training=training)
    for name in ('results','figures','tables'):(tmp_path/name).mkdir()
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'SYNTHETIC_TEST_ONLY.txt').write_text('Synthetic test fixtures. Not research results.')
    for i,spec in enumerate(training):
        report=dict(pair_count=12288,path_count=1536,selected_step=6000,selected_score=.1,
                    seconds_training_and_validation=1.,validation=[dict(step=6000,score=.1)])
        (tmp_path/'results'/f'train{i}.json').write_text(json.dumps(report))
    bench=[dict(arm=arm,sites=L,paths=n,seconds=.1,repeat=1,duration=.08) for arm in ('mt','flow') for L in c['sites'] for n in (1,64,128)]
    (tmp_path/'results/benchmark.json').write_text(json.dumps(dict(rows=bench)))
    def fake_read(name):
        if name.startswith('train'):
            state=int(name.split('test')[1]);task=dict(state=state,duration=.08,arm='flow')
        else:
            prefix,index=name.split('_');index=int(index)
            task=refs[index] if prefix=='reference' else generated[index] if prefix=='generation' else [t for t in generated if t['arm']=='frozen'][index]
        L=states[task['state']]['sites'];dt=.1 if task['arm']=='calibration' else .01
        t=np.arange(round(task['duration']/dt)+1)*dt
        x=np.zeros((2,1,1,3));x[0,...,2]=1.;x[1,...,2]=-1.
        return np.broadcast_to(x,(2,len(t),L,3)),t,dict(seconds_excluding_save=.1)
    monkeypatch.setattr(analyze,'ROOT',tmp_path);monkeypatch.setattr(analyze,'load',lambda:manifest)
    monkeypatch.setattr(analyze,'read',fake_read)
    analyze.run()
    assert len(list((tmp_path/'figures').glob('*.png')))==6
    assert (tmp_path/'index.html').read_text().count('data:image/png;base64,')==6
