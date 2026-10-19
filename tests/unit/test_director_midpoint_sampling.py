import numpy as np
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.director_sampling_convergence import rollout as old_rollout
from tests.experiments.director_midpoint_sampling import rollout, euler_sample


def test_replay_and_paired_random_stream():
    model=DirectorTransition(.12,.01,8).eval()
    initial=np.tile([0.,0.,1.],(5,1))
    cfg=dict(generated_paths=5,generation_batch=3,lag=.01,axis=[0.,0.,1.],rollout_steps=2)
    args=(model,initial,.11,cfg,.02,4,123,'cpu')
    original, state=old_rollout(*args)
    replay, replay_state=rollout(*args,sampler=euler_sample)
    midpoint, mid_state=rollout(*args)
    np.testing.assert_array_equal(original,replay)
    np.testing.assert_array_equal(state,replay_state)
    np.testing.assert_array_equal(state,mid_state)
    assert midpoint.shape==original.shape and np.isfinite(midpoint).all()
    np.testing.assert_allclose(np.linalg.norm(midpoint,axis=-1),1,atol=2e-7)
