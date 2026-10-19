import numpy as np
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.director_sampling_convergence import rollout


def test_rollout_rng_consumption_independent_of_integration_steps():
    model = DirectorTransition(.12, .01, 8).eval()
    initial = np.tile([0.,0.,1.], (5,1))
    cfg = dict(generated_paths=5, generation_batch=3, lag=.01, axis=[0.,0.,1.], rollout_steps=2)
    a, sa = rollout(model, initial, .11, cfg, .02, 2, 123, 'cpu')
    b, sb = rollout(model, initial, .11, cfg, .02, 4, 123, 'cpu')
    c, sc = rollout(model, initial, .11, cfg, .02, 2, 123, 'cpu')
    assert a.shape == b.shape == (5,3,5,3)
    np.testing.assert_array_equal(sa,sb)
    np.testing.assert_array_equal(sa,sc)
    np.testing.assert_array_equal(a,c)
    np.testing.assert_allclose(np.linalg.norm(b,axis=-1),1,atol=2e-7)
