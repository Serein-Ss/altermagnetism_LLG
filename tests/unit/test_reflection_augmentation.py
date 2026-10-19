import torch
import copy

from tests.experiments.reflection_augmentation import augment_pairs, reflect_pairs


def test_reversal_preserves_pair_direction_and_is_involution():
    a = torch.arange(4 * 25 * 3).reshape(4, 25, 3).float()
    b = a + 1000
    before = a.clone(), b.clone()
    mask = torch.tensor([False, True, True, False])
    x, y = reflect_pairs(a, b, mask)
    assert torch.equal(x[0], a[0]) and torch.equal(x[1], a[1].flip(0))
    assert torch.equal(y - x, b - a)
    aa, bb = reflect_pairs(x, y, mask)
    assert torch.equal(aa, a) and torch.equal(bb, b)
    assert torch.equal(a, before[0]) and torch.equal(b, before[1])


def test_augmentation_reproducible_without_consuming_training_or_global_rng():
    train_rng = torch.Generator().manual_seed(9)
    augmentation_rng = torch.Generator().manual_seed(1004000011)
    a = torch.ones(128, 25, 3)
    train_state = train_rng.get_state().clone()
    global_state = torch.random.get_rng_state().clone()
    augmentation_state = augmentation_rng.get_state().clone()
    first = augment_pairs(a, a, augmentation_rng)
    assert torch.equal(train_rng.get_state(), train_state)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert not torch.equal(augmentation_rng.get_state(), augmentation_state)
    augmentation_rng.set_state(augmentation_state)
    second = augment_pairs(a, a, augmentation_rng)
    assert all(torch.equal(x, y) for x, y in zip(first, second))


def test_augmented_step_matches_explicit_transform_and_control_rng():
    from tests.experiments.director_transition import DirectorTransition
    from tests.experiments.director_budget import train_step
    from tests.experiments.reflection_augmentation import augmented_train_step
    torch.manual_seed(4)
    model = DirectorTransition(.12, .01, width=8)
    explicit, control = copy.deepcopy(model), copy.deepcopy(model)
    a = torch.nn.functional.normalize(torch.randn(4, 25, 3), dim=-1)
    b = torch.nn.functional.normalize(a + .01 * torch.randn_like(a), dim=-1)
    theta, lag = torch.full((4,), .11), torch.full((4,), .01)
    axis = torch.tensor([0., 0., 1.]).expand(4, -1)
    rngs = [torch.Generator().manual_seed(8) for _ in range(3)]
    aug = [torch.Generator().manual_seed(31) for _ in range(2)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=.001) for m in [model, explicit, control]]
    actual = augmented_train_step(model, optimizers[0], a, b, theta, lag, axis, rngs[0], .016, aug[0])
    x, y, mask = augment_pairs(a, b, aug[1])
    expected = train_step(explicit, optimizers[1], x, y, theta, lag, axis, rngs[1], .016)
    train_step(control, optimizers[2], a, b, theta, lag, axis, rngs[2], .016)
    assert actual[:2] == expected[:2] and actual[3] == int(mask.sum())
    for p, q in zip(model.parameters(), explicit.parameters()):
        assert torch.equal(p, q) and torch.isfinite(p).all()
    assert all(torch.equal(rngs[0].get_state(), r.get_state()) for r in rngs[1:])
    assert torch.equal(aug[0].get_state(), aug[1].get_state())
