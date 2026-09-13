"""Gate-F feature/ED primitives. Not a full GO decision or event certificate.

Use externally frozen training scales. Inference families, hierarchical
bootstrap and physical ensemble certificates must accompany these point estimates.
"""
import numpy as np


def primary_features(energy_per_spin, magnetization, neel, indices):
    energy, magnetization, neel = map(np.asarray, (energy_per_spin, magnetization, neel))
    if energy.ndim != 2 or magnetization.shape != (*energy.shape, 3) or neel.shape != magnetization.shape:
        raise ValueError('expected [paths,frames] and [paths,frames,3]')
    if len(indices) != 11 or len(set(indices)) != 11 or min(indices) < 0 or max(indices) >= energy.shape[1]:
        raise ValueError('eleven distinct frozen frame indices required')
    result = np.concatenate((energy[..., None], magnetization, neel), -1)[:, indices].reshape(len(energy), -1)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite primary feature')
    return result


def fit_training_scale(features, floor, *, split):
    x, floor = np.asarray(features), np.asarray(floor)
    if split != 'train' or x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('finite training features required')
    if floor.shape != (x.shape[1],) or not np.isfinite(floor).all() or (floor <= 0).any():
        raise ValueError('pre-frozen positive physical scale floor per feature required')
    return x.mean(0), np.maximum(x.std(0), floor)


def energy_distance(x, y):
    """Equal-count empirical V statistic; distance normalized by sqrt(dimension)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape != y.shape or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('matching finite ensembles required')
    def distance(a, b):
        return np.linalg.norm(a[:, None]-b[None], axis=-1).mean()/np.sqrt(a.shape[1])
    return float(2*distance(x, y)-distance(x, x)-distance(y, y))


def fixed_initial_comparison(real_a, real_b, generated):
    """Inputs are already standardized. Never pool distinct initial states here."""
    baseline = energy_distance(real_a, real_b)
    candidate = energy_distance(real_a, generated)
    return dict(real_real_ed=baseline, generated_real_ed=candidate,
                ed_excess=candidate-baseline,
                mean_difference=(np.asarray(generated).mean(0)-np.asarray(real_a).mean(0)).tolist(),
                status='point_estimates_only_not_GO')


def simultaneous_intervals(estimate, replicates, confidence=.975):
    """Centered bootstrap max-error band over the entire registered vector.

    All entries must already be in their frozen comparable physical/standardized
    scales. Call ONCE with all conditions/seeds/baselines, not once per group.
    This finite-sample bootstrap is approximate, not an exact coverage guarantee.
    """
    estimate, replicates = np.asarray(estimate), np.asarray(replicates)
    if estimate.ndim != 1 or replicates.ndim != 2 or replicates.shape[1] != len(estimate) or len(replicates) < 2000:
        raise ValueError('complete family and at least 2000 bootstrap replicates required')
    if not np.isfinite(estimate).all() or not np.isfinite(replicates).all() or not .95 <= confidence < 1:
        raise ValueError('finite estimates and registered confidence required')
    radius = np.quantile(np.max(np.abs(replicates-estimate), axis=1), confidence)
    return np.stack((estimate-radius, estimate+radius), -1)


def bootstrap_family(groups, model_names, *, repetitions=2000, seed, confidence=.975):
    """Source-cluster then independent-noise bootstrap, equal initial weights.

    Each group supplies real_a/real_b [initial,replicate,feature], generated
    {model_name: same shape}, source_families and initial_ids. Names must encode
    training seed as well as model. Input groups are pre-standardized; caller
    freezes and supplies the complete comparison family. Real halves are fixed
    before this function; generated models are independently resampled here.
    Sampling-step paired comparisons require a separate paired-noise analysis.
    """
    if repetitions < 2000 or len(set(model_names)) != len(model_names) or not groups:
        raise ValueError('nonempty registered family and >=2000 replicates required')
    rng = np.random.default_rng(seed)
    checked = []
    seen_sources = set()
    labels = []
    for group in groups:
        a, b = np.asarray(group['real_a']), np.asarray(group['real_b'])
        if a.ndim != 3 or a.shape != b.shape or a.shape[0] < 2 or a.shape[1] < 2:
            raise ValueError('at least two initial states with matching ensembles required')
        generated = {name: np.asarray(group['generated'][name]) for name in model_names}
        if any(g.shape != a.shape for g in generated.values()):
            raise ValueError('all model ensembles must match the real half size')
        if not all(np.isfinite(v).all() for v in [a, b, *generated.values()]):
            raise ValueError('nonfinite family')
        sources = np.asarray(group['source_families'])
        if len(sources) != len(a) or len(set(group['initial_ids'])) != len(a) or len(set(sources)) < 2:
            raise ValueError('at least two independent source families required')
        if seen_sources.intersection(sources):
            raise ValueError('shared source families across groups need a joint cross-group bootstrap; not supported here')
        seen_sources.update(sources)
        checked.append((a, b, generated, sources))
        for name in model_names:
            labels.append(f"{group['group_id']}/{name}/ed_excess")
            labels.extend(f"{group['group_id']}/{name}/mean/{d}" for d in range(a.shape[-1]))
        for baseline in (name for name in model_names if not name.startswith('rfm:')):
            for name in (name for name in model_names if name.startswith('rfm:')):
                labels.append(f"{group['group_id']}/{name}/minus/{baseline}")

    def compute(resample):
        values = []
        for a, b, models, sources in checked:
            families = np.unique(sources)
            selected = rng.choice(families, len(families), replace=True) if resample else families
            indices = np.concatenate([np.flatnonzero(sources == family) for family in selected])
            estimates = {name: [] for name in model_names}
            for index in indices:
                count = a.shape[1]
                ai = a[index, rng.integers(count, size=count)] if resample else a[index]
                bi = b[index, rng.integers(count, size=count)] if resample else b[index]
                reference = energy_distance(ai, bi)
                for name, model in models.items():
                    gi = model[index, rng.integers(count, size=count)] if resample else model[index]
                    estimates[name].append(np.r_[energy_distance(ai, gi)-reference, gi.mean(0)-ai.mean(0)])
            means = {name: np.mean(rows, axis=0) for name, rows in estimates.items()}
            for name in model_names:
                values.extend(means[name])
            for baseline in (name for name in model_names if not name.startswith('rfm:')):
                for name in (name for name in model_names if name.startswith('rfm:')):
                    values.append(means[name][0]-means[baseline][0])
        return np.asarray(values)

    estimate = compute(False)
    draws = np.stack([compute(True) for _ in range(repetitions)])
    intervals = simultaneous_intervals(estimate, draws, confidence)
    return dict(status='distribution_evidence_only_not_GO', labels=labels, estimate=estimate.tolist(),
                simultaneous_ci=intervals.tolist(), bootstrap_seed=seed, repetitions=repetitions,
                confidence=confidence, resampling='source_cluster_then_independent_noise',
                limitation='few source families may give unstable intervals; no event or full-path certificate')
