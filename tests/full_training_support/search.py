import numpy as np


def eligible_frames(starts, steps, frames):
    """Actual transition starts, not last frames used only as future targets."""
    result = np.unique([start+offset for start in starts for offset in range(steps)])
    if len(result) == 0 or result.min() < 0 or result.max()+1 >= frames:
        raise ValueError('invalid transition grid')
    return result


def patch_batch(states, radius):
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 3 or states.shape[-1] != 3 or radius < 1:
        raise ValueError('batch of chain states required')
    size = states.shape[1]
    indices = np.arange(size)[:, None]+np.arange(-radius, radius+1)
    mask = (indices >= 0) & (indices < size)
    p = states[:, np.clip(indices, 0, size-1)]*mask[None, ..., None]
    return p.reshape(-1, 2*radius+1, 3), np.tile(mask, (len(states), 1))


class ExactSearch:
    """Streaming exact axial Procrustes distance with open-mask equality."""
    def __init__(self, query, masks):
        self.query = np.asarray(query, dtype=np.float64)
        self.masks = np.asarray(masks, dtype=bool)
        self.squared = np.full(len(query), np.inf)
        self.locations = [None]*len(query)
        self.groups = [np.flatnonzero(np.all(self.masks == m, axis=1))
                       for m in np.unique(self.masks, axis=0)]

    def update(self, candidates, masks, locations):
        candidates = np.asarray(candidates, dtype=np.float64)
        for ids in self.groups:
            mask = self.masks[ids[0]]
            eligible = np.flatnonzero(np.all(masks == mask, axis=1))
            # Bound the temporary matrix even for the full training inventory.
            for start in range(0, len(eligible), 4096):
                selected = eligible[start:start+4096]; b = candidates[selected]
                q = self.query[ids]
                bxy = np.ascontiguousarray(b[..., 0]+1j*b[..., 1])
                qxy = np.ascontiguousarray(q[..., 0]+1j*q[..., 1])
                dot = np.abs(bxy @ qxy.conj().T) + b[..., 2] @ q[..., 2].T
                squared = (np.sum(b*b, axis=(1, 2))[:, None] +
                           np.sum(q*q, axis=(1, 2))[None, :] - 2*dot)/mask.sum()
                best = np.argmin(squared, axis=0); value = squared[best, np.arange(len(ids))]
                for j in np.flatnonzero(value < self.squared[ids]):
                    index = ids[j]
                    self.squared[index] = value[j]
                    self.locations[index] = locations(int(selected[best[j]]))

    def distances(self):
        if not np.isfinite(self.squared).all():
            raise ValueError('query has no same-boundary-mask training candidate')
        return np.sqrt(np.maximum(0, self.squared))
