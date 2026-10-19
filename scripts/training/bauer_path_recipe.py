"""Pure schedules retained from the frozen whole-teacher control."""
import math
import numpy as np


def schedule(seed,steps):
    if steps%16:raise ValueError('complete16-update cycles required')
    rng=np.random.default_rng(seed);ids=[];branches=[]
    for _ in range(steps//16):
        for parents in rng.permutation(48).reshape(16,3):
            ids.append(np.repeat(parents,16))
            branches.append(np.concatenate([rng.choice(64,16,replace=False) for _ in parents]))
    return np.asarray(ids),np.asarray(branches)


def rate(step,total,peak=3e-4,floor=9e-6,warmup=200):
    warmup=8 if total==16 else warmup
    if step<=warmup:return peak*step/warmup
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))


def roles(step):return np.repeat(((step+np.arange(3))%2)*2,16)
