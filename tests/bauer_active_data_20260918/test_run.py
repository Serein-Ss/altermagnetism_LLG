import numpy as np
import yaml
from .run import BASE,candidate,descriptors,fps

def main():
    cfg=yaml.safe_load((BASE/'config.yaml').read_text())
    for length in cfg['physics']['sizes']:
        states=np.array([candidate(np.random.default_rng(100+i),length,f,cfg)
                         for i,f in enumerate(cfg['sampling']['families'])])
        assert states.shape==(6,length,3)
        assert np.max(abs(np.linalg.norm(states,axis=-1)-1))<1e-12
        d=descriptors(states); assert np.isfinite(d).all() and d.shape==(6,22)
        ids=fps(d,6); assert len(set(ids))==6
        np.testing.assert_array_equal(ids,fps(d,6))
    print('PASS: candidates, norm, descriptors, FPS uniqueness/determinism for all sizes')

if __name__=='__main__': main()
