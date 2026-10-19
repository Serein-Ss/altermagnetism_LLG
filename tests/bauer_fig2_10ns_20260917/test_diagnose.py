"""Synthetic classifier tests; no simulation data changed."""
import numpy as np
from .diagnose import events

def main():
    for middle,expected in [(np.linspace(.85,-.85,20),'committed_switch'),
                            (np.r_[np.linspace(.85,-.3,20),np.linspace(-.3,.85,20)],'crossing_return')]:
        tail=-.85 if expected=='committed_switch' else .85
        m=np.r_[np.full(30,.85),middle,np.full(30,tail)]
        rows=events(np.arange(len(m))*.001,m,.75,5)
        assert sum(r['kind']==expected for r in rows)==1
    m=np.r_[np.full(30,.85),np.zeros(20)]
    assert events(np.arange(len(m))*.001,m,.75,5)[-1]['kind']=='unresolved_transition'
    print('PASS: switch, crossing-return, unresolved synthetic cases')

if __name__=='__main__': main()
