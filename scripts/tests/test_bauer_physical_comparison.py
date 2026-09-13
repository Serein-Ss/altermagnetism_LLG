import numpy as np
from scripts.analysis.bauer_physical_comparison import events, coarse_mechanism


def test_event_labels_and_censoring():
    m=np.ones((1,5,201))*.9
    m[0,1,10:]=-.9
    m[0,2,10]=-.1
    m[0,3,-1]=0
    m[0,4,-1]=-.9
    labels,first=events(m)
    assert not labels[0,0].any() and np.isinf(first[0,0])
    assert labels[0,1].tolist()==[True,True,True,False,False]
    assert first[0,1]==960
    assert labels[0,2].tolist()==[False,True,False,True,False]
    assert labels[0,3,4]
    assert labels[0,4,0] and not labels[0,4,2]


def test_patch_ties_do_not_prefer_left_edge():
    p=np.zeros((1,16,201,25,3));p[...,2]=1
    p[:,:,10:,:,2]=-1
    _,first=events(p[...,2].mean(-1))
    rows=coarse_mechanism(p,first)
    assert rows[0]['patch_site']==13
    assert rows[0]['simultaneous_first_blocks']==21
    assert rows[0]['proxy_to_switch_time']==0
