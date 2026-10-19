import numpy as np
from tests.miltat_library_20260920.geometry import CoreGeometry


def geometry():
    x,y=np.meshgrid(np.arange(32),np.arange(32))
    pos=np.stack((x+.5*(y%2),y*np.sqrt(3)/2,np.zeros_like(x)),-1)*.251
    return CoreGeometry(pos)


def test_homogeneous_states_are_not_skyrmions():
    g=geometry();s=np.zeros_like(g.positions);s[:,2]=-1
    assert g.measure(s)['status']=='NO_POSITIVE_CORE'
    s[:,2]=1
    result=g.measure(s)
    assert result['status']=='CORE_TOUCHES_BOUNDARY'
    assert result['isolated_core_area_radius_nm'] is None


def test_isolated_core_and_boundary_escape():
    g=geometry();s=np.zeros_like(g.positions);s[:,2]=-1
    delta=g.positions-g.positions.mean(0)
    s[np.linalg.norm(delta,axis=-1)<.7,2]=1
    result=g.measure(s)
    assert result['status']=='ISOLATED_POSITIVE_CORE'
    assert result['component_count']==1
    assert 0.5<result['isolated_core_area_radius_nm']<.9
    s[g.positions[:,0]<4,2]=1
    assert g.measure(s)['status']=='CORE_TOUCHES_BOUNDARY'


def test_small_disconnected_patch_is_reported():
    g=geometry();s=np.zeros_like(g.positions);s[:,2]=-1
    delta=g.positions-g.positions.mean(0)
    s[np.linalg.norm(delta,axis=-1)<.7,2]=1
    s[0,2]=1
    result=g.measure(s)
    assert result['status']=='ISOLATED_POSITIVE_CORE'
    assert result['component_count']==2 and result['largest_fraction']<1
