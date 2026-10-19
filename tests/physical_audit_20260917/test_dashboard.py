import numpy as np
import pytest
from tests.physical_audit_20260917.dashboard import select_sites, component_grid


@pytest.mark.parametrize('name,shape',[('bauer',(50,1,3)),('miltat',(200,116,2,3)),('hirst',(30,30,30,4,3))])
def test_grid_preserves_every_signed_component_without_averaging(name,shape):
    ids,pos,_=select_sites(shape,name)
    vectors=np.random.default_rng(71).uniform(-1,1,(len(ids),3))
    grid,rows,columns=component_grid(pos,vectors)
    np.testing.assert_array_equal(grid[rows,columns],vectors)
    assert np.isfinite(grid[:,:,0]).sum()==len(ids)
    if name=='hirst':
        assert grid.shape==(4,19,3)
        assert np.isnan(grid[:,[4,9,14]]).all()


@pytest.mark.parametrize('name,shape',[('bauer',(50,1,3)),('miltat',(200,116,2,3)),('hirst',(30,30,30,4,3))])
def test_fixed_spatial_selection_matches_actual_lattice_indices(name,shape):
    ids,pos,labels=select_sites(shape,name)
    assert len(ids)==len(set(ids)) and np.isfinite(pos).all()
    assert ids.min()>=0 and ids.max()<np.prod(shape[:-1])
    np.testing.assert_array_equal(np.ravel_multi_index(labels.T,shape[:-1]),ids)
    if name=='hirst':
        assert np.all(labels[:,-1]==0)
    if name=='bauer':
        np.testing.assert_array_equal(ids,np.arange(19,31))
