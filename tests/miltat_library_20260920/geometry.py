"""Contiguous positive core diagnostic for the prescribed negative-background protocol."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class CoreGeometry:
    def __init__(self, positions_nm, spacing_nm=.251):
        self.positions = np.asarray(positions_nm).reshape(-1,3)
        self.spacing = spacing_nm
        pairs = cKDTree(self.positions).query_pairs(spacing_nm*(1+1e-6),output_type='ndarray')
        distances = np.linalg.norm(self.positions[pairs[:,0]]-self.positions[pairs[:,1]],axis=-1)
        if not np.allclose(distances,spacing_nm,rtol=1e-6,atol=1e-8):
            raise ValueError('Expected unique sites on a nearest-neighbor triangular lattice')
        a,b=pairs.T
        self.graph=coo_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),
                             shape=(len(self.positions),)*2).tocsr()
        self.degree=np.diff(self.graph.indptr)
        if np.any(self.degree>6):raise ValueError('Triangular coordination exceeded')
        self.boundary=self.degree<6

    def measure(self, spins):
        s=np.asarray(spins).reshape(-1,3)
        if s.shape!=self.positions.shape or not np.isfinite(s).all():
            raise ValueError('Spin/geometry mismatch or nonfinite values')
        ids=np.flatnonzero(s[:,2]>=.5)
        background=float(s[self.boundary,2].mean())
        result=dict(threshold=.5,background_edge_mean_sz=background,threshold_sites=len(ids),
                    component_count=0,largest_component_sites=0,largest_fraction=0.,
                    isolated_core_center_nm=None,isolated_core_area_radius_nm=None,
                    status='NO_POSITIVE_CORE')
        if not len(ids):return result
        count,labels=connected_components(self.graph[ids][:,ids],directed=False)
        sizes=np.bincount(labels);core=ids[labels==sizes.argmax()]
        touches=bool(self.boundary[core].any())
        status=('CORE_TOUCHES_BOUNDARY' if touches else
                'BACKGROUND_NOT_NEGATIVE' if background>=0 else 'ISOLATED_POSITIVE_CORE')
        result.update(component_count=int(count),largest_component_sites=len(core),
                      largest_fraction=float(len(core)/len(ids)),touches_boundary=touches,status=status)
        if status=='ISOLATED_POSITIVE_CORE':
            center=self.positions[core,:2].mean(0)
            radius=np.sqrt(len(core)*np.sqrt(3)/2*self.spacing**2/np.pi)
            result.update(isolated_core_center_nm=center.tolist(),isolated_core_area_radius_nm=float(radius))
        return result
