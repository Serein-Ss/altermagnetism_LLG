"""Geometry and artifact checks against the actual saved lattice."""
import unittest
import h5py
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from .miltat_view import SOURCE, OUT, HERE, A, raster, triangular_graph, core_measure


class MiltatTests(unittest.TestCase):
    def test_graph_matches_physical_positions(self):
        with h5py.File(SOURCE/'thermal.h5') as h:
            p=raster(h['positions_nm'][:])
        g=triangular_graph(*p.shape[:2]);p=p.reshape(-1,3)
        row,col=g.nonzero()
        np.testing.assert_allclose(np.linalg.norm(p[row]-p[col],axis=1),A,atol=1e-6)
        self.assertEqual(int(g.sum(axis=1).max()),6)
        self.assertEqual((g-g.T).nnz,0)

    def test_largest_core(self):
        g=triangular_graph(4,4);p=np.zeros((16,3));p[:,0]=np.arange(16)
        s=np.zeros((16,3));s[:,2]=-1;s[[0,1,15],2]=1
        center,size,count,fraction=core_measure(s,p,g)
        self.assertEqual((size,count),(2,2))
        np.testing.assert_allclose(center,[.5,0])
        self.assertAlmostEqual(fraction,2/3)
        s[:,2]=-1
        self.assertEqual(core_measure(s,p,g)[1],0)

    def test_visual_artifacts(self):
        sheet=Image.new('RGB',(1200,1650),'white');draw=ImageDraw.Draw(sheet)
        for i,name in enumerate(['textures.png','diagnostics.png','thermal.gif']):
            with Image.open(OUT/name) as im:
                if name.endswith('.gif'):
                    self.assertEqual(im.n_frames,51);im.seek(25)
                else:self.assertGreaterEqual(min(im.info['dpi']),299)
                thumb=ImageOps.contain(im.convert('RGB'),(1180,510))
                sheet.paste(thumb,(10,550*i+25));draw.text((10,550*i+5),name,fill='black')
        sheet.save(HERE/'logs/miltat_visual_qa.png')


if __name__=='__main__':unittest.main(verbosity=2)
