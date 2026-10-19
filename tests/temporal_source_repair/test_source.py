import unittest
import torch
from torch.nn import functional as F
from tests.cross_dimension_20260924.model import source as old_source
from .source import correlate,from_noise,draw,contract,check_package


class SourceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(938);self.a=F.normalize(torch.randn(4,16,3,dtype=torch.float64),dim=-1)

    def test_original_exact_and_resume(self):
        a=draw(self.a,11,torch.Generator().manual_seed(1),0.)
        b=old_source(self.a,11,torch.Generator().manual_seed(1),2.)
        self.assertTrue(torch.equal(a,b))
        gen=torch.Generator().manual_seed(3);state=gen.get_state()
        a=draw(self.a,11,gen,.8);gen.set_state(state)
        self.assertTrue(torch.equal(a,draw(self.a,11,gen,.8)))

    def test_structure_rotation(self):
        noise=torch.randn(4,11,16,3,dtype=torch.float64)
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));q[:,0]*=torch.linalg.det(q)
        for rho in (0.,.8):
            z=from_noise(self.a,noise,rho);rot=from_noise(self.a@q,noise@q,rho)
            self.assertTrue(torch.equal(z[:,0],self.a))
            self.assertLess(float(abs(z.norm(dim=-1)-1).max()),1e-12)
            self.assertLess(float(abs(rot-z@q).max()),1e-12)

    def test_marginal_and_correlation(self):
        e=torch.randn(10000,11,2,3,dtype=torch.float64);v=correlate(e,.8)[:,1:]
        self.assertLess(float(abs(v.mean((0,2,3))).max()),.025)
        self.assertLess(float(abs(v.var((0,2,3))-1).max()),.035)
        for lag in (1,2,5):
            self.assertLess(abs(float((v[:,lag:]*v[:,:-lag]).mean())-.8**lag),.025)
        self.assertLess(abs(float((v[:,:,0]*v[:,:,1]).mean())),.025)
        self.assertGreater(float(torch.linalg.eigvalsh(torch.cov(v[:,:,0,0].T)).min()),.05)
        initial=self.a[:1].expand(10000,-1,-1)[:,:2]
        a=from_noise(initial,e,0.);b=from_noise(initial,e,.8)
        self.assertLess(float(abs(a[:,1:].mean((0,2))-b[:,1:].mean((0,2))).max()),.035)

    def test_contract(self):
        ck=dict(source_contract=contract(.8));check_package(ck,.8)
        with self.assertRaises(ValueError):check_package(ck,0.)
        for rho in (-1,1,float('nan')):
            with self.assertRaises(ValueError):correlate(torch.zeros(1,11,2,3),rho)


if __name__=='__main__':unittest.main()
