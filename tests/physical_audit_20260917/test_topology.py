import numpy as np
from tests.physical_audit_20260917.gallery import triangular_charge


def texture():
    x,y=np.meshgrid(np.arange(48),np.arange(40),indexing='ij')
    p=np.stack((x,np.sqrt(3)*y),-1)[...,None,:]+np.array([[0,0],[.5,np.sqrt(3)/2]])
    p-=p.reshape(-1,2).mean(0)
    r=np.linalg.norm(p,axis=-1)
    angle=np.pi*.5*(1+np.tanh((r-5)/1.6))
    s=np.concatenate((np.sin(angle)[...,None]*p/np.maximum(r[...,None],1e-10),np.cos(angle)[...,None]),-1)
    return s[None,None]


def test_uniform_zero_charge():
    s=np.zeros((1,1,8,8,2,3)); s[...,2]=1
    q,qi=triangular_charge(s)
    np.testing.assert_allclose(q,0,atol=1e-12)
    np.testing.assert_allclose(qi,0,atol=1e-12)


def test_known_texture_and_rotation():
    s=texture(); q,qi=triangular_charge(s)
    np.testing.assert_allclose(np.abs(qi),1,atol=1e-6)
    rotation=np.array([[0,1,0],[-1,0,0],[0,0,1]])
    qr,qir=triangular_charge(s@rotation)
    np.testing.assert_allclose(qr,q,atol=1e-12)
    np.testing.assert_allclose(qir,qi,atol=1e-12)
    qn,qni=triangular_charge(-s)
    np.testing.assert_allclose(qn,-q,atol=1e-12)
    np.testing.assert_allclose(qni,-qi,atol=1e-12)
