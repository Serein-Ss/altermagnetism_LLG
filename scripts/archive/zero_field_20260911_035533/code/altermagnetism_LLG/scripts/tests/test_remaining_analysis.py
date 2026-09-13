import numpy as np
from scripts.analysis.analyze_remaining_validation import windows,peaks,clean


def test_windows_no_overlap():
    result=windows(np.arange(9),np.arange(9)[:,None])
    np.testing.assert_allclose(result['first'],[4.5])
    np.testing.assert_allclose(result['second'],[7])
    assert not result['certified']


def test_signed_peaks_not_selected_using_reference():
    power=np.zeros((7,1,4,2));power[2,:,:,:]=5;power[5,:,:,:]=3
    spec=dict(omega=np.array([0,1,2,3,-3,-2,-1]),power=power,
              kx=np.array([np.pi/2,0,np.pi/2,-np.pi/2]),ky=np.array([0,np.pi/2,np.pi/2,np.pi/2]))
    rows=peaks(spec,dict(J1=1,J2=.1,J_tilde=.1,K_SW=0))
    assert len(rows)==4
    assert all(r['positive_peak']==2 and r['negative_peak']==-2 and r['signed_splitting']==0 for r in rows)


def test_missing_numeric_is_null_not_zero():
    assert clean(dict(x=np.array([np.nan,np.inf,2.])))==dict(x=[None,None,2.])
