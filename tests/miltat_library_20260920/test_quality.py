from tests.miltat_library_20260920.quality import cold_status


def test_cold_gate_rejects_lost_core_and_smoke():
    good=dict(core={'status':'ISOLATED_POSITIVE_CORE'},norm_error=1e-16,torque_max=1e-4,smoke=False)
    assert cold_status([good,good,good])
    assert not cold_status([good,dict(good,core={'status':'CORE_TOUCHES_BOUNDARY'})])
    assert not cold_status([dict(good,smoke=True)])
    assert not cold_status([dict(good,norm_error=.1)])
