from .run import rate,tasks


def test_lr_and_factorial_pairing():
    rows=tasks();assert len(rows)==12
    for seed in (11,22,33):
        assert {t['repair'] for t in rows if t['seed']==seed}=={'fresh_constant','fresh_decay','warm_constant','warm_decay'}
    assert rate('fresh_decay',1,6000)==.001
    assert abs(rate('fresh_decay',6000,6000)-.00003)<1e-12
    for step in (1,1000,3000,6000):
        assert rate('warm_decay',step,6000)==rate('fresh_decay',step,6000)
        assert rate('warm_constant',step,6000)==.001
