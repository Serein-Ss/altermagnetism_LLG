import copy
import pytest
from tests.experiments.summarize_stable_sampling import validate_report


def test_sampling_report_completeness():
    steps=(128,256,512,1024)
    report=dict(status='COMPLETE_SAMPLING_DIAGNOSTIC',rows=[
        dict(theta=t,steps=s,families=[dict(family=0),dict(family=1)],previous_evaluation_max_abs_difference=0.)
        for t in (.11,.13) for s in steps])
    validate_report(report,steps)
    for change in ('missing','duplicate','family','replay','running'):
        bad=copy.deepcopy(report)
        if change=='missing':bad['rows'].pop()
        elif change=='duplicate':bad['rows'][-1]=bad['rows'][0]
        elif change=='family':bad['rows'][0]['families'].pop()
        elif change=='replay':bad['rows'][0]['previous_evaluation_max_abs_difference']=.1
        else:bad['status']='RUNNING'
        with pytest.raises(ValueError):validate_report(bad,steps)
