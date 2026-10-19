"""Hold the previously tested sparse auxiliary schedule fixed."""
from tests.auxiliary_frequency_repair.recipe import role_at
def setting(arm):
    if arm not in ('whole','patch2'):raise ValueError('unregistered teacher arm')
    return 64,.1
def call_due(step,arm,smoke=False):
    setting(arm);return step%(8 if smoke else 64)==0
def weight_at(step,arm,smoke=False):
    setting(arm);return .1*min(1.,step/(16 if smoke else 1000))
def expected_calls(total,arm,smoke=False):
    setting(arm);return total//(8 if smoke else 64)
