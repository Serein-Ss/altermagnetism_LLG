"""Scalar-budget controls, not an equality claim for optimizer trajectories."""
import math

SETTINGS={'sparse64':(64,.1),'dense8_same_mass':(8,.0125),'dense8_strong':(8,.1)}


def setting(arm):
    return SETTINGS[arm]


def call_due(step,arm,smoke=False):
    return step%(8 if smoke else setting(arm)[0])==0


def weight_at(step,arm,smoke=False):
    endpoint=step if smoke else math.ceil(step/64)*64
    return setting(arm)[1]*min(1.,endpoint/(16 if smoke else 1000))


def role_at(step,smoke=False):
    return 2*((step//(8 if smoke else 64)-1)%2) if smoke else 2*(((step-1)//64)%2)


def expected_calls(total,arm,smoke=False):
    return total//(8 if smoke else setting(arm)[0])
