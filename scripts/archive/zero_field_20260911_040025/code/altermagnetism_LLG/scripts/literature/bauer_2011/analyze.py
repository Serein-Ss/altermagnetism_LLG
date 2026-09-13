"""Trajectory-clustered reversal statistics; never discard censored chains."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scripts.literature.workflow import require_complete,save_json


def reversals(time,magnetization,threshold,dwell):
    if not 0<threshold<1 or dwell<0:
        raise ValueError('0<threshold<1 and nonnegative dwell required')
    if abs(magnetization[0])<threshold:
        raise ValueError('initial state is outside the specified starting basin')
    basin=int(np.sign(magnetization[0])); start=None; events=[]
    for i,(t,m) in enumerate(zip(time,magnetization)):
        if -basin*m>=threshold:
            if start is None: start=i
            if t-time[start]>=dwell:
                # Completion, not first entry, is the event time.
                events.append(float(t)); basin=-basin; start=None
        else:
            start=None
    return np.array(events)


def kaplan_meier(durations,observed):
    durations=np.asarray(durations,float); observed=np.asarray(observed,bool)
    if len(durations)==0 or durations.shape!=observed.shape or np.any(durations<0):
        raise ValueError('nonempty matching nonnegative survival data required')
    event_times=np.unique(durations[observed]); survival=1.; times=[0.]; values=[1.]
    for t in event_times:
        risk=np.count_nonzero(durations>=t)
        deaths=np.count_nonzero((durations==t)&observed)
        survival*=1-deaths/risk
        times.append(float(t)); values.append(float(survival))
    return np.array(times),np.array(values)


def restricted_mean(durations,observed,horizon):
    t,s=kaplan_meier(durations,observed)
    keep=t<horizon; t=t[keep]; s=s[keep]
    return float(np.sum(np.diff(np.r_[t,horizon])*s))


def statistics(time,magnetization,*,threshold,dwell,bootstrap=1000,seed=0):
    events=[reversals(time,m,threshold,dwell) for m in magnetization.T]
    horizon=float(time[-1]-time[0]); n=len(events)
    observed=np.array([len(e)>0 for e in events])
    durations=np.array([e[0]-time[0] if len(e) else horizon for e in events])
    t,s=kaplan_meier(durations,observed)
    rng=np.random.default_rng(seed)
    rmst=[]; curve=[]; recurrent=[]
    for _ in range(bootstrap):
        idx=rng.integers(0,n,n)
        rmst.append(restricted_mean(durations[idx],observed[idx],horizon))
        bt,bs=kaplan_meier(durations[idx],observed[idx])
        curve.append(bs[np.searchsorted(bt,t,side='right')-1])
        intervals=[np.diff(events[i]) for i in idx if len(events[i])>1]
        if intervals: recurrent.append(float(np.concatenate(intervals).mean()))
    intervals=[np.diff(e) for e in events if len(e)>1]
    return {'independent_trajectories':n,'completed_events':sum(map(len,events)),
            'minimum_500_events_met':sum(map(len,events))>=500,
            'first_pass_observed':observed.tolist(),'first_pass_durations':durations.tolist(),
            'right_censored':int((~observed).sum()),'event_times':[e.tolist() for e in events],
            'basin_threshold':threshold,'dwell_reduced':dwell,
            'threshold_status':'explicit_operational_choice_not_basin_certification',
            'saved_time_resolution':float(np.max(np.diff(time))),
            'km_time':t.tolist(),'km_survival':s.tolist(),
            'km_pointwise_ci95':np.quantile(curve,[.025,.975],axis=0).tolist() if n>1 else None,
            'rmst_horizon':horizon,'rmst':restricted_mean(durations,observed,horizon),
            'rmst_ci95':np.quantile(rmst,[.025,.975]).tolist() if n>1 else None,
            'unrestricted_mean_lifetime':None,
            'mean_completed_recurrent_interval':float(np.concatenate(intervals).mean()) if intervals else None,
            'completed_recurrent_interval_ci95':np.quantile(recurrent,[.025,.975]).tolist() if recurrent and n>1 else None,
            'recurrent_warning':'Completed-interval mean is truncation-biased; not an unrestricted lifetime estimator.',
            'bootstrap_unit':'whole_independent_trajectory','bootstrap_seed':seed}


def main():
    p=argparse.ArgumentParser(); p.add_argument('input',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--basin-threshold',type=float,required=True); p.add_argument('--dwell',type=float,required=True)
    p.add_argument('--bootstrap',type=int,default=1000); args=p.parse_args()
    with h5py.File(args.input) as h:
        require_complete(h); t=h['time'][:]
        mz=np.stack([frame[...,2].mean(-1) for frame in h['spins']])
    result=statistics(t,mz,threshold=args.basin_threshold,dwell=args.dwell,bootstrap=args.bootstrap)
    save_json(args.output,result)


if __name__=='__main__': main()
