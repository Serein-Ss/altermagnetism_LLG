"""Fixed-grid recurrent events using externally calibrated basin criteria.

No automatic threshold fitting and no default |nz| cutoff. The caller must
provide a hashed independent calibration, local-order and structure criteria.
Event times are discrete detection times; cadence uncertainty is retained.
"""
import numpy as np


def classify(time,nz,order,structure,calibration):
    t,z,o,q=map(np.asarray,(time,nz,order,structure))
    if t.ndim!=1 or not (t.shape==z.shape==o.shape==q.shape) or len(t)<2:
        raise ValueError('matching one-dimensional time and observable arrays required')
    if not all(np.isfinite(a).all() for a in (t,z,o,q)) or np.any(np.diff(t)<=0):
        raise ValueError('finite observables and strictly increasing physical time required')
    if not calibration.get('independent_equilibrium_sha256') or calibration.get('status')!='pass':
        raise ValueError('independent basin calibration required')
    positive,negative=calibration['positive_nz_min'],calibration['negative_nz_max']
    dwell=calibration['dwell_reduced_time']
    if not negative<0<positive or dwell<=0: raise ValueError('invalid calibrated basin/dwell')
    ordered=(o>=calibration['order_min']) & (q<=calibration['structure_max'])
    basin=np.where(ordered & (z>=positive),1,np.where(ordered & (z<=negative),-1,0))
    if basin[0]==0: raise ValueError('initial state not in a calibrated basin; separate task required')
    initial=int(basin[0]);crossings=np.flatnonzero((z[1:]*initial<0)&(z[:-1]*initial>=0))+1
    current=initial;candidate=0;start=0;events=[]
    for i,b in enumerate(basin):
        if b==0 or b==current: candidate=0;continue
        if b!=candidate: candidate=int(b);start=i
        if t[i]-t[start]>=dwell:
            events.append(dict(from_basin=current,to_basin=int(b),entry_time=float(t[start]),
                               commitment_time=float(t[i]),detection_index=i))
            current=int(b);candidate=0
    first=next((e['commitment_time'] for e in events if e['to_basin']==-initial),None)
    terminal=int(basin[-1]);is_ordered=bool(ordered[-1])
    if not is_ordered: outcome='critical_disordering'
    elif terminal==0: outcome='unresolved_transition'
    elif first is not None: outcome='committed_switch'
    elif len(crossings) and terminal==initial: outcome='crossed_returned'
    elif len(crossings): outcome='unresolved_transition'  # reverse basin, dwell not yet established
    else: outcome='no_crossing'
    return dict(first_crossing_time=float(t[crossings[0]]) if len(crossings) else None,
        first_committed_switch_time=first,event_observed=first is not None,censor_time=float(t[-1]),
        terminal_basin=terminal,ordered_state_flag=is_ordered,outcome=outcome,
        negative_endpoint=bool(z[-1]<0),crossed_zero=bool(len(crossings)),
        crossing_times=t[crossings].tolist(),recurrent_events=events,
        time_resolution_max=float(np.diff(t).max()),calibration=calibration)
