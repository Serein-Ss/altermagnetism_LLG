"""Weighted Arrhenius/length fits of externally audited lifetime estimates.

Input JSON rows require theta, length, lifetime, standard_error, estimator,
reference_source. No paper grid is invented and RMST cannot silently become
an unrestricted mean lifetime. This entry does not estimate censored tails.
"""
import json
import numpy as np
from scripts.literature.workflow import analysis_arguments,save_json


def fit_scaling(rows,*,variable='inverse_temperature'):
    if variable not in ('inverse_temperature','log_length'):
        raise ValueError('unknown scaling predictor')
    for row in rows:
        if row['estimator']!='unrestricted_mean_lifetime' or not row['reference_source']:
            raise ValueError('audited unrestricted lifetime and source provenance required')
        if min(row[k] for k in ('theta','length','lifetime','standard_error'))<=0:
            raise ValueError('positive inputs required')
    held='length' if variable=='inverse_temperature' else 'theta'
    if len({row[held] for row in rows})!=1:
        raise ValueError('hold the other physical condition fixed for this fit')
    if len(rows)<3: raise ValueError('at least three independently estimated conditions required')
    x=np.array([1/row['theta'] if variable=='inverse_temperature' else np.log(row['length']) for row in rows])
    y=np.log([row['lifetime'] for row in rows])
    se=np.array([row['standard_error']/row['lifetime'] for row in rows])
    design=np.stack((np.ones_like(x),x),1); weighted=design/se[:,None]
    if np.linalg.matrix_rank(weighted)!=2: raise ValueError('rank-deficient condition grid')
    cov=np.linalg.inv(weighted.T@weighted); beta=cov@weighted.T@(y/se)
    residual=y-design@beta
    return {'predictor':variable,'intercept':float(beta[0]),'slope':float(beta[1]),
            'covariance':cov.tolist(),'normal_approx_ci95':np.stack((beta-1.96*np.sqrt(np.diag(cov)),beta+1.96*np.sqrt(np.diag(cov))),1).tolist(),
            'standardized_residuals':(residual/se).tolist(),'chi_square':float(np.sum((residual/se)**2)),
            'degrees_of_freedom':len(rows)-2,'input_rows':rows,
            'certified':False,'assumptions':'Independent lifetime estimates, delta-method log errors; inspect residuals and input estimator validity.'}


if __name__=='__main__':
    args=analysis_arguments(); data=json.loads(args.input.read_text())
    save_json(args.output,fit_scaling(data['rows'],variable=data['variable']))
