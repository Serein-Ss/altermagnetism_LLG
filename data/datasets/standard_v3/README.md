# Standard V3 preparation status

This directory is intentionally separate from `standard_v2`. The registry and
candidate protocol use JSON syntax, which is a valid YAML 1.2 subset and can be
read with Python's standard `json` module.

No V3 system is currently labelled `production_certified`. In particular,
Gomonay production generation is blocked until the equilibrium initial-state
pool, integration time step, save cadence, fixed duration, size convergence and
five drive points have been frozen from preregistered pilots. Mn2Au is blocked
earlier because its complete parameter and neighbor-shell audit is absent.

Raw pilot shards and raw production shards are not allowed to certify
themselves. A later, separate labelling and certification stage must retain
censored paths and populate steady-state/event metadata before promotion.
