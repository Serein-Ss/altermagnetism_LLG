# Nishino reduced LLG reproduction

_Run 20260910_8df33ac_d30ae957; phase primary; status pass_

---

## 📋 Scope and outcome

This is Fig. 1 case A/B only, not a five-system certificate. Passed numerical
conditions: 24/24. Selected step factor: None.
The primary method is the explicit midpoint of Appendix B5/B6 with a separately
declared final projection; raw predictor/final norm errors are reported.[^1]

## 📊 Acceptance

See [report.json](report.json) for every temperature, method, time-step,
stationarity CI, CDF effect size, Holm-adjusted p-value and paired convergence CI.
The mean criterion is abs(mean-exact)+95% CI half-width <= 0.01. CI samples are
independent magnetic moments, not correlated frames. Non-rejection of a CDF test
does not prove equality; the additional effect-size limit is required.
Temperature grid and repeat counts are project validation extensions.

## 🔐 Provenance and limits

Config SHA-256: d30ae957eb3141582ddca185c64b4f924f2b04d95a616a62d717f4ff299b4357.
Code commit: 8df33aca80fa0b6dfe943804400398dad5d31470; working-code SHA-256: 6fd7226f250d0baf38d7c65b4186d41376b6ef4cee88cca9aa6f2dfb23ee4b8d.
The manifest links immutable raw files, derived results, frozen config and plots.
No failed trajectories are discarded. A failed gate prohibits primary production.
R2–R5 and training-dataset generation remain blocked.
The 2018 erratum repairs Fokker–Planck/precession and definition terms;
Fig. 1 settings and the Appendix B integration target are unchanged.[^2]

## 🔗 References

[^1]: Nishino and Miyashita (2015), Phys. Rev. B 91, 134411. https://doi.org/10.1103/PhysRevB.91.134411
[^2]: Nishino and Miyashita (2018), Erratum, Phys. Rev. B 97, 019904. https://doi.org/10.1103/PhysRevB.97.019904
