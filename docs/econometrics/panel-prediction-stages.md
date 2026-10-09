# Panel and predictive workflow stages

Scope established before implementation, 7 October 2026 (MARKET-147/148/184).
Existing continuous penalized solvers, MG/PMG/CCE, forward rolling and their
validated options are reused. These stages do not imply vendor parity.

1. MARKET-147: scalar-response native PLS1, fixed components or training-fold-only
   CV; saved original-unit predictions, no coefficient inference. Independently
   compare held-out predictions, full-rank OLS limit and component CV. Numeric,
   unweighted, in-memory CPU float64 only. Separate future stages cover binomial
   and Poisson penalized losses, weights, categorical encoding, penalty factors
   and forced controls; none are silently routed through Gaussian prediction.
2. MARKET-148a: `xtreg(model='cre')`, means of actual retained estimation sample,
   robust joint Mundlak Wald test, retained time-invariant regressors, persisted
   means and explicit prediction boundary. Reuse RE variance/covariance engine.
3. MARKET-148b: separate Hausman–Taylor estimator, declared X1/X2/Z1/Z2 roles,
   within pilot, between IV variance components, transformed IV fit and full
   covariance. Balanced/unbalanced panels; no weights, categorical inputs or
   Amemiya–MaCurdy variant in this stage. Refuse rank/identification failures.
4. MARKET-184a: reverse recursive retrospective windows and pooled panel windows
   over integer calendar spans. Preserve physical row windows as the existing
   default. Record full results and failed windows. Reverse recursive fitting
   holds the final date fixed and advances the start; it is retrospective,
   never claimed to be causal forecasting.
5. MARKET-184b: separate forward predictive evaluation, with each training fit
   (including tuning/scaling) isolated from the subsequent evaluation rows.
   Persist each fitted predictive state, row boundaries and held-out errors.
   Do not invent coefficient intervals for prediction-only targets.

Each implementation stage requires independent numerical/error tests, generated
registry documentation, a runnable example, saved/read-back results and separate
source, frozen and installed synthetic Mac proof. Dataset and device boundaries
are explicit; no implicit collection, MPS/CUDA execution or silent budget trim.
