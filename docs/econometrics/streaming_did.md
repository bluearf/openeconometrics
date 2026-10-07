# Dataset difference in differences and event studies

`didregress` and `eventstudy` replay the complete retained sample. The outcome,
treatment/event indicators and controls undergo the same global weighted two-way
fixed-effect projection; a bounded TSQR factor estimates the slopes. These are
the existing TWFE estimands, including their staggered-adoption limitations.
There is no subsample fit and no average of separate batch regressions.

Group identities, sorted time identities, adoption status, connected components
and cluster nesting live on owned SQLite storage. Projection vectors use owned
sequential float64 files and bounded reads rather than full in-memory vectors
or memory maps. The native symmetric Kaczmarz/conjugate-gradient projection is
shared with Dataset `reghdfe`. Every complete source pass verifies projected
values and retained physical positions. All temporary files close on failure.

Classical, HC1, default group-clustered robust and one/two-way cluster covariance
retain the entire source. Fixed effects nested inside **any** declared cluster
are excluded from the small-sample parameter count; when both dimensions nest,
one absorbed constant remains. This also repairs the former dense DID rule that
inspected only the first cluster. Frequency/analytic/probability weights follow
the native linear conventions, including global weight normalization. Singleton
observations are kept, as in dense DID/event study.

DID verifies every treatment value and group-level absorbing adoption. Parallel
trends and anticipation use full-source augmented TWFE fits when a common start
and never-treated controls identify those tests. Switching/nonabsorbing adoption
still permits the TWFE regression but records that those tests do not apply.
Event-study verifies constant first-treatment time including never-treated status,
bins optional event-window endpoints, preserves the zero reference row and
reports the joint pretrend test and average post-effect covariance.

Integer time identities retain signed int64 precision; floating event times must
be exactly integral within float64's exact period range. Datetime event-study
time uses global sorted consecutive ranks, matching the existing contract and
recording that calendar gaps are not inferred. The model/output width and cohort
metadata have explicit budgets. Input blocks shrink to the captured workspace
budget. Projection scratch requires O(N*K) disk space and repeated scans; this
is a bounded-RAM implementation, not a promise that arbitrary datasets are fast.

Results retain global counts, hashes, diagnostic/resource plans, at most 400
physical reporting rows and JSON/LaTeX export. Tests compare dense native fits,
independent explicit dummy-variable projection and multiway covariance, declared
categories, missing/frequency samples, shuffled dates, bounded readers, meta
default-device restoration and source/resource failure cleanup. No external
estimator is imported by the runtime adapter.
