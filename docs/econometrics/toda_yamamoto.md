# Toda–Yamamoto predictive noncausality

`oe.tycausality` fits an augmented **levels** VAR and returns modified Wald
tests plus the fitted system's coefficient table. Estimation uses the native
CPU float64 system QR; pandas supplies tabular input and output. No external
statistics estimator is called.

```python
tests = oe.tycausality(
    data=df, y=["income", "consumption", "investment"], time="year",
    lags=2, dmax=1,
)
display(tests["tests"])
display(tests["coefficients"])
print(tests.to_latex(notes=tests.attrs["notes"]))

# A joint directional hypothesis in one equation:
joint = oe.tycausality(
    data=df, y=["income", "consumption", "investment"], lags=2, dmax=1,
    causes=["income", "investment"], effects=["consumption"],
)
```

The method follows the original [Toda and Yamamoto (1995) paper,
equation (8)](https://doi.org/10.1016/0304-4076(94)01616-8). With base order
`k=lags` and an upper integration order `dmax`, all `k+dmax` lags enter every
equation. The hypothesis sets only the first **k** cause coefficients to zero;
the extra lags remain unrestricted. The covariance uses `Sigma = U'U/T`
(original maximum-likelihood divisor), and the reference distribution is
asymptotic chi-square with `k × number_of_causes` degrees of freedom. These
p-values describe predictive Granger noncausality, not structural causal effects.

This implementation accepts declared `dmax` values 0, 1 and 2. Its chi-square
inference requires a valid upper integration order, a correctly specified
base lag order and deterministic terms, and nonsingular iid innovations with
the paper's moment conditions. Choosing an insufficient integration bound or
lag order invalidates that reference. Neither is selected or tested
automatically. An intercept is included by default; a linear trend is optional.
There is no finite-sample size guarantee, HAC correction, bootstrap,
break correction, or asymmetric/Fourier transformation in this procedure.

By default every directed pair is tested, and for at least three variables
each equation also receives the joint exclusion of all its other variables.
An explicit `causes` group is tested jointly in each selected effect equation.
Set `joint=False` for pairwise tests. Explicit cause and effect groups must be
disjoint; with implicit effects the cause variable's own equation is skipped.
Augmentation coefficients are marked in the coefficient table and recorded
separately from each row's actual restrictions in `attrs['tested_terms']`.

Input must be complete. Integer time keys are checked without converting
signed or unsigned integer storage to float64; floating keys must be exact
integers within `2**53-1`. Datetimes must follow a regular calendar frequency,
including monthly or business-day calendars. Without `time`, row order
defines the sequence. No missing row, duplicate time key or interior gap is
silently removed.

The in-memory procedure checks the native 256 MiB lag-design limit, 1,000
system coefficient limit, an estimated 512 MiB numerical workspace limit and
`T × m² <= 2,000,000,000` QR work bound **before** allocating the lag matrix.
The workspace estimate excludes the caller's resident table and Python result
objects; it is not a total-process RAM guarantee. Collinear designs, exact
equations, singular residual covariance, unsupported magnitudes, and singular
restriction covariance produce explicit errors. Restrictions are never
silently replaced with a different reduced-rank hypothesis.

Verification compares the full coefficient table, standard errors,
maximum-likelihood residual covariance, Wald statistics and p-values against
independent NumPy QR and SciPy chi-square tails. A second oracle uses the
paper's projection onto the complement of all nuisance regressors. Tests cover
I(0), I(1) and I(2) fixtures, deterministic specifications, exclusion of the
added lags, multiple-cause groups, scaling/translation invariance, exact large
time keys and regular calendars, saved LaTeX output, CPU context restoration,
input preservation and failure before oversized allocation. These numerical
oracles establish implementation agreement, not Monte Carlo size calibration.
