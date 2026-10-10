# Instructor notes · Lab 14

The [student handout](../labs/14-randomized-program/index.md) keeps randomized assignment central. The oracle potential-participation columns are for teaching only and must never be included as empirical adjustment covariates.

## Worked answers

1. Offer Z is randomized; receipt D is behavior. ITT compares by Z, including nonparticipants and crossover participants, preserving assignment rather than conditioning on a post-assignment choice.
2. Earnings difference 168.363946-149.961048=18.402898 currency units. Uptake difference .713333-.133333=.58, or 58 percentage points.
3. OLS fitted group means equal observed group means. HC2 equals sqrt(s1²/n1+s0²/n0)=2.696077 in this binary intercept design. The reported t(598) interval [13.107968,23.697828] is a declared regression convention, not exact randomization inference.
4. Baseline difference .752123 points favors offered applicants. Adjustment gives 16.714363, SE 2.097627, CI [12.594739,20.833987]. It reduces estimated imbalance contribution and SE here; do not claim a universal improvement or exact finite-sample unbiasedness of this additive adjustment.
5. Ability raises uptake and earnings, including through baseline. Raw participation comparison 37.767047 exceeds the known 30-unit receipt effect. Random offer does not randomize actual receipt.
6. Oracle .543333333 × 30 = 16.3. Random assignment samples different people into the two observed groups; the realized 18.402898 estimate and .58 uptake difference fluctuate around their corresponding targets.
7. Differential attrition, post-assignment selection, spillovers, misrecorded assignment or inconsistent treatment versions can undermine the simple interpretation.
8. Include N=600, balanced assignment, ITT estimate/interval, observed uptake in both groups and synthetic scope. Separate receipt effects and participation associations.

## Reproduction and verification

`run_lab(output_dir="lab14-review")` explicitly exports the complete [reference](../labs/14-randomized-program/reference.json) and table. Default execution writes no files. Independent two-group means and Neyman variance reproduce the unadjusted point estimate and HC2 SE. Independent matrix OLS and HC2 sandwiches check all three fitted models. The source verifies exactly 300 offers, monotone potential uptake, complete follow-up, identical sample positions and full JSON roundtrip. Initial coefficient/covariance errors were below $2\times10^{-13}$.

Native displays are two `OLSResult` objects, an assignment-summary `DataFrame`, and `Latex`. Distinguish noncompliance dilution, the all-sample participation-based association and per-protocol selection that drops offered decliners. None invalidates the original ITT comparison under this simulated complete-follow-up design.

**Prepared input:** [training_program.xlsx](../labs/14-randomized-program/training_program.xlsx). Use the stored observations for the published analysis. The [original generator](generators/14-randomized-program.py) supports new dataset editions; update results and answers when changing observations. The two potential-uptake columns are simulation oracles, not observed empirical controls. All fitted models use the stated observed variables.
