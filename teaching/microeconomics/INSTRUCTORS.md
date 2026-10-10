# Instructor guide · Microeconomics with OpenEconometrics

Distribute the separate student edition to the class. This guide and the chapter companions contain worked answers and reproduction evidence.

## Chapter guides

| Lab | Instructor companion |
|---|---|
| 01 | [Scarcity and Increasing Opportunity Cost](instructors/01-production-frontier.md) |
| 02 | [Comparative Advantage and Trading Prices](instructors/02-comparative-advantage.md) |
| 03 | [Market Equilibrium and a Demand Shift](instructors/03-market-equilibrium.md) |
| 04 | [Point Elasticity and Total Revenue](instructors/04-elasticity-revenue.md) |
| 05 | [A Budget Line and Cobb–Douglas Choice](instructors/05-consumer-choice.md) |
| 06 | [Income and Price Comparative Statics](instructors/06-demand-comparative-statics.md) |
| 07 | [Separating Substitution and Income Effects](instructors/07-substitution-income.md) |
| 08 | [Affordability and Revealed Preference](instructors/08-revealed-preference.md) |
| 09 | [Marginal Product and Labor Demand](instructors/09-production-input.md) |
| 10 | [Total, Average and Marginal Cost](instructors/10-cost-curves.md) |
| 11 | [Cost Minimization with Two Inputs](instructors/11-conditional-inputs.md) |
| 12 | [Competitive Supply and Shutdown](instructors/12-competitive-supply.md) |
| 13 | [Monopoly Output, Markup and Welfare](instructors/13-monopoly-welfare.md) |
| 14 | [A Binding Price Ceiling and Rationing](instructors/14-price-ceiling.md) |
| 15 | [A Unit Tax: Incidence and Deadweight Loss](instructors/15-tax-incidence.md) |
| 16 | [A Producer Subsidy and Overproduction](instructors/16-subsidy-welfare.md) |
| 17 | [An External Cost and a Corrective Tax](instructors/17-external-cost.md) |
| 18 | [Vertical Summation for a Public Good](instructors/18-public-good.md) |
| 19 | [Cournot Competition and the Number of Firms](instructors/19-cournot-competition.md) |
| 20 | [From a Demand Model to an Estimated Elasticity](instructors/20-estimated-demand.md) |

## Reproduce and inspect the evidence

Every workbook has a flat `Data` sheet with original rows, order and missing cells, and a `Dictionary` sheet with definitions and units. Never replace a blank measurement with zero merely to simplify import. Original mechanisms in `instructors/generators/` expose `make_data()` and the seed; these files are excluded from the student edition. The ordinary student workflow reads prepared observations and never regenerates them. Resampling lessons derive their draws from the supplied observations with stated seeds and replication counts.

Run focused verification with the locked application environment and numerical threads capped:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync python scripts/verify_course_series.py --course microeconomics
```

The verifier compares each workbook with its original mechanism at `1e-13`, preserving identifiers, column order, row order and the missing mask. Fresh Python execution and exact native worker execution are compared with the complete saved reference at `1e-8`. Full native tables are compared, not only selected headline numbers. Model results, where present, preserve covariance, sample positions and declared inference. Explicit export and reopened application history are checked separately. Independent development references use SciPy and statsmodels only in verification; the delivered analyses use OpenEconometrics, Python, pandas and Torch.

Build the offline editions with `scripts/build_teaching_handouts.py --course microeconomics` in the documented teaching renderer environment. Print the student's complete edition to `output/teaching/microeconomics/student/microeconomics-labs.pdf` and the instructor edition to `output/teaching/microeconomics/instructor/instructor-guide.pdf`. Then run `scripts/package_teaching_handouts.py --course microeconomics`. It checks local links, source-identical Excel inputs, audience separation, portable PDF links and ZIP readback.

The renderer reads saved reference values to draw figures; it does not refit an analysis. Regenerate all dependent material together when intentionally changing a mechanism or method. Numerical agreement on these examples is distinct from empirical validity, installed acceptance and public-release evidence.
