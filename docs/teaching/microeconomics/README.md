# Microeconomics with OpenEconometrics

Twenty English labs develop statistical questions through explanations, equations, complete application code, computed results, figures and student exercises. Each supplies a descriptively named Excel workbook with fixed original synthetic observations and a variable dictionary. Resampling appears only where it is the subject of the lesson.

## Chapters

| Lab | Student handout | Prepared dataset | Runnable analysis |
|---|---|---|---|
| 01 | [Scarcity and Increasing Opportunity Cost](labs/01-production-frontier/README.md) | [Excel](labs/01-production-frontier/production_frontier.xlsx) | [Python](labs/01-production-frontier/lab.py) |
| 02 | [Comparative Advantage and Trading Prices](labs/02-comparative-advantage/README.md) | [Excel](labs/02-comparative-advantage/comparative_advantage.xlsx) | [Python](labs/02-comparative-advantage/lab.py) |
| 03 | [Market Equilibrium and a Demand Shift](labs/03-market-equilibrium/README.md) | [Excel](labs/03-market-equilibrium/market_equilibrium.xlsx) | [Python](labs/03-market-equilibrium/lab.py) |
| 04 | [Point Elasticity and Total Revenue](labs/04-elasticity-revenue/README.md) | [Excel](labs/04-elasticity-revenue/elasticity_revenue.xlsx) | [Python](labs/04-elasticity-revenue/lab.py) |
| 05 | [A Budget Line and Cobb–Douglas Choice](labs/05-consumer-choice/README.md) | [Excel](labs/05-consumer-choice/consumer_choice.xlsx) | [Python](labs/05-consumer-choice/lab.py) |
| 06 | [Income and Price Comparative Statics](labs/06-demand-comparative-statics/README.md) | [Excel](labs/06-demand-comparative-statics/demand_comparative_statics.xlsx) | [Python](labs/06-demand-comparative-statics/lab.py) |
| 07 | [Separating Substitution and Income Effects](labs/07-substitution-income/README.md) | [Excel](labs/07-substitution-income/substitution_income.xlsx) | [Python](labs/07-substitution-income/lab.py) |
| 08 | [Affordability and Revealed Preference](labs/08-revealed-preference/README.md) | [Excel](labs/08-revealed-preference/revealed_preference.xlsx) | [Python](labs/08-revealed-preference/lab.py) |
| 09 | [Marginal Product and Labor Demand](labs/09-production-input/README.md) | [Excel](labs/09-production-input/production_input.xlsx) | [Python](labs/09-production-input/lab.py) |
| 10 | [Total, Average and Marginal Cost](labs/10-cost-curves/README.md) | [Excel](labs/10-cost-curves/cost_curves.xlsx) | [Python](labs/10-cost-curves/lab.py) |
| 11 | [Cost Minimization with Two Inputs](labs/11-conditional-inputs/README.md) | [Excel](labs/11-conditional-inputs/conditional_inputs.xlsx) | [Python](labs/11-conditional-inputs/lab.py) |
| 12 | [Competitive Supply and Shutdown](labs/12-competitive-supply/README.md) | [Excel](labs/12-competitive-supply/competitive_supply.xlsx) | [Python](labs/12-competitive-supply/lab.py) |
| 13 | [Monopoly Output, Markup and Welfare](labs/13-monopoly-welfare/README.md) | [Excel](labs/13-monopoly-welfare/monopoly_welfare.xlsx) | [Python](labs/13-monopoly-welfare/lab.py) |
| 14 | [A Binding Price Ceiling and Rationing](labs/14-price-ceiling/README.md) | [Excel](labs/14-price-ceiling/price_ceiling.xlsx) | [Python](labs/14-price-ceiling/lab.py) |
| 15 | [A Unit Tax: Incidence and Deadweight Loss](labs/15-tax-incidence/README.md) | [Excel](labs/15-tax-incidence/tax_incidence.xlsx) | [Python](labs/15-tax-incidence/lab.py) |
| 16 | [A Producer Subsidy and Overproduction](labs/16-subsidy-welfare/README.md) | [Excel](labs/16-subsidy-welfare/subsidy_welfare.xlsx) | [Python](labs/16-subsidy-welfare/lab.py) |
| 17 | [An External Cost and a Corrective Tax](labs/17-external-cost/README.md) | [Excel](labs/17-external-cost/external_cost.xlsx) | [Python](labs/17-external-cost/lab.py) |
| 18 | [Vertical Summation for a Public Good](labs/18-public-good/README.md) | [Excel](labs/18-public-good/public_good.xlsx) | [Python](labs/18-public-good/lab.py) |
| 19 | [Cournot Competition and the Number of Firms](labs/19-cournot-competition/README.md) | [Excel](labs/19-cournot-competition/cournot_competition.xlsx) | [Python](labs/19-cournot-competition/lab.py) |
| 20 | [From a Demand Model to an Estimated Elasticity](labs/20-estimated-demand/README.md) | [Excel](labs/20-estimated-demand/estimated_demand.xlsx) | [Python](labs/20-estimated-demand/lab.py) |

## Use a chapter

Import the named workbook into OpenEconometrics without renaming it. Open the complete Python file in an empty Python document and run the whole file. The script loads the most recent import with that filename or stem. For ordinary Python, keep the workbook beside its script. An explicit `run_lab(data_path="/path/to/workbook.xlsx")` selects another location. Default execution displays results without writing exports.

Read the handout on GitHub or in the separate offline student edition. Students receive explanations, worked analyses and unanswered exercises. Teaching notes, exercise solutions, original data mechanisms and numerical evidence are indexed separately in [the instructor guide](INSTRUCTORS.md) and excluded from the student archive.

All observations, explanations and exercises are original synthetic teaching material under the repository's Apache-2.0 license. Results describe the supplied simulations. External readings provide conceptual background; their exercises and datasets are not reproduced.
