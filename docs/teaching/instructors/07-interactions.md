# Instructor companion · Lab 07

Draw two lines before displaying coefficients. First ask what an additive model rules out, then let students substitute group indicators into the interaction equation. The main group coefficient is a gap at the reference experience level, while the interaction is a slope difference. Significance of one slope and nonsignificance of another does not test their difference.

## Worked exercise answers

1. At 20 years, weights are sector = 1 and sector_experience = 5. Gap = 12.552502340, HC3 SE = 0.441092542, 95% CI [11.685773038, 13.419231643]. Include the covariance cross term rather than adding standard errors.
2. Recentered at 20, the group-A intercept becomes 29.592269162 and main group gap 12.552502340. The experience slope 0.788457633 and interaction 0.459389476 are unchanged. All equivalent fitted profiles and contrasts remain unchanged.
3. Coding A as 1 makes B the reference. Its intercept at 15 is 35.905535954 and slope 1.247847110. The A-minus-B main gap is −10.255554959 and interaction −0.459389476. Reparameterization preserves fitted outcomes.
4. Test `full.lincom({"experience_centered": 1, "sector_experience": 1}, constant=-1)`. The difference from 1 is 0.247847110, SE 0.037831067, t = 6.551417339, p = 1.4822e−10. Testing the interaction alone asks whether slopes differ, not whether B's slope equals 1.
5. The additive gap 11.560443307 compresses a changing relationship; interaction gaps at 10/15/25 years are 7.958607578/10.255554959/14.849449722. Do not require the additive coefficient to equal their unweighted arithmetic average.
6. Grade the explicit overlap/extrapolation description. A model equation can generate unsupported predictions without making them observed comparisons.

## Scientific and reproduction notes

The source fits additive, centered-interaction and equivalent uncentered models on 480 identical workers. It checks independent matrix OLS/HC3, contrast variance with cross terms, parameterization identity, group-slope identity and full ResultBundle JSON preservation. HC3 Student-t df = 476 for the interaction model. Native outputs: model, table, plot. [Full reference](../labs/07-interactions/reference.json) includes all three models and gap intervals. Explicit regeneration: `run_lab(output_dir="docs/teaching/labs/07-interactions")`. Default run creates no files; simulated categories carry no empirical group implication.

**Prepared input:** [wage_interactions.xlsx](../labs/07-interactions/wage_interactions.xlsx). Use the stored observations for the published analysis. The [original generator](generators/07-interactions.py) supports new dataset editions; update results and answers when changing observations.
