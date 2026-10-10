# Instructor companion · Lab 01

The student handout is an original synthetic study-design exercise. Lead with the agency's decision before revealing the known generating effect. Ask students to state what the raw difference compares. The key distinction is between an arithmetically valid observed contrast and a justified counterfactual comparison.

## Worked exercise answers

1. Descriptive: how do observed participant and nonparticipant wages differ at the follow-up date? Predictive: which applicants are likely to have low future wages? Causal: how would follow-up wages change under participation versus nonparticipation? Accept alternative clearly specified populations and outcomes; require the question and proposed use to agree.
2. The 75/25 skill target gives $0.75(3.9164208881)+0.25(4.3922403634)=4.035375757$ currency/hour. The target changes the weights, not the cell outcomes. Do not treat any weighting choice as universally correct.
3. Participants have 66 higher-skill workers among 306, giving 0.215686. Nonparticipants have 234 among 294, giving 0.795918. Substituting these shares and the four cell means reconstructs 19.878187691 and 26.281028482 respectively.
4. A cell with no nonparticipants provides no observed within-stratum counterfactual comparison. A standardized target placing positive weight there is unsupported without additional modeling/extrapolation assumptions. Zero counts are not zero outcomes.
5. Prior earnings is a pre-participation candidate; certification obtained through training is a post-participation variable. Adjusting for a mediator can remove part of the effect being asked about. Timing alone does not establish a sufficient adjustment set.
6. Accept a headline such as “Synthetic wage comparison reverses after aligning prior-skill composition.” Require explicit synthetic labeling and descriptive wording. The sample does not evaluate a real agency.

## Scientific and reproduction notes

The generating effect is 4, while the realized standardized contrast is 4.154330626. Raw gap: −6.402840792. No estimator is fitted, so `models` is intentionally empty. All rows are complete and all four cells are populated. Native outputs are table, table, plot. The script independently checks the weighted-mean and standardization identities and explicit JSON export readback. [Full reference](../labs/01-economic-question/reference.json) contains the executed cell means and checks.

Complete-file execution leaves `lab_result`, requires no `__file__`, and writes no files by default. For an explicit analysis export, load the source and call `run_lab(output_dir="docs/teaching/labs/01-economic-question")` from the repository root. Use a clean namespace when changing the analysis specification.

**Prepared input:** [training_and_wages.xlsx](../labs/01-economic-question/training_and_wages.xlsx). Use the stored observations for the published analysis. The [original generator](generators/01-economic-question.py) supports new dataset editions; update results and answers when changing observations.
