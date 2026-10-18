# Instructor companion · Lab 17

## Teaching sequence

Begin with the three schedules and ask who bears each cost: producers bear `private_mc`, third parties bear `external_mc`. Have students build social marginal cost by adding the two columns and answer the "Pause before running" question (marginal benefit should be compared with social marginal cost) before they find either crossing.

After running, keep three numbers apart: the marginal damage that sets the tax (10 per unit), the welfare gain from correcting output (50 a day) and the total damage at the private output (500 a day). Exercises 3 and 4 target the two most common errors, setting the tax at total damage and subtracting tax revenue as a second cost.

## Worked exercise answers

**1.** With damage 16, $MSC=26+0.5Q$ and $60-0.5Q=26+0.5Q$ gives an efficient output of 34 units a day. There, marginal benefit is 43 and private marginal cost 27, so the corrective tax is $43-27=16$. Welfare is $W(Q)=34Q-\tfrac12Q^2$, which is 450 at the private output of 50 and 578 at 34, a gain of 128 currency/day ($=\tfrac12\times16\times16$).

**2.** With $t=10$ and $Q=40$, buyers pay $60-0.5\times40=40$ and producers keep $10+0.5\times40=30$. Consumer surplus is $\tfrac12(60-40)\times40=400$, producer surplus $\tfrac12(30-10)\times40=400$, tax revenue $10\times40=400$ and total damage $10\times40=400$, all in currency/day. Welfare is $CS+PS+\text{revenue}-\text{damage}=400+400+400-400=800$. Revenue enters with a plus sign because the government (and those it spends on) receives it.

**3.** The tax is a per-unit price in currency/unit, while 500 is a total in currency/day; the two have different units. On these schedules $Q_t=50-t$, so a tax of 500 would require negative output: no unit has a benefit-cost gap that large, so trade stops at $Q=0$. Welfare would be 0, which is 800 below the efficient outcome and 750 below even the unregulated market.

**4.** That calculation gives $400+400-400-400=0$ currency/day. It subtracts the tax revenue as if it were a resource cost, but revenue is a transfer from market participants to the government; the true social cost of the externality, the damage of 400, is already subtracted. The correct welfare is 800.

**5.** With $MED=0.25Q$, $MSC=10+0.75Q$ and $60-0.5Q=10+0.75Q$ gives $Q^S=40$. The corrective tax is marginal damage at that output, $0.25\times40=10$. Tax revenue is $10\times40=400$ but total damage is $\int_0^{40}0.25q\,dq=200$. (Welfare is $50Q-0.625Q^2$: 1,000 at 40 units versus 937.5 at the private 50.) Revenue equaled damage in the base case only because marginal damage was constant; a corrective tax is set by marginal damage and is not designed to equal, or compensate, total damage.

**6.** For example: "In this stipulated market, a tax of 10 per unit, equal to marginal external damage, lowers output from 50 to 40 units a day and raises the buyer price from 35 to 40. The ten units removed each cost society more than they were worth, so welfare net of damage rises from 750 to 800 currency units a day; the tax payments are a transfer, not a loss to society."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/17-external-cost/external_cost.xlsx) for the published analysis. The [original generator](generators/17-external-cost.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/17-external-cost/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/17-external-cost/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: social foc, corrective tax implements optimum. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A real corrective tax requires evidence about marginal damage and responses. Total damage at the original output is not the optimal unit tax.
