# Instructor companion · Lab 06

## Teaching sequence

Have students distinguish the batch observational unit from the offer-level trials, and list the four binomial assumptions before writing the formula. Work $P(X=3)$ by hand so the binomial coefficient is seen as a count of arrangements.

Before running, use the pause question in Section 5 (does "at least five" include five?) and verify the support and tail boundary before calculating. Stress that the chart shows the model only; exercise 2 is where students compare it with the batches.

## Worked exercise answers

**1.** $P(X=5)=\binom{10}{5}0.3^5 0.7^5=252\times0.00243\times0.16807=0.1029$ and $P(X>5)=P(X\ge6)=0.0473$. Their sum is $0.1029+0.0473=0.1503=P(X\ge5)$. Reading "at least five" as "more than five" drops the $k=5$ term, which is more than twice the rest of the tail, so the answer falls by about two-thirds, from 0.150 to 0.047.

**2.** Expected batches $500\,P(X=k)$ versus observed counts, for $k=0,\dots,10$: 14.1 vs 13; 60.5 vs 62; 116.7 vs 102; 133.4 vs 134; 100.1 vs 110; 51.5 vs 60; 18.4 vs 12; 4.5 vs 5; 0.7 vs 1; 0.07 vs 1; 0.00 vs 0. The largest gap is at 2 orders (14.7 fewer batches than expected), followed by 4 orders (9.9 more) and 5 orders (8.5 more). The single 9-order batch is rare under the model (0.07 expected) but is one batch. Gaps of this size are what a finite simulation from the model produces; they are not evidence against it.

**3.** With twenty offers the mean is $20\times0.3=6$ orders, the variance $20\times0.3\times0.7=4.2$ orders squared and the standard deviation $\sqrt{4.2}=2.049$ orders. The standard deviation does not double: variance is proportional to $n$, so the standard deviation grows by $\sqrt2=1.414$.

**4.** With $p=0.35$, $P(X\ge5)=0.2485$, up from 0.1503: an increase of 0.098, or about 65% in relative terms. The mean rises only from 3 to 3.5, about 17%. Tail probabilities respond much more strongly to $p$ than the mean does, so an error in $p$ matters most for tail questions.

**5.** The 15% is $P(X\ge5)$, the probability of five *or more* orders; the probability of exactly five is 0.103. A correct version: "If each of ten offers independently leads to an order with probability 0.3, the probability that a batch brings at least five orders (five included) is 0.150."

**6.** Example: a shared weekly advertising shock raises or lowers every offer's chance in the same batch, or different batches go to customer groups with different order probabilities. Either makes counts more spread out than the binomial variance 2.1 allows (if $p$ varies across batches with mean 0.3, $\operatorname{Var}(X)=10\bar p(1-\bar p)+90\operatorname{Var}(p)>2.1$), and both very low and very high counts become more common, typically raising $P(X\ge5)$ above 0.150. Negative dependence, such as limited stock that caps orders, would instead reduce the spread.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/06-discrete-model/ten_offer_orders.xlsx) for the published analysis. The [original generator](generators/06-discrete-model.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/06-discrete-model/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/06-discrete-model/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: pmf sums to one, mean matches np, variance matches npq, counts in support. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Omitting the combinatorial factor; mixing counts and rates; using variance np instead of np(1−p); treating model and observed tail fractions as identical by definition.
