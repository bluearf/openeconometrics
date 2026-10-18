# Instructor companion · Lab 02

## Teaching sequence

Start with two audiences: a shop accountant who needs the till total and a store manager who wants a basket near the middle of the line. Require students to choose a target question before computing a statistic, and to say that every center here is per basket, not per customer.

Before running, use the pause question in Section 4: give the mean and median and ask where the geometric mean will fall. Connect each formula to the property it preserves (total, rank, ratio) before interpreting the output, then treat the largest basket as a sensitivity check rather than a deletion.

## Worked exercise answers

**1.** Total observed spending is $240 \times 32.4390 = 7{,}785.36$ currency units. Removing the 320.00 basket leaves $7{,}465.36$, and $7{,}465.36/239 = 31.2358$, the sensitivity mean reported by the lab.

**2.** The 120th and 121st ordered values are 24.6616 and 24.7198, and their average is the median 24.6907. They differ by only 0.058 units because baskets are densely packed near the middle of the distribution (the histogram bin containing the median holds 84 baskets), so neighboring order statistics are close.

**3.** With 239 baskets the median is the 120th ordered value, 24.6616, a fall of 0.0291 units. The mean falls by 1.2032, about 41 times as much. Removing the top basket changes only which position is the middle, by half a step; the mean loses the full magnitude of a value 288.76 units above the other baskets' mean.

**4.** The mean becomes 34.4390 and the median 26.6907, each exactly 2 higher, because adding a constant shifts every value and every order statistic by that constant. The geometric mean becomes 27.1392, an increase of 2.553, more than 2. A fixed fee is a large proportional increase for small baskets (the 2.80 basket rises by about 72%) and a small one for large baskets; the geometric mean averages proportional changes, so it rises by more than the fee. In general $g(x+c)\ge g(x)+c$ for positive data.

**5.** 151 of the 240 baskets (62.9%) spend less than the mean of 32.44, and 89 spend more. Because the mean lies above the median, all 120 baskets in the lower half are below it, plus the 31 baskets between the median and the mean. A few large baskets raise the mean above most individual baskets.

**6.** Reply: revenue per basket must add up to the till total, which only the arithmetic mean does. The 320.00 basket is valid revenue. Using the median would give $240 \times 24.6907 = 5{,}925.77$, understating actual revenue by 1,859.59 currency units. The median remains a good description of a middle basket. Example rewrites: for the accountant, "Mean spending is 32.44 currency units per basket across all 240 baskets, which multiplied by 240 gives total revenue"; for the manager, "Half of the 240 baskets spent 24.69 currency units or less."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/02-center/shop_spending.xlsx) for the published analysis. The [original generator](generators/02-center.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/02-center/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/02-center/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: median matches order statistics, payroll identity, geometric below arithmetic, tail removal changes mean upward gap. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Calling the median the “correct” mean; deleting the valid largest basket; treating exponentiated mean logs as arithmetic spending; confusing sensitivity analysis with an alternative primary sample.
