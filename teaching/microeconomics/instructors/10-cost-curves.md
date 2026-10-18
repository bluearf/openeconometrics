# Instructor companion · Lab 10

## Teaching sequence

Have students split $C(q)$ into fixed and variable parts and write AFC, AVC, ATC and MC before running anything. Derive $d\,ATC/dq=(MC-ATC)/q$ and use it to explain the crossing at the minimum, then pose the pause question (ATC and MC at 50 units). After running, contrast the continuous minimum (31.62) with the whole-unit minimum (32), and the derivative MC with the discrete step cost.

Use the chart to show marginal cost crossing ATC at its minimum; the marginal cost line is checked against the workbook's cost steps. Note that `fixed_cost` and `mc_at_50` are written into `analyze` as constants rather than read from the workbook.

## Worked exercise answers

**1.** At 20 units, $(MC-ATC)/q=(6-9)/20=-0.15$ currency per unit per extra unit; the workbook gives $ATC(21)-ATC(20)=186.1/21-9=8.862-9=-0.138$. At 50 units, $(12-9)/50=0.06$; the workbook gives $ATC(51)-ATC(50)=462.1/51-9=9.061-9=0.061$. The one-unit changes differ slightly from the derivatives because ATC is curved, but the signs and sizes agree: ATC is falling at 20 and rising at 50.

**2.** $C(51)-C(50)=462.1-450.0=12.1$, and the step into 50 units is $450.0-438.1=11.9$. Their average is 12.0, equal to the derivative. For a quadratic cost the backward step is $MC(q)-0.1$ and the forward step $MC(q)+0.1$, so the central average $[C(q+1)-C(q-1)]/2$ equals $2+0.2q$ exactly.

**3.** $ATC(31)=258.1/31=8.3258$ and $ATC(33)=274.9/33=8.3303$, both above $ATC(32)=8.325$. ATC is lowest among whole units at 32 because 32 is nearer to 31.62 than 31 is (0.38 versus 0.62 units) and ATC is very flat and close to symmetric near its minimum. The derivative condition itself holds only at 31.62, which is not a whole unit.

**4.** With fixed cost 160, $-160/q^2+0.1=0$ gives $q^*=\sqrt{1600}=40$ units. Then $ATC=160/40+2+0.1\times40=4+2+4=10$ and $MC=2+0.2\times40=10$. The marginal cost curve does not move, because the fixed cost does not enter $dC/dq$; a larger fixed cost moves the ATC minimum to the right along the same MC line.

**5.** $AVC=2+0.1q$ rises at every output, so it is lowest at the smallest output: 2.1 at 1 unit on the schedule, approaching 2 as output approaches zero. There $MC=2+0.2q$ also approaches 2. Applying the Section 3 argument, $d\,AVC/dq=(MC-AVC)/q=0.1q/q=0.1>0$: MC is above AVC at every positive output, so AVC never falls, and the two meet only in the limit at zero output.

**6.** For example: "At 50 units, the 50th unit adds about 12 currency to daily cost, while the average cost of all 50 units is 9 per unit (total cost 450). Because marginal cost exceeds average cost, producing more would raise the average."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/10-cost-curves/cost_curves.xlsx) for the published analysis. The [original generator](generators/10-cost-curves.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/10-cost-curves/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/10-cost-curves/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: mc crosses atc at minimum, fixed cost per unit falls. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Cost curves describe a specified technology and input-price environment. They do not determine market demand or prove that a firm can sell output at a profitable price.
