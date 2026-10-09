# Fixed external bias sensitivity

`oe.bias_sensitivity` takes a table with one row per ordered reduced-form
estimate and separate numeric columns for the estimate, contrast weight,
lower bias and upper bias. Supply the complete joint covariance in that row
order and explicitly declare `restriction="fixed_external"`. Limits, weights
and the scale grid must be specified independently of estimation noise.

For estimated mean vector b, fixed contrast l and bias delta in a scaled box,
the target is l'*(b-delta). The lower bias support is the sum of l_j*lower_j
for positive weights and l_j*upper_j for negative weights; the upper support
reverses those choices. Identification endpoints subtract the upper/lower
support from l'b. These are plug-in identification endpoints, without sampling
confidence coverage by themselves.

The sampling variance is the complete l'*V*l, including every cross covariance.
Subtract/add the Gaussian critical value times its square root to obtain the
union of pointwise intervals over all admissible biases. Under the declared
Gaussian model this union covers the true scalar target with at least the
specified level whenever its fixed bias belongs to the box. With asymptotic
Gaussian inputs this is an asymptotic claim. It is conservative and does not
optimize interval length. The tipping scales report when zero first enters
the identification or confidence interval along this exact scaled box; `None`
means no finite nonnegative scale can reach zero.

The decomposition of reduced-form means into target and bias is the starting
point of [Rambachan and Roth (2023), equation1](https://www.jonathandroth.com/assets/files/HonestParallelTrends_Main.pdf).
Their methods additionally account for restrictions derived from noisy
pretrend estimates. This implementation uses external fixed coordinatewise
limits and does **not** implement their conditional/hybrid procedures,
optimal fixed-length intervals, relative-magnitude/smoothness restrictions or
HonestDiD. A pretrend maximum chosen from the same noisy estimates does not
satisfy this API's fixed-external restriction.

All input terms, original positions/labels, full supplied covariance, exact
support contributions, critical value, identification endpoints, union
intervals and tipping scales are preserved by `causal_design_save/load`.
Covariance must be exactly symmetric and numerically positive semidefinite;
invalid matrices are refused without projection. Mean, support and covariance projections use native compensated sums to preserve small terms across cancellation. Nonzero active variance and tipping ratios cannot silently become deterministic or zero after underflow. A zero variance is allowed
for a deterministic contrast. Nonzero unrepresentable variance/contributions
and overflow are refused. There are at most64 terms and256 increasing
nonnegative scales. Torch float64 CPU handles numerical work; Dataset,
generic weights, missing-row deletion and other devices are unsupported.
Workspace and work budgets cover the complete computation, with no truncation.

Independent tests enumerate all box vertices, preserve off-diagonal covariance,
check Gaussian coverage analytically at vertices/interior points, and verify
tipping scales, singular deterministic contrasts, tiny variance, resource
refusals and exact complete JSON restoration. These bounded method checks do
not enable licensed vendor or whole-product parity.
