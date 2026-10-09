"""Fixed-design Gaussian calibration of OLS-residual CUSUM and squares."""

from numbers import Integral

import pandas as pd
import torch

from openecon.analysis import _coerce_frame, _hash_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.econometrics.unitroot.stability import _model, _notes
from openecon.econometrics.unitroot.common import column_names, check_flag, period_label
from openecon.resources import plan_workspace


@resident_cpu
def ols_cusum(data, y, x, *, time=None, intercept=True, replications=999,
              seed=1729, alpha=0.05, errors="iid_gaussian", max_work=100_000_000):
    """OLS residual CUSUM/CUSUMSQ, conditional fixed-X Gaussian Monte Carlo.

    Every draw projects Gaussian errors on exactly the saved full design and
    rescales its own residual sum of squares. No Brownian-bridge critical
    values or recursive-residual bounds are substituted. This is an iid
    Gaussian fixed-design test, not robust/HAC or a selective break test.
    """
    if (isinstance(replications, bool) or not isinstance(replications, Integral)
            or not 99 <= replications <= 9999 or isinstance(seed, bool)
            or not isinstance(seed, Integral) or not 0 <= seed < 2**63
            or isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1
            or errors != "iid_gaussian" or isinstance(max_work, bool)
            or not isinstance(max_work, Integral) or max_work < 1):
        raise AnalysisError("invalid_option", "Use 99..9999 draws, a bounded seed, alpha and iid_gaussian.")
    replications, seed = int(replications), int(seed)
    requested = column_names(x, "x")
    check_flag(intercept, "intercept")
    if len(requested) + int(intercept) > 32:
        raise AnalysisError("work_budget", "At most 32 requested regressors including the intercept.")
    # Admit before any numeric tensor, QR or draw allocation. Mapping inputs
    # become resident tables; wide unused DataFrame columns are not copied.
    if isinstance(data, (list, tuple, pd.DataFrame)) and len(data) > 2048:
        raise AnalysisError("work_budget", "At most 2048 supplied rows.")
    if isinstance(data, dict) and any(hasattr(v, "__len__") and len(v) > 2048 for v in data.values()):
        raise AnalysisError("work_budget", "At most 2048 supplied rows.")
    frame = _coerce_frame(data)
    if len(frame) > 2048:
        raise AnalysisError("work_budget", "At most 2048 supplied rows.")
    plan = plan_workspace("conditional OLS-residual CUSUM", {
        "input, design and QR factors": 64*len(frame)*(len(requested)+2),
        "Gaussian batch, projected errors and two cumulative paths": 64*32*len(frame),
        "saved null maxima and quantile workspace": 64*replications,
    }).record()
    model = _model(frame, y, requested, time, intercept, "OLS-residual CUSUM")
    n, k = model.n, model.k
    if n > 2048 or k > 32 or n <= k + 2:
        raise AnalysisError("work_budget", "Use k+3..2048 rows and at most 32 independent regressors.")
    work = n*k*k + replications*n*(2*k+12)
    if work > max_work:
        raise AnalysisError("work_budget", "Conditional projection simulation exceeds max_work.")
    q, _ = torch.linalg.qr(model.x, mode="reduced")
    residual = model.y - q @ (q.T @ model.y)
    fraction = torch.arange(1, n+1, dtype=torch.float64)/n

    def paths(e):
        ssr = e.square().sum(-1)
        if not torch.isfinite(e).all() or not torch.isfinite(ssr).all() or (ssr <= 0).any():
            raise AnalysisError("degenerate_residuals", "Every observed and simulated RSS must be positive.")
        return e.cumsum(-1)/ssr.sqrt()[..., None], e.square().cumsum(-1)/ssr[..., None]-fraction

    first, second = paths(residual)
    observed = torch.stack([first.abs().max(), second.abs().max()])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    maxima = []
    for start in range(0, replications, 32):
        noise = torch.randn(min(32, replications-start), n, dtype=torch.float64, generator=generator)
        e = noise - (noise @ q) @ q.T
        a, b = paths(e)
        maxima.append(torch.stack([a.abs().amax(1), b.abs().amax(1)], dim=1))
    draws = torch.cat(maxima)
    cutoffs = torch.quantile(draws, 1-alpha, dim=0, interpolation="higher")
    p = (1+(draws >= observed).sum(0)).to(torch.float64)/(replications+1)
    names = ["ols_cusum", "ols_cusumsq"]
    output = TableSet(
        {"tests": table([dict(test=name, statistic=float(observed[i]), p_value=float(p[i]),
                              critical=float(cutoffs[i]), reject=bool(p[i] <= alpha))
                         for i, name in enumerate(names)]),
         "path": table({"row": list(range(n)), "period": [period_label(model.labels, i) for i in range(n)], "fraction": fraction.tolist(),
                        "cusum": first.tolist(), "cusumsq": second.tolist()}),
         "null_maxima": table(draws.tolist(), columns=names)},
        replications=replications, failed_draws=0, seed=seed, alpha=alpha, nobs=n,
        rank=k, planned_work=work, resource_plan=plan,
        calibration="conditional fixed-X iid Gaussian projection; own-RSS studentization; complete draw denominator; plus-one upper-tail p",
        assumptions="Independent Gaussian errors of common variance, independent of prespecified fixed exogenous regressors",
        curve_family="each whole time path separately, not simultaneous across both diagnostics",
        sample_order="time-sorted complete numeric rows, no imputed rows", notes=_notes(model),
        error_model=errors, time=time, outcome=y, regressors=model.terms,
        supplied_data_hash=_hash_frame(frame.loc[:, [y, *requested, *([time] if time else [])]]),
        precision="float64", device="cpu", stata_parity_validated=False,
    )
    return saved_summary(output)
