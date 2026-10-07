"""Predictions after ``mixed``: linear predictions, BLUPs and fitted values with BLUPs.

The best linear unbiased predictions (empirical Bayes means) of the random
effects at the estimated parameters are

    u_c = G Z_c' V_s^-1 r_s,     v_s = g 1' V_s^-1 r_s,     r = y - X b,

computed with the same Woodbury algebra as the likelihood (``lmm_kernels``),
so no N-by-N matrix is formed. They are Stata's ``predict, reffects``; the
fitted values ``X b + Z u + v`` are ``predict, fitted``; ``X b`` is
``predict, xb``. Predictions are made for the rows of ``data`` that pass the
model's missing-data policy; BLUPs need the outcome.
"""

from __future__ import annotations

from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, kernel_call, table
from openecon.econometrics.mixed import common
from openecon.econometrics.mixed.lmm_kernels import CovStructure, MixedLikelihood
from openecon.models import ResultBundle

_KINDS = ("xb", "fitted", "reffects")


def mixed_predict(result: ResultBundle, data: Any, *, kind: str = "xb", batch_rows=None,
                  max_group_rows=100000, max_disk_bytes=1073741824) -> Any:
    """Predictions after ``oe.mixed`` (Stata's ``predict`` after ``mixed``).

    Parameters
        result: the ResultBundle returned by ``oe.mixed``.
        data: the data to predict for (usually the estimation data); it must contain
            every column of the model and follows the model's missing-data policy
            (``missing='drop'`` skips incomplete rows). The outcome may be missing for
            ``kind='xb'``; BLUPs (``'fitted'``, ``'reffects'``) are computed from the
            outcomes of each group's rows in ``data``, which may hold a single group.
        kind: ``'xb'`` the linear prediction of the fixed portion ``X b``
            (``predict, xb``); ``'fitted'`` the fitted values ``X b + Z u`` including
            the BLUPs (``predict, fitted``); ``'reffects'`` the BLUPs of the random
            effects themselves (``predict, reffects``).

    Returns
        ``'xb'``/``'fitted'``: a table with columns ``row`` (0-based position in
        ``data``) and ``xb`` or ``fitted``. ``'reffects'``: a long table with one row
        per group and random effect, columns ``level`` (grouping column), ``group``
        (its label; for a nested level the label of the lowest column), ``effect``
        (``_cons`` or the slope variable) and ``blup``.

    Example
        >>> fit = oe.mixed(data=df, y="y", x=["x"], group="g", random=["x"])
        >>> blups = oe.mixed_predict(fit, df, kind="reffects")
    """
    if not isinstance(result, ResultBundle) or result.spec.estimator != "mixed":
        raise AnalysisError("invalid_result", "mixed_predict needs the result of oe.mixed.")
    if kind not in _KINDS:
        raise AnalysisError("invalid_option", f"kind must be one of {', '.join(_KINDS)}.")
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        if kind == 'xb':
            from openecon.econometrics.postest.prediction import predict
            return predict(result, data, kind='xb', batch_rows=batch_rows)
        from openecon.econometrics.postest.grouped_prediction import group_predict
        return group_predict(result, data, kind=kind, target='posterior', batch_rows=batch_rows,
                             max_group_rows=max_group_rows, max_disk_bytes=max_disk_bytes)
    spec = result.spec
    # The linear prediction needs no outcome (Stata's predict, xb); BLUPs do.
    frame = ModelFrame(spec, data, allow_missing=[spec.outcome] if kind == "xb" else ())
    fixed = [c for c in result.coefficients if c.equation == spec.outcome]
    design = frame.design()
    index = {term: i for i, term in enumerate(design.terms)}
    missing = [c.term for c in fixed if c.term not in index]
    if missing:
        raise AnalysisError("invalid_data", "The data do not reproduce the model's terms "
                            f"({', '.join(missing)}); predict with data like the estimation data.")
    x = design.x[:, [index[c.term] for c in fixed]]
    beta = torch.tensor([c.estimate for c in fixed], dtype=torch.float64)
    xb = x @ beta
    rows = frame.positions
    if kind == "xb":
        return table({"row": rows, "xb": xb.tolist()}, columns=["row", "xb"], kind="xb")
    extra = result.extra
    groups = frame.role("group")
    random = frame.role("random")
    names = extra["random_effects"]["effects"]
    columns = [frame.numeric(name) for name in random]
    if "_cons" in names:
        columns.append(torch.ones(frame.n, dtype=torch.float64))
    z = torch.stack(columns, dim=1)
    structure = CovStructure(extra["covstructure"], z.shape[1])
    levels = common.grouping_codes(frame, groups, minimum=1)
    low_codes, n_low = levels[-1]
    top_of_low = n_top = None
    if len(levels) == 2:
        top_codes, n_top = levels[0]
        top_of_low = common.parent_codes(low_codes, top_codes, n_low)
    # BLUPs at the estimated (b, theta) do not involve the REML term.
    like = kernel_call(MixedLikelihood, x, frame.numeric(spec.outcome), z, low_codes, n_low,
                       structure, top_of_low=top_of_low, n_top=n_top or 0,
                       frequency=frame.weights(), reml=False)
    theta = torch.tensor(extra["theta"], dtype=torch.float64)
    out = kernel_call(like.evaluate, theta, blups=True, beta=beta)
    if kind == "fitted":
        fitted = xb + (z * out.low_blup[low_codes]).sum(dim=1)
        if out.top_blup is not None:
            fitted = fitted + out.top_blup[levels[0][0]]
        return table({"row": rows, "fitted": fitted.tolist()}, columns=["row", "fitted"],
                     kind="fitted")
    records: dict[str, list[Any]] = {"level": [], "group": [], "effect": [], "blup": []}
    if out.top_blup is not None:
        labels = common.level_labels(frame, groups[:1], levels[0][0], levels[0][1])
        records["level"].extend([groups[0]] * len(labels))
        records["group"].extend(labels)
        records["effect"].extend(["_cons"] * len(labels))
        records["blup"].extend(out.top_blup.tolist())
    labels = common.level_labels(frame, groups, low_codes, n_low)
    for position, name in enumerate(names):
        records["level"].extend([groups[-1]] * len(labels))
        records["group"].extend(labels)
        records["effect"].extend([name] * len(labels))
        records["blup"].extend(out.low_blup[:, position].tolist())
    return table(records, columns=["level", "group", "effect", "blup"], kind="reffects")
