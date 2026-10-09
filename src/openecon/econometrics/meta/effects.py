"""Summary-data effect sizes on explicitly declared analysis scales."""
from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from .common import SOURCES, critical, finite, level_check, sample

ROLES = {"MD": ("m1", "sd1", "n1", "m2", "sd2", "n2"),
         "SMD": ("m1", "sd1", "n1", "m2", "sd2", "n2"),
         "OR": ("a", "b", "c", "d"), "RR": ("a", "b", "c", "d"),
         "ZCOR": ("r", "n")}


@torch.no_grad()
def meta_effectsize(*, data, measure: str, columns: dict[str, str], study: str | None = None,
                    zero: str = "reject", level: float = .95, dependence: str = "independent"):
    """Prepare independent-study MD, Hedges g (SMD/LS), log OR/RR or Fisher z.

    ``columns`` maps the roles in the measure's definition to numeric columns.
    Binary roles a/b are events/non-events in group 1, c/d in group 2.
    zero='add_half' adds .5 to all four cells only in studies containing zeros.
    Double-zero events or non-events are uninformative and rejected. SMD uses
    exact gamma-ratio Hedges correction and the Hedges--Olkin LS variance.
    Returns every study, yi/vi/se, analysis-scale CI, back-transformed estimate
    and endpoints, correction flags and an input hash; never drops a study.
    Complete in-memory independent summaries only, CPU float64, <=2000 studies.
    """
    if not isinstance(measure, str) or measure not in ROLES or not isinstance(columns, dict) or set(columns) != set(ROLES.get(measure, ())):
        raise AnalysisError("invalid_spec", "Use MD/SMD (m1,sd1,n1,m2,sd2,n2), OR/RR (a,b,c,d), or ZCOR (r,n) with exactly those column roles.")
    if zero not in ("reject", "add_half") or (measure not in ("OR", "RR") and zero != "reject"):
        raise AnalysisError("invalid_zero_policy", "zero='add_half' is available only for binary measures; otherwise use 'reject'.")
    level = level_check(level)
    _, values, ids, digest = sample(data, [columns[r] for r in ROLES[measure]], study, dependence=dependence)
    v = {r: values[columns[r]] for r in ROLES[measure]}
    corrected = torch.zeros(len(ids), dtype=torch.bool, device="cpu")
    if measure in ("MD", "SMD"):
        for r in ("n1", "n2"):
            if bool(((v[r] < 2) | (v[r] != v[r].round()) | (v[r] > 2**53-1)).any()):
                raise AnalysisError("invalid_counts", "Group sample sizes must be exact integers >=2 within the float64 integer range.")
        if bool(((v["sd1"] <= 0) | (v["sd2"] <= 0)).any()):
            raise AnalysisError("invalid_sd", "Group standard deviations must be positive.")
        yi = v["m1"]-v["m2"]
        vi = v["sd1"].square()/v["n1"]+v["sd2"].square()/v["n2"]
        if measure == "SMD":
            df = v["n1"]+v["n2"]-2
            if bool((df > 1e7).any()):
                raise AnalysisError("resource_limit", "Exact gamma-ratio SMD preparation is bounded to combined group df <=1e7.")
            pooled = ((v["n1"]-1)*v["sd1"].square()+(v["n2"]-1)*v["sd2"].square())/df
            correction = (torch.lgamma(df/2)-torch.lgamma((df-1)/2)-.5*(df/2).log()).exp()
            yi = correction*yi/pooled.sqrt()
            vi = 1/v["n1"]+1/v["n2"]+yi.square()/(2*(v["n1"]+v["n2"]))
    elif measure in ("OR", "RR"):
        cells = torch.stack([v[r] for r in ("a", "b", "c", "d")], dim=1)
        if bool(((cells < 0) | (cells != cells.round()) | (cells > 2**53-1)).any()):
            raise AnalysisError("invalid_counts", "Binary cells must be nonnegative exact integer counts.")
        if bool((((cells[:, 0]+cells[:, 2]) == 0) | ((cells[:, 1]+cells[:, 3]) == 0) | (cells[:, :2].sum(1) == 0) | (cells[:, 2:].sum(1) == 0)).any()):
            raise AnalysisError("uninformative_study", "Empty groups, double-zero events or double-zero non-events are unsupported; no study is silently dropped.")
        corrected = (cells == 0).any(1)
        if bool(corrected.any()) and zero == "reject":
            raise AnalysisError("zero_cells", "Zero cells require explicit zero='add_half'.")
        cells = cells+.5*corrected[:, None]
        a, b, c, d = cells.unbind(1)
        if measure == "OR":
            yi, vi = a.log()+d.log()-b.log()-c.log(), 1/a+1/b+1/c+1/d
        else:
            yi, vi = a.log()-(a+b).log()-c.log()+(c+d).log(), 1/a-1/(a+b)+1/c-1/(c+d)
    else:
        if bool(((v["r"].abs() >= 1) | (v["n"] <= 3) | (v["n"] != v["n"].round()) | (v["n"] > 2**53-1)).any()):
            raise AnalysisError("invalid_correlation", "Fisher z requires |r|<1 and exact integer n>3.")
        yi, vi = torch.atanh(v["r"]), 1/(v["n"]-3)
    finite(torch.stack([yi, vi]), "effects/variances")
    if bool((vi <= 0).any()):
        raise AnalysisError("invalid_variance", "Every computed sampling variance must be positive.")
    se = vi.sqrt()
    c = critical(level, "z", None)
    low, high = yi-c*se, yi+c*se
    transform = torch.exp if measure in ("OR", "RR") else torch.tanh if measure == "ZCOR" else lambda x: x
    back = torch.stack([transform(yi), transform(low), transform(high)])
    finite(back, "back-transformed intervals")
    return table({"study": ids, "yi": yi.tolist(), "vi": vi.tolist(), "se": se.tolist(),
                  "ci_low": low.tolist(), "ci_high": high.tolist(), "estimate_original": back[0].tolist(),
                  "ci_low_original": back[1].tolist(), "ci_high_original": back[2].tolist(),
                  "corrected": corrected.tolist()},
                 procedure="meta_effectsize", measure=measure, level=level, zero_policy=zero,
                 correction_rule=".5 to all four cells of zero-containing studies only",
                 smd_variance="Hedges-Olkin LS; exact gamma-ratio J", sample_hash=digest,
                 n_studies=len(ids), dependence=dependence, source=SOURCES["effects"],
                 execution="CPU float64; complete study summaries; no silent deletion")
