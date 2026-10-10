"""Identified AR(1) Gaussian factors with an explicit known finite prior.

This conditional likelihood retains arbitrary missing cells and all dates.
Anchor loading rows equal I_r; Q is unrestricted positive definite, so factor
rotation, scale, sign and labels are fixed by declared anchor measurements.
The shared proper-prior Kalman/RTS kernel supplies every likelihood/moment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.tsworkflows.common import ml_optimize
from openecon.econometrics.tsworkflows.ssengine import (
    FLOAT, FactorAdmission, _finite, _symmetrize, factor_kalman, factor_smooth,
)
from openecon.econometrics.tsworkflows.sspace import _cpu_call
from openecon.econometrics.tsworkflows.ssdiffuse import (
    _calendar, _digest, _future_labels, _index_record, _json_admission, _json_value,
    _load_record, _raw_array, _resident_probe, _response_dtype_admission,
    _restore_index,
)
from openecon.resources import plan_workspace
from openecon.engines.optimize import _HISTORY, _POLISH_ITERATIONS

SCHEMA = "openecon.dynamic-factor.ar1-known-prior.v1"
RESULT_SCHEMA = "openecon.dynamic-factor.result.v1"
MAX_WORK = 200_000_000_000
DEFAULT_WORK = 20_000_000_000
MAX_PARAMETERS = 256
OPTIMIZER_PROVENANCE = "ML counters/value history are bounded optimizer provenance; EM iterates and ML likelihood/score/information endpoints are numerically replayed"


def _positive(matrix, name):
    _finite(matrix, name)
    if not torch.equal(matrix, matrix.T):
        scale = max(float(matrix.detach().abs().max()), torch.finfo(FLOAT).tiny)
        if float((matrix-matrix.T).detach().abs().max()) > 64*len(matrix)*torch.finfo(FLOAT).eps*scale:
            raise AnalysisError("invalid_covariance", f"{name} must be symmetric.")
    factor, status = torch.linalg.cholesky_ex(_symmetrize(matrix))
    if int(status):
        raise AnalysisError("invalid_covariance", f"{name} must be positive definite; no variance floor or jitter.")
    return factor


def _input_positive(matrix, name):
    """Input covariance symmetry/positivity in its own coordinate units."""
    _finite(matrix, name)
    diagonal = matrix.diagonal()
    if not bool((diagonal > 0).all()):
        raise AnalysisError("invalid_covariance", f"{name} requires strictly positive declared variances.")
    scale = diagonal.sqrt()
    normalized = matrix/scale[:, None]/scale[None, :]
    _finite(normalized, f"{name} declared coordinate covariance")
    tolerance = 64*len(matrix)*torch.finfo(FLOAT).eps*max(1., float(normalized.detach().abs().max()))
    if float((normalized-normalized.T).detach().abs().max()) > tolerance:
        raise AnalysisError("invalid_covariance", f"{name} must be symmetric in its declared coordinate units.")
    _positive(normalized, f"{name} declared coordinate covariance")
    return _positive(matrix, name)


def _geometry(n, p, r, *, em_iterations, ml_iterations, restarts, max_work):
    for value, lo, hi, name in ((n, 4, 20000, "dates"), (p, 3, 96, "measurements"),
                                (r, 1, 3, "factors"), (em_iterations, 1, 1000, "EM iterations"),
                                (ml_iterations, 1, 1000, "ML iterations"), (restarts, 1, 8, "restarts"),
                                (max_work, 1, MAX_WORK, "work")):
        if type(value) is not int or not lo <= value <= hi:
            raise AnalysisError("system_budget", f"{name} must be an integer in {lo}..{hi}.")
    if p < 2*r+1:
        raise AnalysisError("unidentified_system", "Exact factors require at least 2*r+1 measurement series.")
    k = 2*p + (p-r)*r + r*r + r*(r+1)//2
    if k > MAX_PARAMETERS:
        raise AnalysisError("system_budget", "The complete factor OIM admits at most 256 free parameters.")
    one = n*(r**3+p**3+r*r*p+r*p*p)
    # Includes line searches (40 trials), all EM/restart passes, and full
    # Hessian/Jacobian replay. This deliberately does not promise cheap OIM
    # just because a low-dimensional factor filter is computationally cheap.
    work = one*(restarts*(88*em_iterations+400*(ml_iterations+_POLISH_ITERATIONS)
                         +(40*_POLISH_ITERATIONS+8)*k*k) + 4*k*k+24)
    if work > max_work:
        raise AnalysisError("system_budget", f"Complete factor fit work {work} exceeds max_work={max_work}.")
    workspace = plan_workspace("identified dynamic factor EM, OIM and portable result", {
        "resident response and observation masks": n*p*32,
        "shared likelihood and retained RTS/disturbance buffers": n*(12*r*r+9*p*p+8*r*p+9*r+9*p)*24,
        "score, full OIM and autodiff graph": n*(r*r+p*p+r*p)*(256+32*k)+k*k*256,
        "full numerical state and serialization": n*(12*r*r+9*p*p+8*r*p+9*r+9*p)*96,
        "EM paths, restart records and native optimization": restarts*(em_iterations+1)*(k+4)*96,
        "typed source identities and metadata": 6*1024**2,
    }).record()
    workspace.update(estimated_work=work, max_work=max_work, free_parameters=k)
    return workspace


class FactorModel:
    """An identified coordinate chart; all parameter covariance uses its J."""

    def __init__(self, p, r, anchors, a0, P0):
        if (not isinstance(anchors, (list, tuple)) or len(anchors) != r
                or any(type(i) is not int or not 0 <= i < p for i in anchors)
                or len(set(anchors)) != r):
            raise AnalysisError("unidentified_system", "Declare one distinct anchor measurement for each factor.")
        self.p, self.r, self.anchors = p, r, list(anchors)
        self.free_rows = [i for i in range(p) if i not in anchors]
        self.qcells = [(i, j) for i in range(r) for j in range(i+1)]
        if (not isinstance(a0, torch.Tensor) or tuple(a0.shape) != (r,)
                or not isinstance(P0, torch.Tensor) or tuple(P0.shape) != (r, r)
                or any(v.device.type != "cpu" or v.dtype != FLOAT for v in (a0, P0))):
            raise AnalysisError("invalid_system", "Known a0/P0 require CPU float64 factor-shaped tensors.")
        _finite(a0, "known factor mean")
        _input_positive(P0, "known factor covariance")
        self.a0, self.P0 = a0.detach(), P0.detach()
        self.names = ([f"intercept[{i}]" for i in range(p)]
                      + [f"loading[{i},{j}]" for i in self.free_rows for j in range(r)]
                      + [f"AR[{i},{j}]" for i in range(r) for j in range(r)]
                      + [f"factor_covariance[{i},{j}]" for i, j in self.qcells]
                      + [f"idiosyncratic_variance[{i}]" for i in range(p)])
        self.profile = FactorAdmission()

    def values(self, theta):
        k = len(self.names)
        if (not isinstance(theta, torch.Tensor) or theta.dtype != FLOAT or theta.device.type != "cpu"
                or tuple(theta.shape) != (k,)):
            raise AnalysisError("invalid_parameter", "Use the complete CPU float64 factor parameter chart.")
        _finite(theta, "factor parameters")
        offset = self.p
        d = theta[:offset]
        L = torch.zeros((self.p, self.r), dtype=FLOAT, device="cpu")
        L[self.anchors] = torch.eye(self.r, dtype=FLOAT, device="cpu")
        length = len(self.free_rows)*self.r
        L[self.free_rows] = theta[offset:offset+length].reshape(-1, self.r)
        offset += length
        A = theta[offset:offset+self.r**2].reshape(self.r, self.r)
        offset += self.r**2
        radius = float(torch.linalg.eigvals(A.detach()).abs().max())
        if not math.isfinite(radius) or radius >= 1-1e-10:
            raise AnalysisError("unstable_system", "Factor AR(1) must have spectral radius strictly below one.")
        C = torch.zeros((self.r, self.r), dtype=FLOAT, device="cpu")
        for cell in self.qcells:
            i, j = cell
            C[i, j] = theta[offset].exp() if i == j else theta[offset]
            offset += 1
        Q = _symmetrize(C @ C.T)
        H = torch.diag(theta[offset:].exp())
        _positive(Q, "factor innovation covariance")
        _positive(H, "idiosyncratic variance")
        return dict(a0=self.a0, P0=self.P0, Z=L, T=A, Q=Q, H=H,
                    c=torch.zeros(self.r, dtype=FLOAT, device="cpu"), d=d)

    def physical(self, theta):
        values = self.values(theta)
        return torch.cat((values["d"], values["Z"][self.free_rows].flatten(), values["T"].flatten(),
                          torch.stack([values["Q"][i, j] for i, j in self.qcells]), values["H"].diagonal()))

    def encode(self, values):
        C = _positive(values["Q"], "factor innovation covariance")
        H = values["H"]
        _positive(H, "idiosyncratic variance")
        if not torch.equal(H, torch.diag(H.diagonal())):
            raise AnalysisError("invalid_system", "This factor stage requires diagonal white idiosyncratic covariance.")
        if not torch.equal(values["Z"][self.anchors], torch.eye(self.r, dtype=FLOAT, device="cpu")):
            raise AnalysisError("unidentified_system", "Anchor loading rows must equal the identity exactly.")
        theta = torch.cat((values["d"], values["Z"][self.free_rows].flatten(), values["T"].flatten(),
                           torch.stack([C[i, j].log() if i == j else C[i, j] for i, j in self.qcells]),
                           H.diagonal().log()))
        self.values(theta)
        return theta


def _start(y, model, ar):
    """Deterministic anchor proxy initialization; no random imputation/EM data."""
    n, p = y.shape
    means, variances = [], []
    for j in range(p):
        observed = y[:, j][torch.isfinite(y[:, j])]
        if len(observed) < max(4, model.r+2):
            raise AnalysisError("unidentified_system", "Every measurement needs enough observed dates to identify its mean/noise.")
        means.append(observed.mean())
        variances.append(((observed-observed.mean())**2).mean())
    d, variance = torch.stack(means), torch.stack(variances)
    if not bool((variance > 0).all()):
        raise AnalysisError("unidentified_system", "A constant measurement cannot identify a strictly positive factor/noise model.")
    proxy = y[:, model.anchors]-d[model.anchors]
    L = torch.zeros((p, model.r), dtype=FLOAT, device="cpu")
    L[model.anchors] = torch.eye(model.r, dtype=FLOAT, device="cpu")
    for i in model.free_rows:
        present = torch.isfinite(y[:, i]) & torch.isfinite(proxy).all(dim=1)
        X = proxy[present]
        if len(X) <= model.r or int(torch.linalg.matrix_rank(X)) != model.r:
            raise AnalysisError("unidentified_system", "Anchor overlaps must identify initial loadings; provide an explicit identified start for sparse overlaps.")
        L[i] = torch.linalg.solve(X.T @ X, X.T @ (y[present, i]-d[i]))
    A = torch.eye(model.r, dtype=FLOAT, device="cpu")*ar
    Q = torch.diag(variance[model.anchors]*.5*(1-ar*ar))
    H = torch.diag(variance*.5)
    return model.encode(dict(Z=L, T=A, Q=Q, H=H, d=d))


def _em_update(y, model, theta, posterior):
    """Expected complete regressions on observed cells, with exact anchors.

    Missing measurements have no residual term in the objective. The latent
    factor sufficient statistics come from all retained calendar dates.
    Cov(a[t+1],a[t]|Y) is the shared RTS lag-one covariance orientation.
    """
    old = model.values(theta)
    f, V, lag = (posterior[k] for k in ("smoothed", "smoothed_covariance", "lag_one_covariance"))
    E = V + f[:, :, None]*f[:, None, :]
    cross = lag + f[1:, :, None]*f[:-1, None, :]
    left = E[:-1].sum(dim=0)
    A = torch.linalg.solve(left, cross.sum(dim=0).T).T
    Q = _symmetrize((E[1:].sum(dim=0) - A @ cross.sum(dim=0).T
                     - cross.sum(dim=0) @ A.T + A @ left @ A.T)/(len(y)-1))
    _positive(Q, "EM factor innovation covariance")
    L, d, noise = old["Z"].clone(), old["d"].clone(), old["H"].diagonal().clone()
    for j in range(model.p):
        present = torch.isfinite(y[:, j])
        fy, ey, yy = f[present], E[present], y[present, j]
        count = len(yy)
        if j in model.anchors:
            d[j] = (yy-fy @ L[j]).mean()
        else:
            S = torch.zeros((model.r+1, model.r+1), dtype=FLOAT, device="cpu")
            S[0, 0], S[0, 1:], S[1:, 0], S[1:, 1:] = count, fy.sum(dim=0), fy.sum(dim=0), ey.sum(dim=0)
            rhs = torch.cat((yy.sum()[None], (yy[:, None]*fy).sum(dim=0)))
            coefficients = torch.linalg.solve(S, rhs)
            d[j], L[j] = coefficients[0], coefficients[1:]
        residual = yy-d[j]
        noise[j] = (residual.square().sum()-2*(residual[:, None]*fy).sum(dim=0) @ L[j]
                    + L[j] @ ey.sum(dim=0) @ L[j])/count
    candidate = dict(Z=L, T=A, Q=Q, H=torch.diag(noise), d=d)
    return candidate


def _likelihood(y, model, theta):
    return factor_kalman(y, model.values(theta), profile=model.profile)["log_likelihood"]


def _moments_record(output):
    # Workspace records depend on the current resource override; they are
    # retained in the admission provenance rather than numerical semantics.
    return {k: _json_value(v) for k, v in output.items() if "workspace" not in k}


def _inference(y, model, theta):
    with torch.enable_grad():
        point = theta.detach().clone().requires_grad_(True)
        likelihood = _likelihood(y, model, point)
        score = torch.autograd.grad(likelihood, point)[0]
        information = -torch.autograd.functional.hessian(lambda z: _likelihood(y, model, z), point)
        information = _symmetrize(information)
        factor = _positive(information, "observed factor information")
        scale = information.diagonal().detach().sqrt()
        correlation = information.detach()/scale[:, None]/scale[None, :]
        eigen = torch.linalg.eigvalsh(correlation)
        if float(eigen[0]) <= 128*len(point)*torch.finfo(FLOAT).eps*float(eigen[-1]):
            raise AnalysisError("unidentified_system", "Observed factor information is near singular in its declared parameter units.")
        chart_covariance = torch.cholesky_inverse(factor)
        J = torch.autograd.functional.jacobian(model.physical, point)
        covariance = _symmetrize(J @ chart_covariance @ J.T)
        parameters = model.physical(point)
    return dict(parameters=parameters.detach(), covariance=covariance.detach(),
                chart_covariance=chart_covariance.detach(), information=information.detach(),
                score=score.detach(), jacobian=J.detach(), reference="normal", df=None,
                covariance_kind="nonrobust-observed-information")


@_cpu_call
def fit_dynamic_factor(y, *, factors=1, anchors=None, a0=None, P0=None, starts=None,
                       max_em_iterations=25, max_ml_iterations=200, restarts=2,
                       tolerance=1e-7, max_work=DEFAULT_WORK):
    """Complete conditional AR(1) MLE after observed-cell EM starts.

    EM is followed by native analytic-score BFGS for an identified interior
    solution. Both phases are declared; EM stopping is not labelled complete
    MLE convergence. Positive definite OIM is mandatory for inferential fits.
    """
    if (not isinstance(y, torch.Tensor) or y.dtype != FLOAT or y.device.type != "cpu" or y.ndim != 2):
        raise AnalysisError("invalid_data", "Use a resident CPU float64 date-by-measurement tensor.")
    n, p = y.shape
    workspace = _geometry(n, p, factors, em_iterations=max_em_iterations, ml_iterations=max_ml_iterations,
                          restarts=restarts, max_work=max_work)
    if type(tolerance) not in {int, float} or not 1e-10 <= tolerance <= 1e-3:
        raise AnalysisError("invalid_tolerance", "tolerance must be finite in 1e-10..1e-3.")
    if bool(torch.isinf(y).any()):
        raise AnalysisError("invalid_data", "Infinite measurements are not missing values.")
    y = y.detach()
    r = factors
    anchors = list(range(r)) if anchors is None else anchors
    a0 = torch.zeros(r, dtype=FLOAT, device="cpu") if a0 is None else a0
    P0 = torch.eye(r, dtype=FLOAT, device="cpu") if P0 is None else P0
    model = FactorModel(p, r, anchors, a0, P0)
    if starts is None:
        starts = [_start(y, model, .15+.7*i/max(1, restarts-1)) for i in range(restarts)]
    elif (not isinstance(starts, (list, tuple)) or len(starts) != restarts
          or any(not isinstance(v, torch.Tensor) for v in starts)):
        raise AnalysisError("invalid_parameter", "Supply exactly one bounded native parameter tensor per restart.")
    for initial in starts:
        model.values(initial)
    # A wholly missing row remains a transition. A wholly missing series has
    # unidentifiable mean/noise, even if other series observe the factors.
    if bool((torch.isfinite(y).sum(dim=0) < r+3).any()):
        raise AnalysisError("unidentified_system", "Every measurement needs at least r+3 observed dates.")
    restart_records, solutions = [], []
    for restart, initial in enumerate(starts):
        theta = initial.detach().clone()
        old = float(_likelihood(y, model, theta))
        path = [dict(theta=theta.tolist(), log_likelihood=old, step=0, fraction=1.)]
        em_stopped = False
        with torch.no_grad():
            for iteration in range(1, max_em_iterations+1):
                posterior = factor_smooth(y, model.values(theta), profile=model.profile)
                proposed = _em_update(y, model, theta, posterior)
                base = model.values(theta)
                accepted = False
                for trial in range(40):
                    fraction = 2.**-trial
                    blended = {k: base[k]+fraction*(proposed[k]-base[k]) for k in proposed}
                    try:
                        candidate = model.encode(blended)
                        objective = float(_likelihood(y, model, candidate))
                    except (AnalysisError, RuntimeError):
                        continue
                    if objective >= old:
                        accepted = True
                        break
                if not accepted:
                    em_stopped = True
                    break
                improvement = objective-old
                theta, old = candidate, objective
                path.append(dict(theta=theta.tolist(), log_likelihood=old, step=iteration, fraction=fraction))
                if improvement <= tolerance*(1+abs(old)):
                    em_stopped = True
                    break
        record = dict(restart=restart, em_path=path, em_stopped=em_stopped,
                      em_iterations=len(path)-1, em_max_iterations=max_em_iterations)
        try:
            solution, _ = ml_optimize(lambda z: _likelihood(y, model, z), theta, max_ml_iterations, tolerance)
            if not solution.converged:
                raise AnalysisError("nonconvergence", "Native factor likelihood polishing did not converge.")
            final = float(_likelihood(y, model, solution.theta))
            if final < old-1e-10*(1+abs(old)):
                raise AnalysisError("nonconvergence", "Final factor likelihood is below its EM starting objective.")
            record.update(converged=True, ml_iterations=solution.iterations,
                          ml_gradient_max=float(solution.gradient.abs().max()),
                          final_theta=solution.theta.tolist(), final_log_likelihood=final,
                          ml_diagnostics=_json_value(solution.diagnostics))
            solutions.append((final, restart, solution.theta.detach()))
        except AnalysisError as error:
            record.update(converged=False, failure_code=error.code, failure=str(error))
        restart_records.append(record)
    if not solutions:
        raise AnalysisError("nonconvergence", "Every factor restart failed convergence/positive OIM; no inferential fit returned.")
    objective, chosen, theta = max(solutions, key=lambda value: (value[0], -value[1]))
    inference = _inference(y, model, theta)
    with torch.no_grad():
        values = model.values(theta)
        output = factor_smooth(y, values, profile=model.profile)
    record = dict(schema=SCHEMA, y=_json_value(y), factors=r, anchors=list(anchors),
                  prior=dict(a0=a0.tolist(), P0=P0.tolist(), target="conditional-known-finite-prior"),
                  settings=dict(max_em_iterations=max_em_iterations, max_ml_iterations=max_ml_iterations,
                                restarts=restarts, tolerance=float(tolerance), max_work=max_work),
                  names=model.names, theta=theta.tolist(), system=_json_value(values),
                  inference=_json_value(inference), output=_moments_record(output),
                  optimization=dict(chosen_restart=chosen, log_likelihood=objective, restarts=restart_records,
                                    converged=True, algorithm="observed-cell constrained EM; native analytic-score BFGS; observed information"),
                  admission=workspace)
    record["optimization"]["provenance_scope"] = OPTIMIZER_PROVENANCE
    _json_admission(record)
    record["sha256"] = _digest(record)
    return record


def _source(data, names, time, factors, settings):
    """Shape-only resident admission before scanning/coercing data columns."""
    import numpy as np
    import pandas as pd
    from openecon.analysis import _coerce_frame

    if (not isinstance(names, (list, tuple)) or not 3 <= len(names) <= 96
            or any(not isinstance(v, str) or not v or len(v) > 200 for v in names)
            or len(set(names)) != len(names)
            or time is not None and (not isinstance(time, str) or not time or len(time) > 200 or time in names)):
        raise AnalysisError("invalid_data", "Declare distinct bounded measurement columns and a separate calendar column.")
    if isinstance(data, pd.DataFrame):
        n = len(data)
        if data.shape[1] > 1024:
            raise AnalysisError("invalid_data", "Project resident DataFrames with more than 1024 source columns first.")
    elif isinstance(data, Mapping):
        columns = list(names)+([time] if time is not None else [])
        if any(v not in data for v in columns):
            raise AnalysisError("invalid_data", "Supply all declared source columns.")
        values = [data[v] for v in columns]
        if any(not isinstance(v, (list, tuple, range, np.ndarray, pd.Index, pd.Series))
               or isinstance(v, np.ndarray) and v.ndim != 1 for v in values):
            raise AnalysisError("unsupported_input", "Use bounded resident columns; Dataset collection and iterators are unsupported.")
        if len({len(v) for v in values}) != 1:
            raise AnalysisError("invalid_data", "All resident source columns need the same date count.")
        n = len(values[0])
    else:
        raise AnalysisError("unsupported_input", "Use a resident DataFrame or bounded resident mapping.")
    _geometry(n, len(names), factors, **settings)
    _, _, admitted = _resident_probe(data, list(names), time)
    if isinstance(admitted, pd.DataFrame):
        admitted = admitted[list(names)+([time] if time is not None else [])]
    frame = _coerce_frame(admitted)
    if len(frame) != n:
        raise AnalysisError("invalid_data", "Source coercion changed admitted date geometry.")
    calendar = rule = None
    if time is not None:
        calendar = pd.Index(frame[time])
        rule = _calendar(calendar)
        calendar = _index_record(calendar)
    y = torch.tensor(frame[list(names)].to_numpy(dtype="float64", na_value=math.nan), dtype=FLOAT, device="cpu")
    dtypes = [str(frame[v].dtype) for v in names]
    _response_dtype_admission(dtypes, _json_value(y))
    return y, dict(responses=list(names), response_dtypes=dtypes, index=_index_record(frame.index),
                   time=calendar, calendar=rule, missing="mask", alpha=.05)


def _tables(record):
    from openecon.engines.distributions import normal_isf, normal_sf

    state, source = record["state"], record["source"]
    inference, output = state["inference"], state["output"]
    n, r = len(state["y"]), state["factors"]
    index = _restore_index(source["index"], n)
    critical = float(normal_isf(source["alpha"]/2))
    rows = []
    for i, name in enumerate(state["names"]):
        estimate, variance = inference["parameters"][i], inference["covariance"][i][i]
        if variance <= 0:
            raise AnalysisError("invalid_covariance", "Every free factor parameter needs positive finite OIM variance.")
        se = math.sqrt(variance)
        z = estimate/se
        rows.append([name, estimate, se, z, 2*float(normal_sf(abs(z))), estimate-critical*se, estimate+critical*se])
    tables = dict(coefficients=table(rows, columns=["term", "estimate", "se", "z", "p", "lo", "hi"]),
                  covariance=table(inference["covariance"], columns=state["names"], index=state["names"]))
    for kind in ("filtered", "smoothed"):
        rows = []
        for t in range(n):
            for j in range(r):
                mean, variance = output[kind][t][j], output[kind+"_covariance"][t][j][j]
                sd = math.sqrt(variance)
                rows.append([t, source["responses"][state["anchors"][j]], mean, sd,
                             mean-critical*sd, mean+critical*sd])
        tables[kind] = table(rows, columns=["position", "anchor", "mean", "conditional_sd", "lo", "hi"], index=index.repeat(r))
        tables[kind].index = index.repeat(r)
    tables["likelihood"] = table(dict(position=list(range(n)),
        log_likelihood=output["log_likelihood_contributions"], observed_cells=output["observed_count"]), index=index)
    tables["likelihood"].index = index
    if source["time"] is not None:
        calendar = _restore_index(source["time"], n)
        for name in ("filtered", "smoothed", "likelihood"):
            tables[name].insert(0, "time", calendar if name == "likelihood" else calendar.repeat(r))
    return DynamicFactorResult(tables, title="Identified dynamic Gaussian factors (AR1)",
        dynamic_factor_result=record, n_periods=n, observed_cells=sum(output["observed_count"]),
        log_likelihood=output["log_likelihood"], converged=True, alpha=source["alpha"],
        sample_positions=list(range(n)), likelihood_target="conditional-known-finite-prior",
        inference_reference="normal", inference_df=None,
        notes=["Declared anchor loading rows equal identity; factors have explicit scale, sign and labels.",
               "Native observed-cell EM starts followed by analytic-score likelihood optimization; complete nonrobust observed-information covariance.",
               "Factor/state intervals condition on fitted coefficients; forecasts distinguish process variance and delta-method parameter variance.",
               "The initial factor mean/covariance are fixed caller assumptions, before the first measurement.",
               "All original dates, source identities and arbitrary observation masks are retained."])


class DynamicFactorResult(TableSet):
    def to_json(self):
        import json
        return json.dumps(self.attrs["dynamic_factor_result"], sort_keys=True, allow_nan=False)


@_cpu_call
def dfactor(data, y, *, factors=1, anchors=None, time=None, a0=None, P0=None, starts=None,
            max_em_iterations=25, max_ml_iterations=200, restarts=2, tolerance=1e-7,
            alpha=.05, max_work=DEFAULT_WORK) -> DynamicFactorResult:
    """Fit identified AR(1) factors to resident measurements, preserving dates.

    Anchor names specify the identity loading block. White diagonal
    idiosyncratic noise and the known finite factor prior define this stage.
    """
    if type(alpha) not in {int, float} or not 0 < alpha < 1:
        raise AnalysisError("invalid_alpha", "alpha must lie strictly between zero and one.")
    if not isinstance(y, (list, tuple)) or not 3 <= len(y) <= 96:
        raise AnalysisError("invalid_data", "Declare 3..96 bounded resident measurement names.")
    names = list(y)
    y_tensor, source = _source(data, names, time, factors, dict(em_iterations=max_em_iterations,
        ml_iterations=max_ml_iterations, restarts=restarts, max_work=max_work))
    if anchors is None:
        anchor_indices = list(range(factors))
    elif isinstance(anchors, (list, tuple)) and all(isinstance(v, str) for v in anchors):
        if any(v not in names for v in anchors):
            raise AnalysisError("unidentified_system", "Every factor anchor must name a measurement column.")
        anchor_indices = [names.index(v) for v in anchors]
    else:
        raise AnalysisError("unidentified_system", "Declare factor anchors by measurement names.")
    source["alpha"] = float(alpha)
    prior = {}
    for key, value, shape in (("a0", a0, (factors,)), ("P0", P0, (factors, factors))):
        if value is None or isinstance(value, torch.Tensor):
            prior[key] = value
        elif _raw_array(value, shape):
            prior[key] = torch.tensor(value, dtype=FLOAT, device="cpu")
        else:
            raise AnalysisError("invalid_system", "Known factor prior has invalid resident geometry.")
    state = fit_dynamic_factor(y_tensor, factors=factors, anchors=anchor_indices, starts=starts,
        **prior, max_em_iterations=max_em_iterations, max_ml_iterations=max_ml_iterations,
        restarts=restarts, tolerance=tolerance, max_work=max_work)
    record = dict(schema=RESULT_SCHEMA, state=state, source=source)
    _json_admission(record)
    record["sha256"] = _digest(record)
    return _tables(record)


@_cpu_call
def restore_dynamic_factor_state(value):
    """Recompute source likelihood, OIM and RTS semantics without any optimizer."""
    try:
        state = _load_record(value)
        keys = {"schema", "y", "factors", "anchors", "prior", "settings", "names", "theta", "system",
                "inference", "output", "optimization", "admission", "sha256"}
        if set(state) != keys or state["schema"] != SCHEMA:
            raise ValueError("Invalid factor state schema")
        if state["sha256"] != _digest({k: v for k, v in state.items() if k != "sha256"}):
            raise ValueError("Invalid factor state checksum")
        settings = state["settings"]
        if set(settings) != {"max_em_iterations", "max_ml_iterations", "restarts", "tolerance", "max_work"}:
            raise ValueError("Invalid factor settings")
        n, p, r = len(state["y"]), len(state["y"][0]), state["factors"]
        plan = _geometry(n, p, r, em_iterations=settings["max_em_iterations"], ml_iterations=settings["max_ml_iterations"],
                         restarts=settings["restarts"], max_work=settings["max_work"])
        if (not isinstance(state["admission"], dict) or set(state["admission"]) != set(plan)
                or any(_digest(state["admission"][k]) != _digest(plan[k]) for k in plan if k != "budget_bytes")
                or type(state["admission"]["budget_bytes"]) is not int
                or state["admission"]["budget_bytes"] < plan["estimated_workspace_bytes"]):
            raise ValueError("Invalid original factor admission plan")
        if (not _raw_array(state["y"], (n, p), missing=True)
                or not _raw_array(state["prior"]["a0"], (r,)) or not _raw_array(state["prior"]["P0"], (r, r))
                or set(state["prior"]) != {"a0", "P0", "target"}
                or state["prior"]["target"] != "conditional-known-finite-prior"):
            raise ValueError("Invalid source/prior geometry")
        y = torch.tensor([[math.nan if v is None else v for v in row] for row in state["y"]], dtype=FLOAT, device="cpu")
        model = FactorModel(p, r, state["anchors"], torch.tensor(state["prior"]["a0"], dtype=FLOAT, device="cpu"),
                            torch.tensor(state["prior"]["P0"], dtype=FLOAT, device="cpu"))
        if state["names"] != model.names or not _raw_array(state["theta"], (len(model.names),)):
            raise ValueError("Invalid identified parameter coordinates")
        theta = torch.tensor(state["theta"], dtype=FLOAT, device="cpu")
        values = model.values(theta)
        if _digest(_json_value(values)) != _digest(state["system"]):
            raise ValueError("Saved physical factor equations differ from parameter coordinates")
        inference = _inference(y, model, theta)
        with torch.no_grad():
            output = factor_smooth(y, values, profile=model.profile)
        if (_digest(_json_value(inference)) != _digest(state["inference"])
                or _digest(_moments_record(output)) != _digest(state["output"])):
            raise ValueError("Saved factor likelihood/OIM/posterior does not replay")
        optimization = state["optimization"]
        if (set(optimization) != {"chosen_restart", "log_likelihood", "restarts", "converged", "algorithm", "provenance_scope"}
                or optimization["algorithm"] != "observed-cell constrained EM; native analytic-score BFGS; observed information"
                or optimization["provenance_scope"] != OPTIMIZER_PROVENANCE
                or type(optimization["converged"]) is not bool or not optimization["converged"]
                or type(optimization["chosen_restart"]) is not int
                or not 0 <= optimization["chosen_restart"] < settings["restarts"]
                or len(optimization["restarts"]) != settings["restarts"]
                or optimization["log_likelihood"] != state["output"]["log_likelihood"]):
            raise ValueError("Invalid factor convergence record")
        tolerance = settings["tolerance"]
        if type(tolerance) not in {int, float} or not 1e-10 <= tolerance <= 1e-3:
            raise ValueError("Invalid factor tolerance")
        scaled = float(inference["score"] @ inference["chart_covariance"] @ inference["score"])
        if scaled > max(tolerance*tolerance, 1e-9):
            raise ValueError("Cached factor solution fails observed-information convergence")
        for i, restart in enumerate(optimization["restarts"]):
            path = restart["em_path"]
            expected_keys = {"restart", "em_path", "em_stopped", "em_iterations", "em_max_iterations", "converged"}
            expected_keys |= {"ml_iterations", "ml_gradient_max", "final_theta", "final_log_likelihood", "ml_diagnostics"} if restart["converged"] else {"failure_code", "failure"}
            if (type(restart["restart"]) is not int or restart["restart"] != i
                    or set(restart) != expected_keys
                    or type(restart["converged"]) is not bool or type(restart["em_stopped"]) is not bool
                    or not isinstance(path, list) or not 1 <= len(path) <= settings["max_em_iterations"]+1
                    or type(restart["em_iterations"]) is not int or restart["em_iterations"] != len(path)-1
                    or type(restart["em_max_iterations"]) is not int or restart["em_max_iterations"] != settings["max_em_iterations"]):
                raise ValueError("Invalid EM iteration record")
            last = -math.inf
            previous = None
            for j, step in enumerate(path):
                if (set(step) != {"theta", "log_likelihood", "step", "fraction"}
                        or type(step["step"]) is not int or step["step"] != j
                        or type(step["fraction"]) not in {int, float} or step["fraction"] not in {2.**-v for v in range(40)}
                        or not _raw_array(step["theta"], (len(model.names),))):
                    raise ValueError("Invalid EM path step")
                current = torch.tensor(step["theta"], dtype=FLOAT, device="cpu")
                if previous is not None:
                    with torch.no_grad():
                        base = model.values(previous)
                        posterior = factor_smooth(y, base, profile=model.profile)
                        proposed = _em_update(y, model, previous, posterior)
                        blend = {k: base[k]+step["fraction"]*(proposed[k]-base[k]) for k in proposed}
                        expected_theta = model.encode(blend)
                    if _digest(expected_theta.tolist()) != _digest(step["theta"]):
                        raise ValueError("Saved EM step does not reproduce constrained expected regressions")
                elif step["fraction"] != 1.:
                    raise ValueError("Initial factor EM path fraction must equal one")
                objective = float(_likelihood(y, model, current))
                if objective != step["log_likelihood"] or objective < last:
                    raise ValueError("EM likelihood path is not valid and monotone")
                last = objective
                previous = current
            if restart["converged"]:
                if (not _raw_array(restart["final_theta"], (len(model.names),))
                        or type(restart["ml_iterations"]) is not int
                        or not 0 <= restart["ml_iterations"] <= settings["max_ml_iterations"]+_POLISH_ITERATIONS):
                    raise ValueError("Invalid ML convergence record")
                endpoint = torch.tensor(restart["final_theta"], dtype=FLOAT, device="cpu")
                objective = float(_likelihood(y, model, endpoint))
                if objective != restart["final_log_likelihood"] or objective < last-1e-10*(1+abs(last)):
                    raise ValueError("Invalid final restart likelihood")
                endpoint_inference = inference if _digest(restart["final_theta"]) == _digest(state["theta"]) else _inference(y, model, endpoint)
                endpoint_scaled = float(endpoint_inference["score"] @ endpoint_inference["chart_covariance"] @ endpoint_inference["score"])
                endpoint_gradient = float(endpoint_inference["score"].abs().max())
                diagnostic = restart["ml_diagnostics"]
                expected_keys = {"gradient_max", "scaled_gradient", "last_step_max", "backtracks", "nonconcave_iterations",
                                 "restarts", "polish_iterations", "function_evaluations", "concave", "hessian_source", "message", "value_history"}
                if (not isinstance(diagnostic, dict) or set(diagnostic) != expected_keys
                        or type(diagnostic["concave"]) is not bool or not diagnostic["concave"]
                        or diagnostic["hessian_source"] != "hessian_fn"
                        or endpoint_scaled > max(tolerance*tolerance, 1e-9)):
                    raise ValueError("Invalid native ML information/convergence diagnostic")
                limit = 400*(settings["max_ml_iterations"]+_POLISH_ITERATIONS)+1
                for key in ("backtracks", "nonconcave_iterations", "restarts", "polish_iterations", "function_evaluations"):
                    if type(diagnostic[key]) is not int or not 0 <= diagnostic[key] <= limit:
                        raise ValueError("Invalid native ML diagnostic counter")
                if (diagnostic["polish_iterations"] > _POLISH_ITERATIONS
                        or diagnostic["nonconcave_iterations"] > restart["ml_iterations"]
                        or diagnostic["restarts"] > settings["max_ml_iterations"]+1
                        or diagnostic["function_evaluations"] < restart["ml_iterations"]+1
                        or diagnostic["backtracks"] >= diagnostic["function_evaluations"]):
                    raise ValueError("Native ML diagnostic counters exceed declared iterations")
                messages = {"converged: gradient and scaled gradient", "converged: no further step at float64 resolution"}
                messages.add(f"converged: scaled gradient after {diagnostic['polish_iterations']} Newton polishing step(s)")
                if (not isinstance(diagnostic["message"], str) or diagnostic["message"] not in messages
                        or type(diagnostic["last_step_max"]) not in {int, float}
                        or not math.isfinite(diagnostic["last_step_max"]) or diagnostic["last_step_max"] < 0):
                    raise ValueError("Invalid native ML provenance message/step")
                for actual, expected in ((restart["ml_gradient_max"], endpoint_gradient),
                                         (diagnostic["gradient_max"], endpoint_gradient),
                                         (diagnostic["scaled_gradient"], endpoint_scaled)):
                    unit = max(abs(expected), torch.finfo(FLOAT).tiny)
                    if (type(actual) not in {int, float} or not math.isfinite(actual) or actual < 0
                            or abs(actual-expected) > 512*len(model.names)*torch.finfo(FLOAT).eps*unit):
                        raise ValueError("Saved native ML gradient does not reproduce the endpoint")
                history = diagnostic["value_history"]
                if (not isinstance(history, list) or len(history) != min(restart["ml_iterations"]+1, _HISTORY)
                        or any(type(v) not in {int, float} or not math.isfinite(v) for v in history)
                        or any(b < a-1e-10*(1+abs(a)) for a, b in zip(history, history[1:]))
                        or history[-1] != objective
                        or restart["ml_iterations"]+1 <= _HISTORY and history[0] != last):
                    raise ValueError("Invalid bounded native ML objective provenance")
            elif (not isinstance(restart["failure_code"], str) or not 1 <= len(restart["failure_code"]) <= 100
                  or not isinstance(restart["failure"], str) or not 1 <= len(restart["failure"]) <= 2000):
                raise ValueError("Invalid declared failure provenance")
        selected = optimization["restarts"][optimization["chosen_restart"]]
        if (not selected["converged"] or _digest(selected["final_theta"]) != _digest(state["theta"])
                or selected["final_log_likelihood"] != optimization["log_likelihood"]
                or any(v["converged"] and v["final_log_likelihood"] > optimization["log_likelihood"]
                       for v in optimization["restarts"])):
            raise ValueError("The chosen factor restart does not reproduce the selected solution")
        return state
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, RuntimeError) as error:
        raise AnalysisError("invalid_state", "Saved dynamic factor state fails optimizer-free semantic replay.") from error


@_cpu_call
def dfactor_restore(result) -> DynamicFactorResult:
    """Restore complete typed rows, masks, conditional model and inference."""
    try:
        value = result.attrs["dynamic_factor_result"] if isinstance(result, DynamicFactorResult) else result
        record = _load_record(value)
        if (set(record) != {"schema", "state", "source", "sha256"} or record["schema"] != RESULT_SCHEMA
                or record["sha256"] != _digest({k: v for k, v in record.items() if k != "sha256"})):
            raise ValueError("Invalid factor result schema/checksum")
        state = restore_dynamic_factor_state(record["state"])
        source = record["source"]
        n, p = len(state["y"]), len(state["y"][0])
        if (set(source) != {"responses", "response_dtypes", "index", "time", "calendar", "missing", "alpha"}
                or not isinstance(source["responses"], list) or len(source["responses"]) != p
                or any(not isinstance(v, str) or not v or len(v) > 200 for v in source["responses"])
                or len(set(source["responses"])) != p or source["missing"] != "mask"
                or not isinstance(source["response_dtypes"], list) or len(source["response_dtypes"]) != p
                or type(source["alpha"]) not in {int, float} or not 0 < source["alpha"] < 1):
            raise ValueError("Invalid factor source metadata")
        _response_dtype_admission(source["response_dtypes"], state["y"])
        _restore_index(source["index"], n)
        if source["time"] is None:
            if source["calendar"] is not None:
                raise ValueError("Calendar rule needs typed dates")
        elif _calendar(_restore_index(source["time"], n)) != source["calendar"]:
            raise ValueError("Calendar frequency does not reproduce source dates")
        return _tables(record)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        raise AnalysisError("invalid_state", "Saved dynamic factor typed source fails semantic replay.") from error


def _prediction_admit(state, cells, *, parameter_uncertainty, max_work):
    if type(parameter_uncertainty) is not bool:
        raise AnalysisError("invalid_inference", "parameter_uncertainty must be an explicit boolean.")
    if type(cells) is not int or not 1 <= cells <= 512:
        raise AnalysisError("system_budget", "Joint nowcast/forecast uncertainty admits 1..512 target cells.")
    if type(max_work) is not int or not 1 <= max_work <= MAX_WORK:
        raise AnalysisError("invalid_resource_budget", "Prediction work must be a bounded integer.")
    try:
        if (not isinstance(state, dict) or not isinstance(state["y"], list) or not state["y"]
                or not isinstance(state["y"][0], list) or not state["y"][0]
                or not isinstance(state["theta"], list) or not 1 <= len(state["theta"]) <= MAX_PARAMETERS
                or type(state["factors"]) is not int or not 1 <= state["factors"] <= 3):
            raise ValueError("Invalid prediction geometry")
        n, p, r, k = len(state["y"]), len(state["y"][0]), state["factors"], len(state["theta"])
        if not 4 <= n <= 20000 or not 2*r+1 <= p <= 96:
            raise ValueError("Invalid prediction source dimensions")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise AnalysisError("invalid_state", "Prediction needs a bounded complete dynamic factor state.") from error
    work = n*(r**3+p**3+r*r*p+r*p*p)*(4*k if parameter_uncertainty else 8) + n*cells*cells*r**3
    if work > max_work:
        raise AnalysisError("system_budget", "Full prediction covariance/derivatives exceed the declared work budget.")
    return plan_workspace("joint dynamic factor prediction and parameter delta covariance", {
        "prediction conditional and delta covariance": cells*cells*8*12,
        "target derivative and parameter covariance": (cells*k+k*k)*8*12,
        "smoother and differentiable prediction paths": n*(r*r+p*p+r*p)*(512+16*k if parameter_uncertainty else 128),
        "prediction typed identities and serialization": cells*cells*96+4*1024**2,
    }).record()


def _replay_model(state):
    r, p = state["factors"], len(state["y"][0])
    y = torch.tensor([[math.nan if v is None else v for v in row] for row in state["y"]], dtype=FLOAT, device="cpu")
    model = FactorModel(p, r, state["anchors"], torch.tensor(state["prior"]["a0"], dtype=FLOAT, device="cpu"),
                        torch.tensor(state["prior"]["P0"], dtype=FLOAT, device="cpu"))
    return y, model, torch.tensor(state["theta"], dtype=FLOAT, device="cpu")


def _prediction_tables(means, conditional, delta, labels, responses, alpha, **attrs):
    from openecon.engines.distributions import normal_isf

    total = _symmetrize(conditional+delta)
    _finite(total, "joint forecast covariance")
    critical = float(normal_isf(alpha/2))
    rows = []
    for i, (mean, name) in enumerate(zip(means.tolist(), responses)):
        variance = float(total[i, i])
        if variance <= 0:
            raise AnalysisError("invalid_covariance", "Prediction variance must remain strictly positive.")
        sd = math.sqrt(variance)
        rows.append([name, mean, float(conditional[i, i]), float(delta[i, i]), sd,
                     mean-critical*sd, mean+critical*sd])
    tables = dict(predictions=table(rows, columns=["response", "mean", "conditional_variance", "parameter_variance", "sd", "lo", "hi"], index=labels),
                  conditional_covariance=table(conditional.tolist()), parameter_covariance=table(delta.tolist()),
                  covariance=table(total.tolist()))
    tables["predictions"].index = labels
    return TableSet(tables, title="Dynamic factor prediction", alpha=alpha,
        inference="Gaussian conditional process/state variance; first-order delta method for fitted mean parameter uncertainty",
        parameter_uncertainty=attrs.pop("parameter_uncertainty"), **attrs)


@_cpu_call
def dfactor_forecast(result, steps, *, future_time=None, parameter_uncertainty=True,
                     max_work=DEFAULT_WORK) -> TableSet:
    """Joint future measurements with process and optional parameter variance.

    The parameter component is a first-order delta approximation for the
    conditional mean, including the fitted terminal state's dependence on
    coefficients. It is not an integrated Bayesian predictive distribution.
    """
    import pandas as pd

    if type(steps) is not int or not 1 <= steps <= 128:
        raise AnalysisError("invalid_horizon", "Factor forecast admits 1..128 future dates.")
    value = result.attrs["dynamic_factor_result"] if isinstance(result, DynamicFactorResult) else result
    raw = _load_record(value)
    state, source = raw["state"], raw["source"]
    p = len(state["y"][0])
    workspace = _prediction_admit(state, steps*p, parameter_uncertainty=parameter_uncertainty, max_work=max_work)
    explicit = _future_labels(future_time, steps) if future_time is not None else None
    restored = dfactor_restore(raw)
    record = restored.attrs["dynamic_factor_result"]
    state, source = record["state"], record["source"]
    if source["time"] is not None:
        calendar = _restore_index(source["time"], len(state["y"]))
        rule = source["calendar"]
        if explicit is None:
            if rule["kind"] == "datetime":
                explicit = pd.date_range(calendar[-1], periods=steps+1, freq=rule["frequency"])[1:]
            else:
                last = int(calendar[-1]) if type(rule["step"]) is int else float(calendar[-1])
                explicit = pd.Index([last+rule["step"]*(i+1) for i in range(steps)])
        combined = calendar.append(explicit)
        if _calendar(combined) != rule:
            raise AnalysisError("time_gaps", "Future factor dates must continue the saved regular calendar exactly.")
    else:
        explicit = pd.RangeIndex(1, steps+1, name="horizon") if explicit is None else explicit
    labels = explicit.repeat(p)
    y, model, theta = _replay_model(state)

    def future_means(z):
        values = model.values(z)
        filtered = factor_kalman(y, values, profile=model.profile)
        mean = filtered["next_mean"]
        predictions = []
        for _ in range(steps):
            predictions.append(values["Z"] @ mean+values["d"])
            mean = values["T"] @ mean
        return torch.stack(predictions).flatten()

    with torch.no_grad():
        values = model.values(theta)
        A, L, Q, H = (values[k] for k in ("T", "Z", "Q", "H"))
        means = future_means(theta)
        P = torch.tensor(state["output"]["next_covariance"], dtype=FLOAT, device="cpu")
        covariances = []
        for _ in range(steps):
            covariances.append(P)
            P = _symmetrize(A @ P @ A.T+Q)
        conditional = torch.zeros((steps*p, steps*p), dtype=FLOAT, device="cpu")
        for i in range(steps):
            cross = covariances[i]
            for j in range(i, steps):
                block = L @ cross @ L.T
                if i == j:
                    block = block+H
                conditional[i*p:(i+1)*p, j*p:(j+1)*p] = block
                conditional[j*p:(j+1)*p, i*p:(i+1)*p] = block.T
                cross = cross @ A.T
    delta = torch.zeros_like(conditional)
    if parameter_uncertainty:
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(future_means, theta)
        covariance = torch.tensor(state["inference"]["chart_covariance"], dtype=FLOAT, device="cpu")
        delta = _symmetrize(J @ covariance @ J.T).detach()
    return _prediction_tables(means, conditional, delta, labels, source["responses"]*steps, source["alpha"],
        parameter_uncertainty=parameter_uncertainty, workspace=workspace, result_sha256=record["sha256"],
        horizon=steps, future_time=_index_record(explicit), fit_origin=len(state["y"])-1)


@_cpu_call
def dfactor_nowcast(result, targets, *, parameter_uncertainty=True, max_work=DEFAULT_WORK) -> TableSet:
    """Joint missing measurement moments on the retained historical calendar.

    Targets are (original positional date, response name). Every target must
    be missing in the fitted source. This API conditions on the saved entire
    information set; vintage/release-origin policies are a separate layer.
    """
    if (not isinstance(targets, (list, tuple)) or not 1 <= len(targets) <= 512
            or any(not isinstance(v, (list, tuple)) or len(v) != 2 or type(v[0]) is not int
                   or not isinstance(v[1], str) or len(v[1]) > 200 for v in targets)):
        raise AnalysisError("invalid_prediction", "Supply bounded resident (date position, measurement name) target pairs.")
    value = result.attrs["dynamic_factor_result"] if isinstance(result, DynamicFactorResult) else result
    raw = _load_record(value)
    state, source = raw["state"], raw["source"]
    workspace = _prediction_admit(state, len(targets), parameter_uncertainty=parameter_uncertainty, max_work=max_work)
    restored = dfactor_restore(raw)
    record = restored.attrs["dynamic_factor_result"]
    state, source = record["state"], record["source"]
    n, r = len(state["y"]), state["factors"]
    cells = []
    for position, name in targets:
        if not 0 <= position < n or name not in source["responses"]:
            raise AnalysisError("invalid_prediction", "Nowcast target must name an original source date/measurement.")
        j = source["responses"].index(name)
        if state["y"][position][j] is not None:
            raise AnalysisError("invalid_prediction", "Nowcasts target unobserved source measurements only.")
        cells.append((position, j))
    if len(set(cells)) != len(cells):
        raise AnalysisError("invalid_prediction", "Nowcast target cells must be distinct.")
    y, model, theta = _replay_model(state)

    def target_means(z):
        values = model.values(z)
        posterior = factor_smooth(y, values, profile=model.profile)
        return torch.stack([values["d"][j]+values["Z"][j] @ posterior["smoothed"][t] for t, j in cells])

    with torch.no_grad():
        values = model.values(theta)
        A, L, H = (values[k] for k in ("T", "Z", "H"))
        output = state["output"]
        posterior = torch.tensor(output["smoothed_covariance"], dtype=FLOAT, device="cpu")
        filtered = torch.tensor(output["filtered_covariance"], dtype=FLOAT, device="cpu")
        prior = torch.tensor(output["prior_covariance"], dtype=FLOAT, device="cpu")
        gains = [torch.cholesky_solve(A @ filtered[t], _positive(prior[t+1], "factor predicted covariance")).T
                 for t in range(n-1)]
        means = target_means(theta)
        conditional = torch.zeros((len(cells), len(cells)), dtype=FLOAT, device="cpu")
        for i, (t, j) in enumerate(cells):
            for h in range(i, len(cells)):
                s, k = cells[h]
                low, high = min(t, s), max(t, s)
                mapping = torch.eye(r, dtype=FLOAT, device="cpu")
                for date in range(low, high):
                    mapping = mapping @ gains[date]
                cross = mapping @ posterior[high]
                if t > s:
                    cross = cross.T
                covariance = L[j] @ cross @ L[k]
                if i == h:
                    covariance = covariance+H[j, j]
                conditional[i, h] = conditional[h, i] = covariance
    delta = torch.zeros_like(conditional)
    if parameter_uncertainty:
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(target_means, theta)
        covariance = torch.tensor(state["inference"]["chart_covariance"], dtype=FLOAT, device="cpu")
        delta = _symmetrize(J @ covariance @ J.T).detach()
    labels = _restore_index(source["time"] if source["time"] is not None else source["index"], n).take([t for t, _ in cells])
    return _prediction_tables(means, conditional, delta, labels, [name for _, name in targets], source["alpha"],
        parameter_uncertainty=parameter_uncertainty, workspace=workspace, result_sha256=record["sha256"],
        target_positions=[t for t, _ in cells], conditioning="all saved observations; no release-vintage admission yet")
