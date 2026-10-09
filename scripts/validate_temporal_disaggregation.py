"""Independent constrained-quadratic/GLS references and fixed Gaussian simulations.

The scientific oracle imports no OpenEconometrics implementation helper. KKT
constraints and whitened NumPy QR are separate from the Torch runtime solvers.
Public basis fits identify its linear maps; batched simulations then use these
publicly measured maps, not 48,000 repeated rich-table API calls.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import shutil

import numpy as np
from scipy import stats
import torch

import openecon as oe

SEEDS = (731159, 910287)
REPLICATIONS = 1000
AGGREGATIONS = ("sum", "mean", "first", "last")
ROUTES = (("Y", "Q", 4), ("Y", "M", 12), ("Q", "M", 3))
DENTON = (
    ("denton_additive", 1, False, False),
    ("denton_additive_second", 2, False, False),
    ("denton_proportional", 1, True, False),
    ("denton_proportional_second", 2, True, False),
    ("denton_cholette", 1, False, True),
    ("denton_cholette", 1, True, True),
)


def calendar(m, low_frequency="Y", high_frequency="Q"):
    if low_frequency == "Y":
        low = [str(2000 + i) for i in range(m)]
        if high_frequency == "Q":
            high = [f"{2000+i//4}Q{i%4+1}" for i in range(4*m)]
        else:
            high = [f"{2000+i//12}-{i%12+1:02d}" for i in range(12*m)]
    else:
        low = [f"{2000+i//4}Q{i%4+1}" for i in range(m)]
        high = [f"{2000+i//12}-{i%12+1:02d}" for i in range(3*m)]
    return dict(low_periods=low, high_periods=high,
                low_frequency=low_frequency, high_frequency=high_frequency)


def conversion(m, ratio, aggregation):
    c = np.zeros((m, m*ratio))
    for i in range(m):
        for j in range(ratio):
            if aggregation == "sum":
                c[i, ratio*i+j] = 1.
            elif aggregation == "mean":
                c[i, ratio*i+j] = 1./ratio
            elif aggregation == "first" and j == 0:
                c[i, ratio*i+j] = 1.
            elif aggregation == "last" and j == ratio-1:
                c[i, ratio*i+j] = 1.
    return c


def kkt_reference(low, indicator, c, order, proportional, unanchored):
    """Difference each padded coordinate basis; solve the full quadratic KKT."""
    n, m = c.shape[1], c.shape[0]
    identity = np.eye(n)
    operator = np.diff(np.pad(identity, ((order, 0), (0, 0))), n=order, axis=0)
    if unanchored:
        operator = operator[order:]
    units = indicator if proportional else np.ones(n)
    b = c*units
    q = operator.T @ operator
    k = np.block([[q, b.T], [b, np.zeros((m, m))]])
    solution = np.linalg.solve(k, np.r_[np.zeros(n), low-c@indicator])
    value = indicator + units*solution[:n]
    stationarity = q@solution[:n] + b.T@solution[n:]
    return value, float(np.max(np.abs(stationarity)))


def covariance_shape(method, n, rho):
    """Independent process recursion, with explicitly declared initial states."""
    if method == "chow_lin":
        return rho**np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])/(1-rho*rho)
    innovation_map = np.zeros((n, n))
    current_increment = np.zeros(n)
    current_level = np.zeros(n)
    for i in range(n):
        current_increment = rho*current_increment
        current_increment[i] += 1.
        current_level = current_level + current_increment
        innovation_map[i] = current_level
    return innovation_map @ innovation_map.T


def gls_reference(low, supplied, c, q, alpha=.05):
    x = np.column_stack((np.ones(len(supplied)), supplied))
    xl = c@x
    w = c@q@c.T
    chol = np.linalg.cholesky(w)
    a = np.linalg.solve(chol, xl)
    target = np.linalg.solve(chol, low)
    orthogonal, triangular = np.linalg.qr(a, mode="reduced")
    beta = np.linalg.solve(triangular, orthogonal.T@target)
    v = np.linalg.solve(triangular, np.eye(triangular.shape[0]))
    v = v@v.T
    residual = low-xl@beta
    sse = float(np.linalg.solve(chol, residual)@np.linalg.solve(chol, residual))
    df = len(low)-x.shape[1]
    sigma2 = sse/df
    d = np.linalg.solve(w, c@q).T
    h = x-d@xl
    value = x@beta+d@residual
    high_shape = q-d@c@q+h@v@h.T
    high_shape = (high_shape+high_shape.T)/2
    high_covariance = sigma2*high_shape
    coefficient_covariance = sigma2*v
    critical = stats.t.ppf(1-alpha/2, df)
    se = np.sqrt(np.maximum(np.diag(high_covariance), 0))
    beta_se = np.sqrt(np.diag(coefficient_covariance))
    mle_scale = sse/len(low)
    loglik = -.5*(len(low)*(math.log(2*math.pi*mle_scale)+1)+np.linalg.slogdet(w)[1])
    beta_map = np.linalg.solve(triangular, orthogonal.T@np.linalg.solve(chol, np.eye(len(low))))
    value_map = x@beta_map+d@(np.eye(len(low))-xl@beta_map)
    return dict(beta=beta, value=value, beta_covariance=coefficient_covariance,
                high_covariance=high_covariance, beta_shape=v, high_shape=high_shape,
                sigma2=sigma2, sse=sse, df=df, loglik=loglik,
                lower=value-critical*se, upper=value+critical*se,
                beta_se=beta_se, beta_t=beta/beta_se,
                beta_p=2*stats.t.sf(np.abs(beta/beta_se), df),
                beta_lower=beta-critical*beta_se, beta_upper=beta+critical*beta_se,
                residual=residual, low_fitted=xl@beta,
                se=se, beta_map=beta_map, value_map=value_map,
                x=x, xl=xl, w=w)


def field(result, table, column):
    return result[table][column].to_numpy(dtype=float)


def compare(actual, expected, *, rtol=2e-7, atol=2e-8):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))


def gls_call(method, low, supplied, periods, aggregation, rho):
    options = dict(periods, aggregation=aggregation)
    if method != "fernandez":
        options["rho"] = rho
    return getattr(oe, method)(low.tolist(), supplied.tolist(), **options)


def deterministic_checks():
    rows = []
    for low_frequency, high_frequency, ratio in ROUTES:
        m = 7
        periods = calendar(m, low_frequency, high_frequency)
        n = m*ratio
        t = np.arange(n, dtype=float)
        indicator = 80+1.5*t+12*np.sin(.31*t)
        latent = indicator+7*np.cos(.18*t)+3*np.sin(.8*t)
        supplied = np.column_stack((t/n, np.cos(.31*t)+.3*np.sin(.19*t)))
        for aggregation in AGGREGATIONS:
            c = conversion(m, ratio, aggregation)
            low = c@latent
            for method, order, proportional, unanchored in DENTON:
                options = dict(periods, aggregation=aggregation)
                if unanchored:
                    options["criterion"] = "proportional" if proportional else "additive"
                result = getattr(oe, method)(low.tolist(), indicator.tolist(), **options)
                expected, stationarity = kkt_reference(low, indicator, c, order, proportional, unanchored)
                error = compare(field(result, "series", "value"), expected)
                restriction = compare(c@field(result, "series", "value"), low)
                rows.append(dict(method=method, route=f"{low_frequency}->{high_frequency}",
                                 aggregation=aggregation, proportional=proportional,
                                 maximum_value_error=error, maximum_reaggregation_error=restriction,
                                 oracle_stationarity_error=stationarity, passed=True))
            for method in ("chow_lin", "fernandez", "litterman"):
                for rho in ((0.,) if method == "fernandez" else (-.45, 0., .65)):
                    q = covariance_shape(method, n, rho)
                    result = gls_call(method, low, supplied, periods, aggregation, rho)
                    expected = gls_reference(low, supplied, c, q)
                    errors = {
                        "value": compare(field(result, "series", "value"), expected["value"]),
                        "beta": compare(field(result, "coefficients", "coefficient"), expected["beta"]),
                        "coefficient_covariance": compare(result["coefficient_covariance"].to_numpy(), expected["beta_covariance"]),
                        "high_covariance": compare(result["high_covariance"].to_numpy(), expected["high_covariance"]),
                        "ci_lower": compare(field(result, "latent_intervals", "ci_lower"), expected["lower"]),
                        "ci_upper": compare(field(result, "latent_intervals", "ci_upper"), expected["upper"]),
                        "coefficient_se": compare(field(result, "coefficients", "standard_error"), expected["beta_se"]),
                        "coefficient_t": compare(field(result, "coefficients", "t"), expected["beta_t"]),
                        "coefficient_p": compare(field(result, "coefficients", "p_value"), expected["beta_p"]),
                        "coefficient_ci_lower": compare(field(result, "coefficients", "ci_lower"), expected["beta_lower"]),
                        "coefficient_ci_upper": compare(field(result, "coefficients", "ci_upper"), expected["beta_upper"]),
                        "latent_se": compare(field(result, "latent_intervals", "standard_error"), expected["se"], atol=3e-6),
                        "low_fitted": compare(field(result, "low_fit", "fitted"), expected["low_fitted"]),
                        "low_residual": compare(field(result, "low_fit", "residual"), expected["residual"]),
                    }
                    fit = result["fit"].iloc[0]
                    errors["sigma2"] = compare(fit.sigma2, expected["sigma2"])
                    errors["sse"] = compare(fit.weighted_sse, expected["sse"])
                    errors["loglik"] = compare(fit.log_likelihood, expected["loglik"])
                    assert int(fit.df_residual) == expected["df"]
                    errors["aggregation"] = compare(c@field(result, "series", "value"), low)
                    errors["aggregate_covariance"] = compare(c@result["high_covariance"].to_numpy()@c.T, np.zeros((m, m)), atol=1e-6)
                    rows.append(dict(method=method, rho=rho, route=f"{low_frequency}->{high_frequency}",
                                     aggregation=aggregation, errors=errors, df=expected["df"], passed=True))
    assert len(rows) == 156
    return rows


def public_maps(method, supplied, c, periods, aggregation, rho):
    """Measure linear maps using complete benchmark vectors, without private state."""
    m = len(c)
    beta_columns, value_columns = [], []
    beta_shape = high_shape = None
    for i in range(m):
        basis = np.eye(m)[i]
        result = gls_call(method, basis, supplied, periods, aggregation, rho)
        beta_columns.append(field(result, "coefficients", "coefficient"))
        value_columns.append(field(result, "series", "value"))
        if i == 0:
            scale = float(result["fit"].iloc[0].sigma2)
            assert scale > 0
            beta_shape = result["coefficient_covariance"].to_numpy()/scale
            high_shape = result["high_covariance"].to_numpy()/scale
    return np.column_stack(beta_columns), np.column_stack(value_columns), beta_shape, high_shape


def simulation_checks():
    rows = []
    maps = {}
    alpha = .05
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for regime in (0, 1):
            m = 12 if regime == 0 else 18
            periods = calendar(m)
            n = 4*m
            t = np.arange(n, dtype=float)
            supplied = np.column_stack((t/n, np.cos(.31*t)+.3*np.sin(.19*t)))
            x = np.column_stack((np.ones(n), supplied))
            true_beta = np.array([2., .5, -.3])
            sigma2 = .49
            for method in ("chow_lin", "fernandez", "litterman"):
                rho = 0. if method == "fernandez" else (-.45 if regime == 0 else .65)
                q = covariance_shape(method, n, rho)
                innovation = rng.normal(size=(REPLICATIONS, n))
                high = x@true_beta+math.sqrt(sigma2)*innovation@np.linalg.cholesky(q).T
                for aggregation in AGGREGATIONS:
                    c = conversion(m, 4, aggregation)
                    low = high@c.T
                    key = (regime, method, aggregation)
                    if key not in maps:
                        maps[key] = public_maps(method, supplied, c, periods, aggregation, rho)
                    beta_map, value_map, beta_shape, high_shape = maps[key]
                    oracle = gls_reference(low[0], supplied, c, q)
                    compare(beta_map, oracle["beta_map"])
                    compare(value_map, oracle["value_map"])
                    compare(beta_shape, oracle["beta_shape"])
                    compare(high_shape, oracle["high_shape"])
                    beta = low@beta_map.T
                    prediction = low@value_map.T
                    residual = low-beta@(c@x).T
                    whitened = np.linalg.solve(np.linalg.cholesky(c@q@c.T), residual.T).T
                    scales = np.sum(whitened*whitened, axis=1)/(m-x.shape[1])
                    beta_se = np.sqrt(scales[:, None]*np.diag(beta_shape))
                    critical = stats.t.ppf(1-alpha/2, m-x.shape[1])
                    beta_cover = np.mean(np.abs(beta-true_beta) <= critical*beta_se, axis=0)
                    unconstrained = np.flatnonzero(np.diag(high_shape) > 1e-8)
                    chosen = unconstrained[np.linspace(0, len(unconstrained)-1, 5, dtype=int)]
                    high_se = np.sqrt(scales[:, None]*np.diag(high_shape)[chosen])
                    errors = prediction[:, chosen]-high[:, chosen]
                    high_cover = np.mean(np.abs(errors) <= critical*high_se, axis=0)
                    expected_beta_variance = sigma2*np.diag(beta_shape)
                    expected_high_variance = sigma2*np.diag(high_shape)[chosen]
                    beta_ratio = np.var(beta, axis=0, ddof=1)/expected_beta_variance
                    high_ratio = np.var(errors, axis=0, ddof=1)/expected_high_variance
                    coverage_gate = 4*math.sqrt(.95*.05/REPLICATIONS)+.008
                    variance_gate = 4*math.sqrt(2/(REPLICATIONS-1))+.02
                    coverage_ok = bool(np.all(np.abs(np.r_[beta_cover, high_cover]-.95) <= coverage_gate))
                    variance_ok = bool(np.all(np.abs(np.r_[beta_ratio, high_ratio]-1) <= variance_gate))
                    aggregation_error = float(np.max(np.abs(prediction@c.T-low)))
                    scale_ratio = float(np.mean(scales)/sigma2)
                    scale_gate = 4*math.sqrt(2/(REPLICATIONS*(m-x.shape[1])))+.01
                    row = dict(seed=seed, regime=regime, method=method, rho=rho,
                               aggregation=aggregation, planned=REPLICATIONS, completed=REPLICATIONS,
                               failed=0, n_low=m, n_high=n, df=m-x.shape[1],
                               coefficient_coverage=beta_cover.tolist(), high_periods_checked=[periods["high_periods"][i] for i in chosen],
                               latent_pointwise_coverage=high_cover.tolist(), coverage_gate=coverage_gate,
                               coefficient_variance_ratios=beta_ratio.tolist(), latent_error_variance_ratios=high_ratio.tolist(),
                               variance_ratio_gate=variance_gate, mean_variance_scale_ratio=scale_ratio,
                               mean_variance_scale_gate=scale_gate, maximum_aggregation_error=aggregation_error,
                               passed=coverage_ok and variance_ok and abs(scale_ratio-1) <= scale_gate and aggregation_error < 1e-8)
                    rows.append(row)
    assert len(rows) == 48
    return rows, sum((12 if key[0] == 0 else 18) for key in maps)


def run():
    deterministic = deterministic_checks()
    simulations, public_basis_calls = simulation_checks()
    return dict(status="passed" if all(row["passed"] for row in simulations) else "failed",
                oracle_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=version("scipy"), torch=torch.__version__, openecon=version("openecon")),
                deterministic_cases=156, deterministic=deterministic,
                seeds=list(SEEDS), replications_per_cell=REPLICATIONS,
                planned_cells=48, completed_cells=len(simulations),
                planned_replications=48000, completed_replications=48000,
                failed_replications=0, public_basis_fit_calls=public_basis_calls,
                complete_public_api_calls=156+public_basis_calls,
                simulation_execution="public API basis maps measured once per design; batch NumPy applications, not 48000 rich-table API calls",
                simulations=simulations,
                reference_urls=["https://journal.r-project.org/articles/RJ-2013-028/", "https://cynkra.github.io/tempdisagg/reference/td.html"],
                reference_package_source_version="tempdisagg 1.2.0; mathematical source inspection only, no GPL runtime code copied",
                rscript_available=shutil.which("Rscript") is not None, r_execution_verified=False,
                scope="complete regular calendars, exact known benchmarks, specified fixed Gaussian covariance and initial conditions; deterministic movement preservation",
                not_verified="estimated rho, non-Gaussian robustness, ragged calendars, forecasts, licensed vendor execution, Windows/CUDA or public release readiness")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    receipt = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: receipt[key] for key in ("status", "deterministic_cases", "completed_cells", "completed_replications", "public_basis_fit_calls")}))
    if receipt["status"] != "passed":
        raise SystemExit(1)
