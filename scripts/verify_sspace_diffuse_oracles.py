"""Execute actual native R/KFAS as a development-only exact-diffuse oracle.

No R/SciPy package is imported by the estimator runtime. This script executes
the complete KFAS package, retains exact versions/binary/input/code hashes,
and compares full state and disturbance moments rather than coefficients.
Install KFAS1.6.0/jsonlite in a separate R library before running this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from openecon.econometrics.tsworkflows.ssdiffuse import (
    MAX_STATE_BYTES, RESULT_SCHEMA, SCHEMA, _digest, _load_record, smooth_exact_diffuse,
)


def fixtures():
    scalar = dict(a0=[.3], P_inf=[[1.]], P_star=[[0.]], Z=[[1.]], T=[[1.]],
                  Q=[[.25]], H=[[.6]], c=[.1], d=[-.2])
    # KFAS's documented common domain: P1inf is diagonal 0/1 and P1
    # rows/columns belonging to diffuse states are exactly zero. General
    # rotated/scaled diffuse priors are covered by independent GLS tests.
    common = dict(a0=[.3, -.4], P_inf=[[1., 0.], [0., 0.]], P_star=[[0., 0.], [0., .4]],
                  Z=[[1., .35], [-.25, 1.]], T=[[.9, .1], [0., .8]],
                  Q=[[.2, .04], [.04, .12]], H=[[.5, .18], [.18, .4]], c=[.05, -.01], d=[.03, -.02])
    cases = [dict(name="local_level", y=[[1.2], [1.8], [.7], [2.1], [1.5]], system=scalar),
             dict(name="leading_and_interior_missing", y=[[None], [None], [1.8], [.7], [None], [1.5]], system=scalar),
             dict(name="correlated_measurements_partial_diffuse", y=[[1., -.2], [None, None], [.3, .8], [None, .4]], system=common),
             dict(name="correlated_measurements_full_diffuse", y=[[1., None], [None, None], [.3, .8], [-.4, .4]],
                  system=dict(common, P_inf=[[1., 0.], [0., 1.]], P_star=[[0., 0.], [0., 0.]])),
             dict(name="diffuse_unit_root_level_slope", y=[[1.], [None], [1.4], [2.1], [2.2], [None], [2.5]],
                  system=dict(a0=[0., 0.], P_inf=[[1., 0.], [0., 1.]], P_star=[[0., 0.], [0., 0.]],
                              Z=[[1., 0.]], T=[[1., 1.], [0., 1.]], Q=[[.1, 0.], [0., .03]], H=[[.4]], c=[0., 0.], d=[0.]))]
    n = 5
    scheduled = dict(common)
    scheduled.update(Z=[[[1., .35+.02*t], [-.25, 1.]] for t in range(n)],
                     T=[[[.9, .1], [0., .75+.02*t]] for t in range(n)],
                     H=[[[.5, .18], [.18, .4+.03*t]] for t in range(n)],
                     Q=[[[.2+.01*t, .04], [.04, .12]] for t in range(n)],
                     c=[[.02*t, -.01] for t in range(n)], d=[[.03, -.02*t] for t in range(n)])
    cases.append(dict(name="scheduled_correlated_missing", y=[[1., -.2], [None, None], [.3, .8], [None, .4], [.2, -.7]], system=scheduled))
    return cases


def _r_vector(value):
    return "c("+",".join("NA_real_" if v is None else format(float(v), ".17g") for v in value)+")"


def _r_matrix(value):
    return f"matrix({_r_vector([v for row in value for v in row])},nrow={len(value)},byrow=TRUE)"


def r_code(cases, library):
    lines = [f".libPaths(c({json.dumps(str(library))},.libPaths()))", "library(KFAS)", "library(jsonlite)",
             "if(as.character(packageVersion('KFAS'))!='1.6.0') stop('KFAS version must be1.6.0')",
             "pack <- function(x) list(shape=if(is.null(dim(x))) length(x) else dim(x),values=as.vector(x))",
             "cases <- list()"]
    for i, case in enumerate(cases):
        n, p, m = len(case["y"]), len(case["y"][0]), len(case["system"]["a0"])
        values = case["system"]
        lines += [f"y <- {_r_matrix(case['y'])}", f"Z <- array(0,c({p},{m+1},{n}))",
                  f"T <- array(0,c({m+1},{m+1},{n}))", f"Q <- array(0,c({m+1},{m+1},{n}))",
                  f"H <- array(0,c({p},{p},{n}))"]
        for t in range(n):
            resolved = {}
            for key in ("Z", "T", "Q", "H", "c", "d"):
                a = np.array(values[key])
                resolved[key] = a[t].tolist() if a.ndim == (2 if key in {"c", "d"} else 3) else a.tolist()
            # Known equation offsets are an exactly deterministic constant state.
            lines += [f"Z[,,{t+1}] <- cbind({_r_matrix(resolved['Z'])},{_r_vector(resolved['d'])})",
                      f"T[1:{m},1:{m},{t+1}] <- {_r_matrix(resolved['T'])}",
                      f"T[1:{m},{m+1},{t+1}] <- {_r_vector(resolved['c'])}", f"T[{m+1},{m+1},{t+1}] <- 1",
                      f"Q[1:{m},1:{m},{t+1}] <- {_r_matrix(resolved['Q'])}", f"H[,,{t+1}] <- {_r_matrix(resolved['H'])}"]
        lines += [f"P1 <- matrix(0,{m+1},{m+1}); P1[1:{m},1:{m}] <- {_r_matrix(values['P_star'])}",
                  f"P1inf <- matrix(0,{m+1},{m+1}); P1inf[1:{m},1:{m}] <- {_r_matrix(values['P_inf'])}",
                  f"a1 <- {_r_vector([*values['a0'],1.])}",
                  f"model <- SSModel(y ~ -1 + SSMcustom(Z=Z,T=T,R=diag({m+1}),Q=Q,a1=a1,P1=P1,P1inf=P1inf),H=H,tol=1e-12)",
                  "stopifnot(identical(unname(model$P1inf),unname(P1inf)),identical(unname(model$P1),unname(P1)))",
                  "stopifnot(identical(unname(model$Z),unname(Z)),identical(unname(model$T),unname(T)),identical(unname(model$Q),unname(Q)),identical(unname(model$H),unname(H)))",
                  "warnings <- character()",
                  "fit <- withCallingHandlers(KFS(model,transform='augment',filtering='state',smoothing=c('state','disturbance')),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')})",
                  "if(length(warnings)) stop(paste(warnings,collapse=';'))",
                  f"cases[[{i+1}]] <- list(name={json.dumps(case['name'])},logLik=as.numeric(fit$logLik),",
                  " state_names=colnames(fit$alphahat),",
                  " model=lapply(model[c('Z','T','Q','H','a1','P1','P1inf')],pack),",
                  " moments=lapply(fit[intersect(c('a','P','Pinf','att','Ptt','alphahat','V','etahat','V_eta','epshat','V_eps','Finf','F','v'),names(fit))],pack))"]
    lines += ["receipt <- list(R=R.version.string,KFAS=as.character(packageVersion('KFAS')),license=packageDescription('KFAS')$License,",
              "binary=system.file('libs','KFAS.so',package='KFAS'),cases=cases)",
              "cat(toJSON(receipt,auto_unbox=TRUE,digits=17,na='null'))"]
    return "\n".join(lines)+"\n"


def array(moment):
    shape = moment["shape"] if isinstance(moment["shape"], list) else [moment["shape"]]
    return np.array(moment["values"], dtype=float).reshape(shape, order="F")


def compare(reference, cases, *, cached_outputs=None):
    reports = []
    for index, (case, other) in enumerate(zip(cases, reference["cases"], strict=True)):
        assert case["name"] == other["name"]
        y = torch.tensor([[float("nan") if v is None else v for v in row] for row in case["y"]], dtype=torch.float64)
        values = {key: torch.tensor(value, dtype=torch.float64) for key, value in case["system"].items()}
        # --state acceptance must evaluate the actual saved native output,
        # never regenerate a substitute with the source estimator.
        actual = (smooth_exact_diffuse(y, values) if cached_outputs is None else cached_outputs[index])
        n, p, m = len(y), y.shape[1], len(values["a0"])
        moments = other["moments"]
        augmented = array(moments["alphahat"]).shape[1] > m+1
        measurement_mean = (array(moments["alphahat"])[:, m+1:] if augmented
                            else array(moments["epshat"]).reshape(n, p))
        measurement_variance = (array(moments["V"])[m+1:, m+1:].transpose(2, 0, 1) if augmented
                                else array(moments["V_eps"]).reshape(p, n).T[:, :, None])
        pairs = {"smoothed": array(moments["alphahat"])[:, :m],
                 "smoothed_covariance": array(moments["V"])[:m, :m].transpose(2, 0, 1),
                 "filtered": array(moments["att"])[:, :m],
                 "filtered_covariance": array(moments["Ptt"])[:m, :m].transpose(2, 0, 1),
                 "prior_mean": array(moments["a"])[:n, :m],
                 "prior_covariance": array(moments["P"])[:m, :m, :n].transpose(2, 0, 1),
                 "prior_diffuse_covariance": array(moments["Pinf"])[:m, :m, :n].transpose(2, 0, 1),
                 "process_disturbance": array(moments["etahat"])[:, :m],
                 "process_disturbance_covariance": array(moments["V_eta"])[:m, :m].transpose(2, 0, 1),
                 "measurement_disturbance": measurement_mean,
                 "measurement_disturbance_covariance": measurement_variance}
        pairs["next_mean"] = array(moments["a"])[n, :m]
        pairs["next_covariance"] = array(moments["P"])[:m, :m, n]
        if augmented:
            # The appended states are the ORIGINAL correlated measurement
            # shocks, including conditional means in unobserved cells. LDL
            # output epshat instead describes transformed observed shocks.
            pairs["measurement_state_covariance"] = array(moments["V"])[m+1:, :m].transpose(2, 0, 1)
        if "F" in moments:
            finite, residual = array(moments["F"]), array(moments["v"]).T
            infinity = array(moments["Finf"]).reshape(p, -1)
            count, contribution = np.zeros(n, dtype=int), np.zeros(n)
            for t in range(n):
                for j in range(p):
                    if np.isnan(case["y"][t][j] if case["y"][t][j] is not None else math.nan):
                        continue
                    infinite = infinity[j, t] if t < infinity.shape[1] else 0.
                    if infinite > 0:
                        count[t] += 1
                        contribution[t] -= .5*math.log(infinite)
                    else:
                        contribution[t] -= .5*(math.log(2*math.pi)+math.log(finite[j, t])+residual[j, t]**2/finite[j, t])
            pairs["log_likelihood_contributions"] = contribution
            pairs["diffuse_observation_count"] = count
            pairs["observed_mask"] = ~np.isnan(y.numpy())
            pairs["observed_count"] = (~np.isnan(y.numpy())).sum(1)
            pairs["proper_observation_count"] = (~np.isnan(y.numpy())).sum(1)-count
            initial_rank = int(np.diag(case["system"]["P_inf"]).sum())
            pairs["initial_diffuse_rank"] = initial_rank
            pairs["prior_diffuse_rank"] = initial_rank-np.concatenate(([0], np.cumsum(count)[:-1]))
            pairs["filtered_diffuse_rank"] = initial_rank-np.cumsum(count)
            pairs["final_diffuse_rank"] = initial_rank-int(count.sum())
        errors = {}
        for name, expected in pairs.items():
            observed = (actual[name].detach().numpy() if isinstance(actual[name], torch.Tensor)
                        else np.asarray(actual[name], dtype=float))
            # KFAS Pinf retains only the diffuse phase; append known exact zeros.
            if name == "prior_diffuse_covariance" and len(expected) < n:
                expected = np.concatenate((expected, np.zeros((n-len(expected), m, m))))
            np.testing.assert_allclose(observed, expected, atol=2e-10, rtol=2e-10, err_msg=case["name"]+":"+name)
            errors[name] = float(np.max(np.abs(observed.astype(float)-np.asarray(expected, dtype=float))))
        np.testing.assert_allclose(float(actual["log_likelihood"]), other["logLik"], atol=2e-10, rtol=2e-10)
        reports.append(dict(name=case["name"], status="passed", maximum_absolute_errors=errors,
                            likelihood_error=abs(float(actual["log_likelihood"])-other["logLik"])))
    return reports


def independent_cached(case, cached):
    """Primitive-noise expansion + GLS; compare saved cross moments directly."""
    y = np.asarray([[math.nan if v is None else v for v in row] for row in case["y"]])
    v = {key: np.asarray(value, dtype=float) for key, value in case["system"].items()}
    n, p = y.shape
    m = len(v["a0"])
    def at(key, t):
        return v[key][t] if v[key].ndim == (2 if key in {"c", "d"} else 3) else v[key]
    # This author-reference overlap is explicitly diagonal 0/1 diffuse.
    initial = np.eye(m)[:, np.diag(v["P_inf"]) == 1]
    r, width = initial.shape[1], (n+1)*m
    primitive = np.zeros((width, width))
    primitive[:m, :m] = v["P_star"]
    expansion = np.zeros((width, width))
    expansion[:m, :m] = np.eye(m)
    means, loadings = [v["a0"]], [initial]
    for t in range(n):
        sl = slice((t+1)*m, (t+2)*m)
        primitive[sl, sl] = at("Q", t)
        expansion[sl] = at("T", t)@expansion[t*m:(t+1)*m]
        expansion[sl, sl] = np.eye(m)
        means.append(at("T", t)@means[-1]+at("c", t))
        loadings.append(at("T", t)@loadings[-1])
    G = expansion@primitive@expansion.T
    mean, A = np.concatenate(means), np.concatenate(loadings)
    positions = np.flatnonzero(~np.isnan(y.ravel()))
    L, H, d = np.zeros((len(positions), width)), np.zeros((n*p, n*p)), np.zeros(n*p)
    for t in range(n):
        H[t*p:(t+1)*p, t*p:(t+1)*p] = at("H", t)
        d[t*p:(t+1)*p] = at("d", t)
    for i, position in enumerate(positions):
        t, j = divmod(position, p)
        L[i, t*m:(t+1)*m] = at("Z", t)[j]
    omega, D, C = L@G@L.T+H[positions][:, positions], L@A, G@L.T
    error = y.ravel()[positions]-L@mean-d[positions]
    saddle = np.block([[omega, D], [D.T, np.zeros((r, r))]])
    inverse_saddle = np.linalg.inv(saddle)
    state_bridge = np.concatenate((C, A), axis=1)
    try:
        inverse = np.linalg.inv(omega)
    except np.linalg.LinAlgError:
        # Exact zero finite-noise laws remain valid if diffuse constraints
        # identify them; the full conditioning equations are invertible.
        posterior_mean = mean+state_bridge@inverse_saddle@np.concatenate((error, np.zeros(r)))
        covariance = G-state_bridge@inverse_saddle@state_bridge.T
        inverse = None
    if inverse is not None and r:
        information = D.T@inverse@D
        B = np.linalg.inv(information)
        beta = B@D.T@inverse@error
        bridge = A-C@inverse@D
        posterior_mean = mean+A@beta+C@inverse@(error-D@beta)
        covariance = G-C@inverse@C.T+bridge@B@bridge.T
    elif inverse is not None:
        posterior_mean, covariance = mean+C@inverse@error, G-C@inverse@C.T
    noise_bridge = np.concatenate((H[:, positions], np.zeros((n*p, r))), axis=1)
    eps_mean = (noise_bridge@inverse_saddle@np.concatenate((error, np.zeros(r)))).reshape(n, p)
    eps_covariance = H-noise_bridge@inverse_saddle@noise_bridge.T
    eps_state = -noise_bridge@inverse_saddle@state_bridge.T
    mu, V = posterior_mean.reshape(n+1, m), covariance.reshape(n+1, m, n+1, m)
    expected = dict(smoothed=mu[:-1], smoothed_joint_covariance=V,
                    smoothed_next_mean=mu[-1], smoothed_next_covariance=V[-1, :, -1, :],
                    lag_one_covariance=np.array([V[t+1, :, t, :] for t in range(n-1)]).reshape(n-1, m, m))
    process, process_variance, process_state, eps_variance, eps_cross, noise_cross, joint_noise = [], [], [], [], [], [], []
    for t in range(n):
        T = at("T", t)
        lag = V[t+1, :, t, :]
        variance = V[t+1, :, t+1, :]+T@V[t, :, t, :]@T.T-lag@T.T-T@lag.T
        measurement_variance = eps_covariance[t*p:(t+1)*p, t*p:(t+1)*p]
        between = (eps_state[t*p:(t+1)*p, (t+1)*m:(t+2)*m]-eps_state[t*p:(t+1)*p, t*m:(t+1)*m]@T.T).T
        process.append(mu[t+1]-T@mu[t]-at("c", t))
        process_variance.append(variance)
        process_state.append(lag-T@V[t, :, t, :])
        eps_variance.append(measurement_variance)
        eps_cross.append(eps_state[t*p:(t+1)*p, t*m:(t+1)*m])
        noise_cross.append(between)
        joint_noise.append(np.block([[variance, between], [between.T, measurement_variance]]))
    expected.update(process_disturbance=np.array(process), process_disturbance_covariance=np.array(process_variance),
                    process_state_covariance=np.array(process_state), measurement_disturbance=eps_mean,
                    measurement_disturbance_covariance=np.array(eps_variance), measurement_state_covariance=np.array(eps_cross),
                    process_measurement_covariance=np.array(noise_cross), joint_disturbance_covariance=np.array(joint_noise))
    errors = {}
    for name, target in expected.items():
        actual = np.asarray(cached[name], dtype=float)
        np.testing.assert_allclose(actual, target, atol=2e-10, rtol=2e-10, err_msg=case["name"]+":cached GLS:"+name)
        errors[name] = float(np.max(np.abs(actual-target))) if actual.size else 0.
    return dict(status="passed", comparison_target="actual cached native output; independent NumPy primitive-noise GLS",
                maximum_absolute_errors=errors)


def saved_case(path):
    with path.open("rb") as source:
        raw = source.read(MAX_STATE_BYTES+1)
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError("Actual saved native JSON exceeds 64 MiB")
    record = _load_record(raw.decode("utf-8"))
    assert record["schema"] == RESULT_SCHEMA
    assert record["sha256"] == _digest({k: v for k, v in record.items() if k != "sha256"})
    state = record["state"]
    assert state["schema"] == SCHEMA and state["smoothing"] is True
    assert state["sha256"] == _digest({k: v for k, v in state.items() if k != "sha256"})
    return (dict(name="actual_saved_native_result", y=state["y"], system=state["system"]),
            state["output"], hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--r-library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--state", type=Path, help="Compare this ACTUAL saved public result's cached outputs without estimator regeneration.")
    args = parser.parse_args()
    cached = None
    if args.state:
        case, output, saved_sha256 = saved_case(args.state)
        cases, cached = [case], [output]
    else:
        cases = fixtures()
    code = r_code(cases, args.r_library)
    with tempfile.TemporaryDirectory(prefix="openecon-diffuse-author-") as work:
        source = Path(work)/"reference.R"
        source.write_text(code)
        run = subprocess.run([args.rscript, "--vanilla", str(source)], capture_output=True, text=True, check=True)
    reference = json.loads(run.stdout)
    binary = Path(reference["binary"])
    reference.update(reference_kind="actual native R/KFAS package execution; supplied deterministic synthetic fixed systems",
                     common_domain="KFAS P1inf is diagonal 0/1; P1 rows/columns for diffuse states are zero; constructed model equations checked identical before KFS",
                     measurement_reference="KFS transform='augment': original correlated measurement shocks are appended states, including missing cells, full covariance and state cross covariance",
                     source_package="https://github.com/cran/KFAS/tree/66aab472a9d29c08c52a8ebcd3da531c9daadba3",
                     binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                     rscript_sha256=hashlib.sha256(Path(args.rscript).resolve().read_bytes()).hexdigest(),
                     setup_sha256=hashlib.sha256(code.encode()).hexdigest(), setup_code=code,
                     inputs=cases, input_sha256=hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                     stdout_stderr=run.stderr)
    # Keep actual raw author outputs even if comparison fails; never normalize
    # a failed external run into a passing fixture.
    args.report.write_text(json.dumps(reference, indent=2)+"\n")
    reference["comparisons"] = compare(reference, cases, cached_outputs=cached)
    if args.state:
        reference["saved_native_state_sha256"] = saved_sha256
        reference["comparison_target"] = "actual saved native result cached moments; no production estimator regeneration"
        reference["cached_full_cross_moments"] = independent_cached(cases[0], cached[0])
    reference["status"] = "passed"
    args.report.write_text(json.dumps(reference, indent=2)+"\n")
    print(f"KFAS1.6.0: {len(cases)} complete numerical cases passed")


if __name__ == "__main__":
    main()
