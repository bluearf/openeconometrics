"""Compare an ACTUAL cached DFM result with native R and independent NumPy.

No estimator is called. The saved observations, chart, prior, equations and
cached outputs are the comparison targets. R/KFAS and optional dfms are
development references, never production dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from openecon.econometrics.tsworkflows.dfm import SCHEMA, RESULT_SCHEMA, _geometry
from openecon.econometrics.tsworkflows.ssdiffuse import _digest, _load_record, _raw_array


def numpy_filter(y, system):
    """Independent covariance Kalman recursion, including full innovations."""
    A, L, Q, H, d = [np.asarray(system[k]) for k in ("T", "Z", "Q", "H", "d")]
    mean, P = np.asarray(system["a0"]), np.asarray(system["P0"])
    n, p = y.shape
    r = len(mean)
    prior, prior_P, filtered, filtered_P, predictions, innovations, innovation_P, contributions = [], [], [], [], [], [], [], []
    for row in y:
        prior.append(mean.copy())
        prior_P.append(P.copy())
        prediction = L @ mean+d
        F_full = L @ P @ L.T+H
        present = np.isfinite(row)
        innovations.append(np.where(present, row-prediction, np.nan))
        predictions.append(prediction)
        innovation_P.append(F_full)
        Z = L[present]
        ll = 0.
        if present.any():
            residual = row[present]-prediction[present]
            noise = H[present][:, present]
            F = F_full[present][:, present]
            ll = -.5*(present.sum()*np.log(2*np.pi)+np.linalg.slogdet(F)[1]+residual @ np.linalg.solve(F, residual))
            gain = np.linalg.solve(F, Z @ P).T
            mean = mean+gain @ residual
            C = np.eye(r)-gain @ Z
            P = C @ P @ C.T+gain @ noise @ gain.T
        contributions.append(ll)
        filtered.append(mean.copy())
        filtered_P.append(P.copy())
        mean, P = A @ mean+system["c"], A @ P @ A.T+Q
    return dict(prior_mean=np.asarray(prior), prior_covariance=np.asarray(prior_P),
                filtered=np.asarray(filtered), filtered_covariance=np.asarray(filtered_P),
                predicted=np.asarray(predictions), innovation=np.asarray(innovations), innovation_covariance=np.asarray(innovation_P),
                log_likelihood_contributions=np.asarray(contributions), log_likelihood=np.sum(contributions),
                next_mean=mean, next_covariance=P, observed_mask=np.isfinite(y), observed_count=np.isfinite(y).sum(axis=1))


def primitive_conditioning(y, system):
    """Full cross-date primitive Gaussian prior; one observed-data solve."""
    n, p = y.shape
    r = len(system["a0"])
    if (n+1)*r > 1024 or np.isfinite(y).sum() > 1024 or n > 256:
        raise ValueError("Independent full Gaussian-conditioning oracle admits <=256 dates and <=1024 state/equation dimensions")
    A, L, Q, H, d, a0, P0 = [np.asarray(system[k]) for k in ("T", "Z", "Q", "H", "d", "a0", "P0")]
    means = np.zeros((n+1, r))
    means[0] = a0
    P = np.zeros(((n+1)*r, (n+1)*r))
    P[:r, :r] = P0
    for t in range(n):
        means[t+1] = A @ means[t]+system["c"]
        for s in range(t+1):
            block = A @ P[t*r:(t+1)*r, s*r:(s+1)*r]
            P[(t+1)*r:(t+2)*r, s*r:(s+1)*r] = block
            P[s*r:(s+1)*r, (t+1)*r:(t+2)*r] = block.T
        P[(t+1)*r:(t+2)*r, (t+1)*r:(t+2)*r] = A @ P[t*r:(t+1)*r, t*r:(t+1)*r] @ A.T+Q
    G = np.zeros((n*p, (n+1)*r))
    for t in range(n):
        G[t*p:(t+1)*p, t*r:(t+1)*r] = L
    present = np.isfinite(y).flatten()
    covariance = G @ P @ G.T+np.kron(np.eye(n), H)
    cross = P @ G.T[:, present]
    S = covariance[present][:, present]
    residual = (y-(G @ means.flatten()).reshape(n, p)-d).flatten()[present]
    posterior_mean = means.flatten()+cross @ np.linalg.solve(S, residual)
    posterior_P = P-cross @ np.linalg.solve(S, cross.T)
    state_means = posterior_mean.reshape(n+1, r)
    process, process_P, measurement, measurement_P, process_state, measurement_state, cross, joint = [], [], [], [], [], [], [], []
    for t in range(n):
        V = posterior_P[t*r:(t+1)*r, t*r:(t+1)*r]
        mapping = -L*np.isfinite(y[t])[:, None]
        error = np.where(np.isfinite(y[t]), y[t]-d-L @ state_means[t], 0.)
        error_P = mapping @ V @ mapping.T+np.diag(np.diag(H)*~np.isfinite(y[t]))
        if t < n-1:
            next_V = posterior_P[(t+1)*r:(t+2)*r, (t+1)*r:(t+2)*r]
            lag = posterior_P[(t+1)*r:(t+2)*r, t*r:(t+1)*r]
            disturbance = state_means[t+1]-A @ state_means[t]-system["c"]
            disturbance_P = next_V+A @ V @ A.T-lag @ A.T-A @ lag.T
            disturbance_state = lag-A @ V
        else:
            disturbance, disturbance_P, disturbance_state = np.zeros(r), Q, np.zeros((r,r))
        disturbance_error = disturbance_state @ mapping.T
        process.append(disturbance)
        process_P.append(disturbance_P)
        measurement.append(error)
        measurement_P.append(error_P)
        process_state.append(disturbance_state)
        measurement_state.append(mapping @ V)
        cross.append(disturbance_error)
        joint.append(np.block([[disturbance_P,disturbance_error],[disturbance_error.T,error_P]]))
    return dict(smoothed=state_means[:-1],
                smoothed_covariance=np.array([posterior_P[t*r:(t+1)*r, t*r:(t+1)*r] for t in range(n)]),
                lag_one_covariance=np.array([posterior_P[(t+1)*r:(t+2)*r, t*r:(t+1)*r] for t in range(n-1)]),
                smoothed_next_mean=posterior_mean[-r:], smoothed_next_covariance=posterior_P[-r:, -r:],
                process_disturbance=np.asarray(process),process_disturbance_covariance=np.asarray(process_P),
                measurement_disturbance=np.asarray(measurement),measurement_disturbance_covariance=np.asarray(measurement_P),
                process_state_covariance=np.asarray(process_state),measurement_state_covariance=np.asarray(measurement_state),
                process_measurement_covariance=np.asarray(cross),joint_disturbance_covariance=np.asarray(joint))


def physical_system(theta, state):
    p, r, anchors = len(state["y"][0]), state["factors"], state["anchors"]
    free = [i for i in range(p) if i not in anchors]
    offset = p
    L = np.zeros((p, r))
    L[anchors] = np.eye(r)
    L[free] = theta[offset:offset+len(free)*r].reshape(-1, r)
    offset += len(free)*r
    A = theta[offset:offset+r*r].reshape(r, r)
    offset += r*r
    C = np.zeros((r, r))
    for i in range(r):
        for j in range(i+1):
            C[i, j] = np.exp(theta[offset]) if i == j else theta[offset]
            offset += 1
    return dict(a0=np.asarray(state["prior"]["a0"]), P0=np.asarray(state["prior"]["P0"]),
                Z=L, T=A, Q=C @ C.T, H=np.diag(np.exp(theta[offset:])),
                c=np.zeros(r), d=theta[:p])


def finite_information(y, state, epsilon):
    point = np.asarray(state["theta"])
    if len(point) > 64:
        raise ValueError("Independent finite-difference OIM oracle admits <=64 parameters")
    def likelihood(theta):
        return numpy_filter(y, physical_system(theta, state))["log_likelihood"]
    hessian = np.zeros((len(point), len(point)))
    value = likelihood(point)
    for i in range(len(point)):
        a = np.eye(len(point))[i]*epsilon
        hessian[i, i] = (likelihood(point+a)-2*value+likelihood(point-a))/epsilon**2
        for j in range(i):
            b = np.eye(len(point))[j]*epsilon
            hessian[i, j] = hessian[j, i] = (likelihood(point+a+b)-likelihood(point+a-b)-likelihood(point-a+b)+likelihood(point-a-b))/(4*epsilon**2)
    return -hessian


def independent_parameter_fields(state):
    """Analytical Cholesky/variance coordinate Jacobian, independent of Torch."""
    theta = np.asarray(state["theta"])
    system = physical_system(theta,state)
    p, r, anchors = len(state["y"][0]),state["factors"],state["anchors"]
    free = [i for i in range(p) if i not in anchors]
    cells = [(i,j) for i in range(r) for j in range(i+1)]
    offset = p+len(free)*r+r*r
    C = np.zeros((r,r))
    for step,(i,j) in enumerate(cells):
        C[i,j] = np.exp(theta[offset+step]) if i == j else theta[offset+step]
    J = np.eye(len(theta))
    J[offset:offset+len(cells),offset:offset+len(cells)] = 0
    for row,(i,j) in enumerate(cells):
        for column,(u,v) in enumerate(cells):
            derivative = (float(i == u)*C[j,v]+float(j == u)*C[i,v])
            if u == v:
                derivative *= C[u,u]
            J[offset+row,offset+column] = derivative
    for j in range(p):
        J[-p+j,-p+j] = system["H"][j,j]
    parameters = np.concatenate((system["d"],system["Z"][free].flatten(),system["T"].flatten(),
                                 np.asarray([system["Q"][i,j] for i,j in cells]),system["H"].diagonal()))
    information = np.asarray(state["inference"]["information"])
    covariance = np.linalg.inv(information)
    return dict(parameters=parameters,jacobian=J,chart_covariance=covariance,covariance=J @ covariance @ J.T)


R_BRIDGE = r'''
args <- commandArgs(TRUE)
.libPaths(c(args[3], args[4], .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(KFAS))
input <- fromJSON(args[1], simplifyVector=FALSE)
mat <- function(x) do.call(rbind, lapply(x, unlist))
Y <- mat(lapply(input$y, function(row) lapply(row, function(v) if(is.null(v)) NA_real_ else v)))
s <- input$system
Z <- mat(s$Z); T <- mat(s$T); Q <- mat(s$Q); H <- mat(s$H); P0 <- mat(s$P0)
a0 <- unlist(s$a0); d <- unlist(s$d)
n <- nrow(Y); p <- ncol(Y); r <- nrow(T)
centered <- sweep(Y,2,d,"-")
model <- SSModel(centered ~ -1+SSMcustom(Z=Z,T=T,R=diag(r),Q=Q,a1=a0,P1=P0,P1inf=matrix(0,r,r),index=seq_len(p)),H=H)
stopifnot(identical(dim(model$Z)[1:2],c(p,r)),
          isTRUE(all.equal(as.numeric(model$Z[,,1]),as.numeric(Z),check.attributes=FALSE)),
          isTRUE(all.equal(as.numeric(model$T[,,1]),as.numeric(T),check.attributes=FALSE)),
          isTRUE(all.equal(as.numeric(model$Q[,,1]),as.numeric(Q),check.attributes=FALSE)),
          isTRUE(all.equal(as.numeric(model$H[,,1]),as.numeric(H),check.attributes=FALSE)),
          isTRUE(all.equal(drop(model$a1),a0,check.attributes=FALSE)),
          isTRUE(all.equal(model$P1,P0,check.attributes=FALSE)))
out <- KFS(model,filtering="state",smoothing="state")
receipt <- list(R=as.character(getRversion()),KFAS=as.character(packageVersion("KFAS")),
     license_KFAS=packageDescription("KFAS")$License,source_identity_checked=TRUE,
     kfas=list(log_likelihood=as.numeric(logLik(model)),filtered=unname(out$att),
       filtered_covariance=aperm(out$Ptt,c(3,1,2)),prior_mean=unname(out$a[1:n,,drop=FALSE]),
       prior_covariance=aperm(out$P[,,1:n,drop=FALSE],c(3,1,2)),
       smoothed=unname(out$alphahat),smoothed_covariance=aperm(out$V,c(3,1,2)),
       next_mean=drop(out$a[n+1,,drop=FALSE]),next_covariance=out$P[,,n+1]))
if(nzchar(args[4])) {
 suppressPackageStartupMessages(library(dfms))
 # dfms predicts its F_0/P_0 once BEFORE the first row. Convert explicitly,
 # verify SPD, and fail when this exact convention cannot represent P0.
 initial <- try(solve(T),silent=TRUE)
 if(inherits(initial,"try-error")) stop("dfms initial timing cannot represent this singular AR system")
 F0 <- drop(initial %*% a0)
 prior0 <- initial %*% (P0-Q) %*% t(initial)
 if(min(eigen(prior0,symmetric=TRUE,only.values=TRUE)$values) <= 0) stop("dfms initial timing requires an inadmissible pre-sample covariance")
 stopifnot(max(abs(T%*%F0-a0)) < 1e-12,max(abs(T%*%prior0%*%t(T)+Q-P0)) < 1e-12)
 package_out <- SKFS(centered,T,Z,Q,H,F0,prior0,TRUE)
 counts <- rowSums(!is.na(centered))
 correction <- sum((p-counts)[counts>0])*log(2*pi)/2
 receipt$dfms <- list(version=as.character(packageVersion("dfms")),license=packageDescription("dfms")$License,
    convention="pre-transition initial prior converted exactly; package likelihood subtracts p*log(2*pi) on partial dates",
    partial_normalization_correction=correction,raw_log_likelihood=package_out$loglik,
    log_likelihood=package_out$loglik+correction,filtered=unname(package_out$F),
    filtered_covariance=aperm(package_out$P,c(3,1,2)),prior_mean=unname(package_out$F_pred),
    prior_covariance=aperm(package_out$P_pred,c(3,1,2)),smoothed=unname(package_out$F_smooth),
    smoothed_covariance=aperm(package_out$P_smooth,c(3,1,2)),
    lag_one_covariance=aperm(package_out$PPm_smooth[,,-1,drop=FALSE],c(3,1,2)))
}
write_json(receipt,args[2],auto_unbox=TRUE,digits=NA,na="null",pretty=TRUE)
'''


def compare(cached, oracle, *, tolerance=2e-10):
    errors = {}
    for key, expected in oracle.items():
        if key not in cached:
            raise ValueError(f"Cached native result is missing {key}")
        if key == "observed_mask" and np.asarray(cached[key]).dtype.kind != "b":
            raise ValueError("Cached observed-mask scalars must be actual booleans")
        if key == "observed_count" and np.asarray(cached[key]).dtype.kind not in "iu":
            raise ValueError("Cached observed counts must be actual integers")
        actual = np.asarray(cached[key], dtype=float)
        expected = np.asarray(expected, dtype=float).reshape(actual.shape)
        if not np.array_equal(np.isnan(actual), np.isnan(expected)):
            raise ValueError(f"{key} missing-value positions differ")
        finite = np.isfinite(expected)
        errors[key] = float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.
        if not np.allclose(actual, expected, atol=tolerance, rtol=tolerance, equal_nan=True):
            raise ValueError(f"Cached {key} differs from independent reference; max error={errors[key]}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--r-library", required=True)
    parser.add_argument("--dfms-library", default="")
    parser.add_argument("--dfms-source", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.state.stat().st_size > 64*1024**2:
        raise ValueError("Cached native file exceeds 64 MiB")
    raw_bytes = args.state.read_bytes()
    raw = _load_record(raw_bytes.decode())
    if (raw["schema"] != RESULT_SCHEMA or raw["sha256"] != _digest({k: v for k, v in raw.items() if k != "sha256"})):
        raise ValueError("Invalid cached public result checksum")
    state = raw["state"]
    if state["schema"] != SCHEMA or state["sha256"] != _digest({k: v for k, v in state.items() if k != "sha256"}):
        raise ValueError("Invalid cached native numerical-state checksum")
    n, p, r = len(state["y"]), len(state["y"][0]), state["factors"]
    settings = state["settings"]
    _geometry(n,p,r,em_iterations=settings["max_em_iterations"],ml_iterations=settings["max_ml_iterations"],
              restarts=settings["restarts"],max_work=settings["max_work"])
    if not _raw_array(state["y"], (n,p), missing=True):
        raise ValueError("Invalid cached original observation geometry")
    y = np.asarray(state["y"], dtype=float)
    system = {k: np.asarray(v) for k, v in state["system"].items()}
    parameter_system_errors = compare(state["system"],physical_system(np.asarray(state["theta"]),state))
    independent_filter = numpy_filter(y,system)
    independent_joint = primitive_conditioning(y,system)
    if set(state["output"]) != set(independent_filter)|set(independent_joint):
        raise ValueError("Cached numerical fields differ from the complete independently checked schema")
    numpy_errors = compare(state["output"], independent_filter)
    gls_errors = compare(state["output"], independent_joint)
    information = finite_information(y,state,1e-4)
    information_half = finite_information(y,state,5e-5)
    expected = np.asarray(state["inference"]["information"])
    if not np.allclose(information,expected,rtol=3e-5,atol=5e-5):
        raise ValueError("Actual cached native OIM differs from independent NumPy likelihood finite differences")
    if not np.allclose(information_half,expected,rtol=1e-4,atol=2e-4):
        raise ValueError("OIM finite differences fail the second step-size check")
    parameter_errors = compare(state["inference"],independent_parameter_fields(state))
    if (state["inference"]["reference"] != "normal" or state["inference"]["df"] is not None
            or state["inference"]["covariance_kind"] != "nonrobust-observed-information"):
        raise ValueError("Cached inference distribution/covariance metadata differ from this model")
    point = np.asarray(state["theta"])
    score = np.zeros(len(point))
    for i in range(len(point)):
        step = np.eye(len(point))[i]*1e-5
        score[i] = (numpy_filter(y,physical_system(point+step,state))["log_likelihood"]
                    -numpy_filter(y,physical_system(point-step,state))["log_likelihood"])/(2e-5)
    if not np.allclose(score,state["inference"]["score"],atol=5e-7,rtol=1e-5):
        raise ValueError("Cached score differs from independent NumPy likelihood finite differences")
    with tempfile.TemporaryDirectory(prefix="openecon-dfm-oracle-") as temp:
        folder=Path(temp)
        (folder/"input.json").write_text(json.dumps(dict(y=state["y"],system=state["system"]),allow_nan=False))
        (folder/"oracle.R").write_text(R_BRIDGE)
        process = subprocess.run([args.rscript,str(folder/"oracle.R"),str(folder/"input.json"),str(folder/"output.json"),
                                 args.r_library,args.dfms_library],timeout=60,capture_output=True,text=True)
        if process.returncode:
            raise RuntimeError("Native R reference refused its exact saved inputs: "+process.stderr)
        native=json.loads((folder/"output.json").read_text())
    kfas_errors = compare(state["output"],native["kfas"])
    dfms_errors = None
    if "dfms" in native:
        data={k:v for k,v in native["dfms"].items() if k in state["output"]}
        dfms_errors=compare(state["output"],data)
    source={}
    if args.dfms_source:
        for file in ("src/KalmanFiltering.cpp","LICENSE","DESCRIPTION","data/BM14_M.rda","data/BM14_Q.rda"):
            source[file]=hashlib.sha256((args.dfms_source/file).read_bytes()).hexdigest()
        source["git_commit"]=subprocess.run(["git","rev-parse","HEAD"],cwd=args.dfms_source,check=True,capture_output=True,text=True).stdout.strip()
    receipt=dict(passed=True,actual_cached_path=str(args.state.resolve()),actual_cached_file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        actual_cached_result_sha256=raw["sha256"],actual_cached_state_sha256=state["sha256"],n=n,p=p,r=r,
        likelihood_target=state["prior"]["target"],estimator_called=False,tolerance=2e-10,
        numpy_filter_max_abs_errors=numpy_errors,primitive_joint_conditioning_max_abs_errors=gls_errors,
        finite_difference_OIM_max_abs_error=float(np.max(abs(information-expected))),
        finite_difference_half_step_max_abs_error=float(np.max(abs(information_half-expected))),
        finite_difference_score_max_abs_error=float(np.max(abs(score-np.asarray(state["inference"]["score"])))),
        independent_parameter_fields_max_abs_errors=parameter_errors,parameter_system_max_abs_errors=parameter_system_errors,
        kfas_max_abs_errors=kfas_errors,dfms_max_abs_errors=dfms_errors,native=native,dfms_source=source,
        reference_scope="Independent KFAS and dfms package-author code plus analytical Gaussian conditioning; not original BM101 replication or vendor/frozen/installed proof")
    args.report.write_text(json.dumps(receipt,indent=2,allow_nan=False))
    print(json.dumps({k:receipt[k] for k in ("passed","actual_cached_file_sha256","estimator_called","finite_difference_OIM_max_abs_error")}))


if __name__ == "__main__":
    main()
