"""Compare ACTUAL saved native fields; never import/refit the native estimator.

External original JSS mixtools0.4.3 GPL>=2 code stays outside the repository.
Its reported last-E posterior and recomputed terminal posterior are distinct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile
import tempfile

import numpy as np
from scipy.special import logsumexp

ARCHIVE_SHA = "f360e316100ee7e87e892e692f85a85771b38b332db0f7ede271cf4ecf0c22ec"
CO2_SHA = "6676876fff5b5bd0072c7b4c75aa9fbb389c44127d017d2284b6a66393f14a7e"
AUTHOR_FILES = {
    "mixtools/R/regmixEM.R": "a6f46453615cec02fe03d926c12cde123cfc4aa691740be197902bd48e742edf",
    "mixtools/R/regmixinit.R": "a182d0c273e7b8f3d0f4e55149e8688388efc6bac256ae690d3687b550294c93",
    "mixtools/data/CO2data.RData": CO2_SHA,
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load(path):
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Actual cached file exceeds64MiB")
    value = json.loads(path.read_text())
    for record in (value, value.get("state", {}), value.get("source", {})):
        if type(record) is not dict or record.get("digest") != digest(
            {k: v for k, v in record.items() if k != "digest"}
        ):
            raise ValueError("Actual cached result/core/source digest is invalid")
    return value


def primitive(theta, x, y, k):
    n, d = x.shape
    q = len(theta)
    beta = theta[: k * d].reshape(k, d)
    sigma = np.exp(theta[k * d : k * d + k])
    logit = np.r_[theta[k * d + k :], 0.0]
    log_pi = logit - logsumexp(logit)
    pi = np.exp(log_pi)
    mu = x @ beta.T
    residual = y[:, None] - mu
    log_component = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * (residual / sigma) ** 2
    joint = log_component + log_pi
    row = logsumexp(joint, axis=1)
    z = np.exp(joint - row[:, None])
    mean = mu @ pi
    output = {
        "log_likelihood": row.sum(),
        "log_density": row,
        "density": np.exp(row),
        "responsibility": z,
        "component_mean": mu,
        "component_log_density": log_component,
        "prior": np.tile(pi, (n, 1)),
        "mean": mean,
        "variance": (sigma**2 + (mu - mean[:, None]) ** 2) @ pi,
    }
    score = np.zeros(q)
    hessian = np.zeros((q, q))
    for i in range(n):
        complete = np.zeros((k, q))
        for j in range(k):
            sl = slice(j * d, (j + 1) * d)
            si = k * d + j
            complete[j, sl] = x[i] * residual[i, j] / sigma[j] ** 2
            complete[j, si] = residual[i, j] ** 2 / sigma[j] ** 2 - 1
            if k > 1:
                complete[j, -k + 1 :] = -pi[:-1]
                if j < k - 1:
                    complete[j, -k + 1 + j] += 1
            hessian[sl, sl] -= z[i, j] * np.outer(x[i], x[i]) / sigma[j] ** 2
            cross = -2 * z[i, j] * x[i] * residual[i, j] / sigma[j] ** 2
            hessian[sl, si] += cross
            hessian[si, sl] += cross
            hessian[si, si] -= 2 * z[i, j] * residual[i, j] ** 2 / sigma[j] ** 2
        if k > 1:
            hessian[-k + 1 :, -k + 1 :] -= (np.diag(pi) - np.outer(pi, pi))[:-1, :-1]
        observed = z[i] @ complete
        score += observed
        hessian += complete.T @ (complete * z[i, :, None]) - np.outer(observed, observed)
    information = -(hessian + hessian.T) / 2
    jac = np.zeros((q + 1, q))
    jac[: k * d, : k * d] = np.eye(k * d)
    jac[k * d : k * d + k, k * d : k * d + k] = np.diag(sigma)
    if k > 1:
        jac[-k:, -k + 1 :] = (np.diag(pi) - np.outer(pi, pi))[:, :-1]
    physical = np.r_[beta.ravel(), sigma, pi]
    return output, score, information, jac, physical


def derivatives(theta, x, y, k, step):
    basis = np.eye(len(theta)) * step

    def ll(t):
        d = x.shape[1]
        beta = t[: k * d].reshape(k, d)
        sigma = np.exp(t[k * d : k * d + k])
        logits = np.r_[t[k * d + k :], 0.0]
        log_joint = (
            logits
            - logsumexp(logits)
            - np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * ((y[:, None] - x @ beta.T) / sigma) ** 2
        )
        return float(logsumexp(log_joint, axis=1).sum())

    score = np.array([(ll(theta + a) - ll(theta - a)) / (2 * step) for a in basis])
    info = np.array(
        [
            [
                -(ll(theta + a + b) - ll(theta + a - b) - ll(theta - a + b) + ll(theta - a - b))
                / (4 * step**2)
                for b in basis
            ]
            for a in basis
        ]
    )
    return score, info


def compare(actual, reference, name, errors, *, atol=2e-9, rtol=2e-10):
    def numeric(value):
        if type(value) is list:
            for child in value:
                numeric(child)
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError(
                f"{name} actual cached numeric types must be finite floats, never bool/int aliases"
            )

    numeric(actual)
    a = np.asarray(actual, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"{name} has invalid shape/nonfinite values")
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=name)
    errors[name] = float(np.max(np.abs(a - b))) if a.size else 0.0


R_SCRIPT = r"""
args<-commandArgs(trailingOnly=TRUE)
library(jsonlite,lib.loc=args[3])
v<-fromJSON(args[1]);source(file.path(args[4],"mixtools/R/regmixinit.R"));source(file.path(args[4],"mixtools/R/regmixEM.R"))
evaluate<-function(theta,x,y,K) {
 d<-ncol(x);beta<-matrix(theta[seq_len(K*d)],K,d,byrow=TRUE);sigma<-exp(theta[K*d+seq_len(K)])
 logits<-c(if(K>1) theta[(K*d+K+1):length(theta)] else numeric(),0)
 logpi<-logits-max(logits)-log(sum(exp(logits-max(logits))));pi<-exp(logpi)
 mu<-x%*%t(beta);logcomp<-sapply(seq_len(K),function(j) dnorm(y,mu[,j],sigma[j],log=TRUE));logcomp<-matrix(logcomp,nrow=length(y),ncol=K)
 joint<-sweep(logcomp,2,logpi,"+"); rowmax<-apply(joint,1,max);ld<-rowmax+log(rowSums(exp(joint-rowmax)))
 posterior<-exp(joint-ld);mean<-as.numeric(mu%*%pi)
 list(log_likelihood=sum(ld),log_density=ld,density=exp(ld),responsibility=posterior,component_mean=mu,
 component_log_density=logcomp,prior=matrix(rep(pi,each=length(y)),ncol=K),mean=mean,
 variance=as.numeric(sweep((mu-mean)^2,2,sigma^2,"+")%*%pi))
}
x<-as.matrix(v$x);y<-as.numeric(v$y);K<-as.integer(v$K);theta<-as.numeric(v$theta)
fixed<-evaluate(theta,x,y,K)
initial<-v$initial;beta<-t(as.matrix(initial$coefficients));scales<-as.numeric(initial$scales);weights<-as.numeric(initial$weights)
if(K>1 && min(round(weights*length(y)))<4) stop("Original author initializer would replace supplied weights; this saved start is outside that reference convention")
author<-if(K>1 && v$author_fit_comparable) regmixEM(y,x,lambda=weights,beta=beta,sigma=scales,addintercept=FALSE,epsilon=1e-12,maxit=1000) else NULL
author_terminal<-NULL
if(!is.null(author)) {
 terminal_theta<-c(as.numeric(author$beta),log(author$sigma),log(author$lambda[-K]/author$lambda[K]))
 # beta columns are each component; as.numeric(column-major) is our component-major chart.
 author_terminal<-evaluate(terminal_theta,x,y,K)
}
load(file.path(args[4],"mixtools/data/CO2data.RData"));stopifnot(nrow(CO2data)==28L)
paper<-regmixEM(CO2data$CO2,CO2data$GNP,lambda=c(1,3)/4,beta=matrix(c(8,-1,1,1),2,2),sigma=c(2,1),epsilon=1e-8,maxit=10000)
paper_theta<-c(as.numeric(paper$beta),log(paper$sigma),log(paper$lambda[1]/paper$lambda[2]))
paper_terminal<-evaluate(paper_theta,paper$x,paper$y,2)
receipt<-list(r_version=R.version.string,fixed=fixed,
 author=if(is.null(author)) NULL else list(lambda=author$lambda,beta=t(author$beta),sigma=author$sigma,log_likelihood=author$loglik,
 posterior_reported=author$posterior,terminal=author_terminal,iterations=length(author$all.loglik)-1L,restarts=author$restarts,
 all_loglik=author$all.loglik),
 original_paper=list(n=28L,beta=t(paper$beta),scales=paper$sigma,weights=paper$lambda,log_likelihood=paper$loglik,
 posterior_reported=paper$posterior,terminal=paper_terminal,x=paper$x,y=paper$y,restarts=paper$restarts,
 stale_last_E_max=max(abs(paper$posterior-paper_terminal$responsibility))))
write_json(receipt,args[2],auto_unbox=TRUE,digits=17,pretty=TRUE)
"""


def verify(args):
    actual = load(args.state)
    state = actual["state"]
    source = actual["source"]
    if (
        actual.get("schema") != "openecon.finite-mixture.result.v1"
        or state.get("schema") != "openecon.finite-mixture.gaussian.v1"
    ):
        raise ValueError("Wrong actual saved model schema")
    k, d = state["components"], state["design_columns"]
    if type(k) is not int or not 1 <= k <= 4 or type(d) is not int or not 1 <= d <= 33:
        raise ValueError("Native reference geometry is invalid")
    n = len(state["y"])
    q = k * d + 2 * k - 1
    if not 3 <= n <= 20000 or q > 64:
        raise ValueError(
            "Independent finite-difference oracle admits at most64 parameters and20k rows"
        )
    reference_work = 64 * n * k * (d + 8) * (4 * q * q + 4 * q) + 128 * n * k * q * q + 32 * q**3
    if reference_work > 300_000_000:
        raise ValueError(
            "Complete independent finite-difference reference work exceeds300million units"
        )
    x = np.array(state["x"], dtype=float)
    y = np.array(state["y"], dtype=float)
    theta = np.array(state["theta"], dtype=float)
    if x.shape != (n, d) or y.shape != (n,) or theta.shape != (q,):
        raise ValueError("Invalid actual source shapes")
    raw = np.array(source["values"], dtype=float)
    projected = np.c_[np.ones(n), raw[:, 1:]] if source["intercept"] else raw[:, 1:]
    np.testing.assert_array_equal(projected, x)
    np.testing.assert_array_equal(raw[:, 0], y)
    if any(type(v) is not bool or not v for v in source["sample"]) or source["positions"] != list(
        range(n)
    ):
        raise ValueError("Complete original sample positions/mask are invalid")
    if len(source["index"].get("labels", source["index"].get("codes", [[]])[0])) != n:
        raise ValueError("Complete original index identities are required")
    primitive_out, score, info, jac, parameters = primitive(theta, x, y, k)
    errors = {}
    if set(state["output"]) != set(primitive_out):
        raise ValueError("Actual cached output fields are incomplete")
    for key, reference in primitive_out.items():
        compare(state["output"][key], reference, "numpy.output." + key, errors)
    for key, reference in (("score", score), ("information", info), ("jacobian", jac)):
        compare(state["inference"][key], reference, "numpy.inference." + key, errors)
    compare(state["parameters"], parameters, "numpy.physical_parameters", errors)
    available = state["inference"]["available"]
    if type(available) is not bool:
        raise ValueError("Invalid inference availability type")
    if available:
        covariance = np.linalg.inv(info)
        compare(
            state["inference"]["chart_covariance"],
            covariance,
            "numpy.full_chart_covariance",
            errors,
        )
        compare(
            state["inference"]["covariance"],
            jac @ covariance @ jac.T,
            "numpy.full_physical_covariance",
            errors,
        )
        compare(
            state["inference"]["scaled_score"],
            score @ covariance @ score,
            "numpy.scaled_score",
            errors,
            atol=2e-15,
            rtol=1e-6,
        )
    elif (
        state["inference"]["chart_covariance"] is not None
        or state["inference"]["covariance"] is not None
    ):
        raise ValueError("Nonregular fit must not manufacture ordinary covariance")
    for s in state["starts"]:
        previous = None
        for point in s["em_path"]:
            t = np.array(point["theta"])
            out, _, _, _, _ = primitive(t, x, y, k)
            compare(
                point["log_likelihood"],
                out["log_likelihood"],
                f"numpy.start{s['start']}.EM_LL",
                errors,
            )
            if previous is not None:
                old = primitive(previous, x, y, k)[0]
                z = old["responsibility"]
                mass = z.sum(0)
                beta = np.stack(
                    [
                        np.linalg.lstsq(
                            x * np.sqrt(z[:, j, None]), y * np.sqrt(z[:, j]), rcond=None
                        )[0]
                        for j in range(k)
                    ]
                )
                sigma = np.maximum(
                    np.sqrt((z * (y[:, None] - x @ beta.T) ** 2).sum(0) / mass),
                    state["settings"]["sigma_min"],
                )
                pi = mass / n
                expected = np.r_[beta.ravel(), np.log(sigma), np.log(pi[:-1] / pi[-1])]
                compare(
                    point["theta"],
                    expected,
                    f"numpy.start{s['start']}.EM_chart",
                    errors,
                    atol=5e-10,
                    rtol=5e-10,
                )
            previous = t
    successful = [s for s in state["starts"] if s["status"] == "converged"]
    winner = max(successful, key=lambda v: (v["log_likelihood"], -v["start"]))
    if winner["start"] != state["chosen_start"]:
        raise ValueError("Highest constrained objective was not retained")
    initial = np.array(winner["initial"])
    _, _, _, _, initial_phys = primitive(initial, x, y, k)
    initial_record = {
        "coefficients": initial_phys[: k * d].reshape(k, d).tolist(),
        "scales": initial_phys[k * d : k * d + k].tolist(),
        "weights": initial_phys[-k:].tolist(),
    }
    if (
        args.author_archive.stat().st_size > 16 * 1024**2
        or hashlib.sha256(args.author_archive.read_bytes()).hexdigest() != ARCHIVE_SHA
    ):
        raise ValueError("Original JSS author archive size/hash mismatch")
    with tempfile.TemporaryDirectory(prefix="finite-mixture-author-") as scratch:
        folder = Path(scratch)
        with tarfile.open(args.author_archive, "r:gz") as archive:
            description = archive.extractfile("mixtools/DESCRIPTION").read().decode()
            if "Version: 0.4.3" not in description or "License: GPL (>= 2)" not in description:
                raise ValueError("Original author version/license mismatch")
            for name, expected_hash in AUTHOR_FILES.items():
                blob = archive.extractfile(name).read()
                if hashlib.sha256(blob).hexdigest() != expected_hash:
                    raise ValueError("Original author file hash mismatch")
                destination = folder / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(blob)
        request = dict(
            x=x.tolist(),
            y=y.tolist(),
            theta=theta.tolist(),
            K=k,
            initial=initial_record,
            author_fit_comparable=available
            and all(
                np.exp(np.array(v["theta"])[k * d : k * d + k]).min()
                > state["settings"]["sigma_min"] * (1 + 128 * np.finfo(float).eps)
                for v in winner["em_path"]
            ),
        )
        input_file = folder / "input.json"
        input_file.write_text(json.dumps(request, allow_nan=False))
        r_file = folder / "reference.R"
        r_file.write_text(R_SCRIPT)
        output_file = folder / "output.json"
        run = subprocess.run(
            [
                str(args.rscript),
                str(r_file),
                str(input_file),
                str(output_file),
                str(args.r_library),
                str(folder),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if run.returncode:
            raise RuntimeError("Original native R reference failed: " + run.stdout + run.stderr)
        r = json.loads(output_file.read_text())
    for key, reference in r["fixed"].items():
        compare(state["output"][key], reference, "native_R.output." + key, errors)
    if r["author"] is not None:
        author = r["author"]
        beta = np.array(author["beta"])
        sigma = np.array(author["sigma"])
        pi = np.array(author["lambda"])
        permutation = sorted(range(k), key=lambda j: (*beta[j], sigma[j], pi[j]))
        author_parameters = np.r_[beta[permutation].ravel(), sigma[permutation], pi[permutation]]
        compare(
            state["parameters"],
            author_parameters,
            "original_author.fitted_parameters",
            errors,
            atol=2e-7,
            rtol=2e-7,
        )
        compare(
            state["output"]["log_likelihood"],
            author["log_likelihood"],
            "original_author.fitted_LL",
            errors,
            atol=1e-9,
            rtol=1e-10,
        )
        compare(
            state["output"]["responsibility"],
            np.array(author["terminal"]["responsibility"])[:, permutation],
            "original_author.recomputed_terminal_posterior",
            errors,
            atol=2e-7,
            rtol=2e-7,
        )
    fd_score, fd_info = derivatives(theta, x, y, k, 1e-4)
    fd_half_score, fd_half_info = derivatives(theta, x, y, k, 5e-5)
    fd_info_extrap = (4 * fd_half_info - fd_info) / 3
    compare(
        state["inference"]["score"],
        (4 * fd_half_score - fd_score) / 3,
        "independent_FD_score",
        errors,
        atol=2e-6,
        rtol=2e-6,
    )
    compare(
        state["inference"]["information"],
        fd_info_extrap,
        "independent_FD_information",
        errors,
        atol=3e-4,
        rtol=3e-6,
    )
    receipt = dict(
        schema="openecon.finite-mixture.actual-cached-oracle.v1",
        actual_state_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
        public_digest=actual["digest"],
        core_digest=state["digest"],
        n=n,
        components=k,
        free_parameters=q,
        no_native_estimator_import_or_call=True,
        independent_reference_work=reference_work,
        cache_fields=list(state["output"]),
        errors=errors,
        native_R=r,
        author_reference=dict(
            paper="https://www.jstatsoft.org/article/view/v032i06",
            package_version="0.4.3",
            license="GPL>=2 external developer oracle",
            archive_sha256=ARCHIVE_SHA,
            files=AUTHOR_FILES,
        ),
        finite_difference=dict(
            full_step=1e-4,
            half_step=5e-5,
            information_full_error=float(np.max(abs(fd_info - info))),
            information_half_error=float(np.max(abs(fd_half_info - info))),
            richardson_error=float(np.max(abs(fd_info_extrap - info))),
        ),
        author_stale_E_convention="Reported last-E posterior retained separately; recomputed terminal posterior uses actual reported author parameters; no widened stale-array tolerance",
        boundary="Source/cached numerical and original-author native R evidence; frozen/installed/vendor gates are separate",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--author-archive",
        type=Path,
        default=Path("/tmp/openecon-mixture-author-preparation/mixtools-original.tar.gz"),
    )
    parser.add_argument("--rscript", type=Path, default=Path("/opt/homebrew/bin/Rscript"))
    parser.add_argument("--r-library", type=Path, default=Path("/tmp/openecon-kfas-native-library"))
    parser.add_argument("--report", type=Path, required=True)
    receipt = verify(parser.parse_args())
    print(
        json.dumps(
            {
                "passed": True,
                "actual_state_sha256": receipt["actual_state_sha256"],
                "max_native_R_error": max(
                    v for k, v in receipt["errors"].items() if k.startswith("native_R.")
                ),
                "original_paper_stale_E_max": receipt["native_R"]["original_paper"][
                    "stale_last_E_max"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
