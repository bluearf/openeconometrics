"""Independent NumPy equations and optional native R/lavaan observed-OIM receipt.

Reads a saved SEM state/receipt or example stdout. Never imports OpenEcon or
Torch. R/lavaan is an external verification dependency, never an engine fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def load_state(path):
    text = path.read_text()
    marker = "LATENT_SEM_RECEIPT="
    if marker in text:
        text = text.split(marker)[-1].splitlines()[0]
    value = json.loads(text)
    return value.get("state", value.get("payload", value))


def source_moments(state):
    source = state["source"]
    if source["kind"] == "raw":
        rows = np.array(source["rows"], dtype=float)
        rows = rows[np.isfinite(rows).all(axis=1)]
        mean = rows.mean(axis=0)
        centered = rows - mean
        return centered.T @ centered / len(rows), mean, len(rows)
    n = source["original_n"]
    s = np.array(source["covariance"])
    if source["divisor"] == "n-1":
        s *= (n - 1) / n
    return s, None if source["means"] is None else np.array(source["means"]), n


def moments(values, spec):
    """Independent original-unit structural inverse, with no solver parameterization."""
    m = len(spec["nodes"])
    b, psi, intercept = np.zeros((m, m)), np.zeros((m, m)), np.zeros(m)
    for r in spec["records"]:
        v = r["fixed"] if r["group"] is None else values[r["group"]]
        i, j, kind = r["i"], r["j"], r["kind"]
        if kind in ("loading", "path"):
            b[i, j] = v
        elif kind == "intercept":
            intercept[i] = v
        elif kind == "variance":
            psi[i, i] = v
        else:
            psi[i, j] = psi[j, i] = v
    a = np.linalg.inv(np.eye(m) - b)
    return a @ psi @ a.T, a @ intercept, b, psi, intercept, a


def objective(values, spec, s, mean, n):
    p, m = len(spec["columns"]), len(spec["nodes"])
    omega, mu, _, psi, intercept, a = moments(values, spec)
    if np.linalg.eigvalsh(psi).min() <= 0:
        return 1e100, np.zeros(len(values))
    sigma = omega[:p, :p]
    inv = np.linalg.inv(sigma)
    residual = np.zeros(p) if not spec["meanstructure"] else mean - mu[:p]
    weight = inv - inv @ (s + np.outer(residual, residual)) @ inv
    gradient = []
    for group in range(len(values)):
        db, dp, dc = np.zeros((m, m)), np.zeros((m, m)), np.zeros(m)
        for r in spec["records"]:
            if r["group"] != group:
                continue
            i, j, kind = r["i"], r["j"], r["kind"]
            if kind in ("loading", "path"):
                db[i, j] += 1
            elif kind == "intercept":
                dc[i] += 1
            elif kind == "variance":
                dp[i, i] += 1
            else:
                dp[i, j] += 1
                dp[j, i] += 1
        da = a @ db @ a
        ds = (da @ psi @ a.T + a @ dp @ a.T + a @ psi @ da.T)[:p, :p]
        dm = (da @ intercept + a @ dc)[:p] if spec["meanstructure"] else np.zeros(p)
        gradient.append(n / 2 * np.sum(weight * ds) - n * dm @ inv @ residual)
    loss = (
        n
        / 2
        * (
            p * math.log(2 * math.pi)
            + np.linalg.slogdet(sigma)[1]
            + np.trace(inv @ s)
            + residual @ inv @ residual
        )
    )
    return loss, np.array(gradient)


def hessian(values, spec, s, mean, n):
    result = np.zeros((len(values), len(values)))
    for j, v in enumerate(values):
        delta = np.zeros(len(values))
        delta[j] = 1e-5 * max(abs(v), 1.0)
        result[:, j] = (
            objective(values + delta, spec, s, mean, n)[1]
            - objective(values - delta, spec, s, mean, n)[1]
        ) / (2 * delta[j])
    return (result + result.T) / 2


def r_reference(state, s, mean, n, library):
    spec = state["spec"]
    # The external reference has its own model parser and parameter ordering.
    # Native receipt names use ordinary identifiers; complex names remain
    # independently covered by source input/refusal tests.
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", node) for node in spec["nodes"]):
        raise ValueError("Native lavaan receipt currently requires ordinary identifier node names.")
    syntax = []
    for r in spec["records"]:
        left = spec["nodes"][r["i"]]
        right = "1" if r["j"] is None else spec["nodes"][r["j"]]
        coefficient = repr(float(r["fixed"])) if r["group"] is None else f"oe_g{r['group']}"
        if r["kind"] == "loading":
            left, right, op = right, left, "=~"
        elif r["kind"] == "path":
            op = "~"
        elif r["kind"] == "intercept":
            op = "~"
        else:
            op = "~~"
            if r["kind"] == "variance":
                right = left
        syntax.append(f"{left} {op} {coefficient}*{right}")
    with tempfile.TemporaryDirectory(prefix="openecon-sem-lavaan-") as directory:
        root = Path(directory)

        def r_path(name):
            return json.dumps(str(root / name))

        matrix = (
            "matrix(c(" + ",".join(repr(float(v)) for v in s.flat) + f"),nrow={len(s)},byrow=TRUE)"
        )
        names = "c(" + ",".join(json.dumps(v) for v in spec["columns"]) + ")"
        mean_expression = (
            "NULL" if mean is None else "c(" + ",".join(repr(float(v)) for v in mean) + ")"
        )
        script = f""".libPaths(c({json.dumps(str(library))},.libPaths()))
suppressPackageStartupMessages(library(lavaan))
options(digits=17)
sample <- {matrix}
dimnames(sample) <- list({names},{names})
observed_mean <- {mean_expression}
if(!is.null(observed_mean)) names(observed_mean) <- {names}
model <- {json.dumps(chr(10).join(syntax))}
fit <- lavaan(model,sample.cov=sample,sample.mean=observed_mean,sample.nobs={n},
              sample.cov.rescale=FALSE,estimator="ML",likelihood="normal",information="observed",
              fixed.x=FALSE,meanstructure={str(spec["meanstructure"]).upper()},
              auto.fix.first=FALSE,auto.var=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE,
              int.ov.free=FALSE,int.lv.free=FALSE,control=list(iter.max=3000))
stopifnot(lavInspect(fit,"converged"))
pt <- parTable(fit)
labels <- paste0("oe_g",0:{spec["parameter_count"] - 1})
selected <- vapply(labels,function(label) which(pt$label==label)[1],integer(1))
free <- pt$free[selected]
write.table(pt$est[selected],file={r_path("parameters.csv")},row.names=FALSE,col.names=FALSE,sep=",")
write.table(vcov(fit)[free,free,drop=FALSE],file={r_path("covariance.csv")},row.names=FALSE,col.names=FALSE,sep=",")
measures <- fitMeasures(fit,c("logl","chisq","df","baseline.chisq","baseline.df","cfi","tli","rmsea","srmr"))
write.table(data.frame(name=names(measures),value=as.numeric(measures)),file={r_path("fit.csv")},row.names=FALSE,sep=",")
writeLines(as.character(packageVersion("lavaan")),{r_path("version.txt")})
"""
        rfile = root / "reference.R"
        rfile.write_text(script)
        process = subprocess.run(
            ["Rscript", "--vanilla", str(rfile)], capture_output=True, text=True, timeout=90
        )
        if process.returncode:
            raise RuntimeError(process.stderr or process.stdout)
        values = np.loadtxt(root / "parameters.csv", delimiter=",")
        covariance = np.loadtxt(root / "covariance.csv", delimiter=",")
        import csv

        with (root / "fit.csv").open() as handle:
            measures = {
                row["name"]: None if row["value"] in {"NA", "NaN"} else float(row["value"])
                for row in csv.DictReader(handle)
            }
        return (
            values,
            covariance,
            measures,
            (root / "version.txt").read_text().strip(),
            process.stderr,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lavaan-library", type=Path)
    args = parser.parse_args()
    state = load_state(args.input)
    spec, actual = state["spec"], state["results"]
    s, mean, n = source_moments(state)
    parameters = np.array(actual["parameters"])
    # Natural-unit starting perturbation forces a distinct optimizer run;
    # numerical receipts use ordinary units, while scaling adversaries are
    # independently tested directly in the source suite.
    start = parameters * (1 + np.linspace(-0.015, 0.02, len(parameters)))
    fit = minimize(
        lambda v: objective(v, spec, s, mean, n),
        start,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-7, "maxiter": 1500},
    )
    score = float(np.max(np.abs(objective(fit.x, spec, s, mean, n)[1])))
    if score > 3e-5:
        raise AssertionError(f"Independent natural-parameter score is {score}.")
    independent_covariance = np.linalg.inv(hessian(fit.x, spec, s, mean, n))
    np.testing.assert_allclose(parameters, fit.x, rtol=4e-6, atol=5e-7)
    np.testing.assert_allclose(actual["covariance"], independent_covariance, rtol=6e-5, atol=3e-8)
    if abs(actual["log_likelihood"] + fit.fun) > 1e-7:
        raise AssertionError("Independent likelihood differs.")
    receipt = {
        "schema": "openecon.latent_sem.independent.v1",
        "engine": "NumPy/SciPy natural-unit score + finite-difference OIM",
        "input_digest": state["digest"],
        "parameter_count": len(parameters),
        "n": n,
        "natural_score_max": score,
        "independent_iterations": int(fit.nit),
        "max_parameter_error": float(np.max(np.abs(parameters - fit.x))),
        "max_joint_covariance_error": float(
            np.max(np.abs(np.array(actual["covariance"]) - independent_covariance))
        ),
        "log_likelihood_error": float(actual["log_likelihood"] + fit.fun),
    }
    if args.lavaan_library:
        values, covariance, measures, version, warnings = r_reference(
            state, s, mean, n, args.lavaan_library
        )
        np.testing.assert_allclose(parameters, values, rtol=4e-6, atol=5e-7)
        np.testing.assert_allclose(actual["covariance"], covariance, rtol=6e-5, atol=3e-8)
        if abs(actual["log_likelihood"] - measures["logl"]) > 1e-7:
            raise AssertionError("Native lavaan likelihood differs.")
        diagnostics = actual["diagnostics"]
        for ours, theirs in (
            ("statistic", "chisq"),
            ("df", "df"),
            ("baseline_statistic", "baseline.chisq"),
            ("baseline_df", "baseline.df"),
            ("CFI", "cfi"),
            ("TLI", "tli"),
            ("RMSEA", "rmsea"),
            ("SRMR", "srmr"),
        ):
            if (diagnostics[ours] is None) != (measures[theirs] is None):
                raise AssertionError(f"Native lavaan {ours} availability differs.")
            if diagnostics[ours] is not None and abs(diagnostics[ours] - measures[theirs]) > 3e-6:
                raise AssertionError(
                    f"Native lavaan {ours} differs: {diagnostics[ours]} vs {measures[theirs]}."
                )
        receipt["lavaan"] = {
            "version": version,
            "information": "observed",
            "likelihood": "normal",
            "max_parameter_error": float(np.max(np.abs(parameters - values))),
            "max_joint_covariance_error": float(
                np.max(np.abs(np.array(actual["covariance"]) - covariance))
            ),
            "measures": measures,
            "warnings": warnings.strip(),
        }
    text = json.dumps(receipt, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
