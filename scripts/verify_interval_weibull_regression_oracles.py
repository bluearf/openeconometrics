"""Standalone original-R and independently differentiated Weibull ML oracle.

Imports neither Torch nor OpenEcon. SciPy is exclusively a development oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tempfile

import numpy as np
from scipy.optimize import minimize


def geometry(raw, design, lower, upper):
    """Independent original-time AFT score/Hessian, including scale cross terms."""
    raw = np.asarray(raw)
    design, lower = np.asarray(design), np.asarray(lower)
    p = np.exp(-raw[-1])
    total, score, hessian = 0.0, np.zeros(len(raw)), np.zeros((len(raw), len(raw)))

    def point(t, x):
        z = p * (np.log(t) - x @ raw[:-1])
        dz = np.r_[-p * x, -z]
        hz = np.zeros((len(raw), len(raw)))
        hz[:-1, -1] = hz[-1, :-1] = p * x
        hz[-1, -1] = z
        h = np.exp(z)
        s = -h * dz
        hs = -h * (np.outer(dz, dz) + hz)
        return z, h, dz, hz, s, hs

    for x, lo, hi in zip(design, lower, upper):
        if hi is None:
            z, h, dz, hz, s, hs = point(lo, x)
            value, g, hh = -h, s, hs
        elif lo == hi:
            z, h, dz, hz, _, _ = point(lo, x)
            value = -raw[-1] - np.log(lo) + z - h
            g = (1 - h) * dz
            g[-1] -= 1
            hh = -h * np.outer(dz, dz) + (1 - h) * hz
        elif lo == 0:
            z, h, dz, hz, _, _ = point(hi, x)
            value = np.log(-np.expm1(-h))
            a = h / np.expm1(h)
            aa = a * (1 - h - a)
            g = a * dz
            hh = aa * np.outer(dz, dz) + a * hz
        else:
            _, hl, _, _, sl, hsl = point(lo, x)
            _, hu, _, _, su, hsu = point(hi, x)
            delta = hu - hl
            r = np.exp(-delta)
            den = -np.expm1(-delta)
            value = -hl + np.log(den)
            a, b = 1 / den, -r / den
            g = a * sl + b * su
            hh = a * (hsl + np.outer(sl, sl)) + b * (hsu + np.outer(su, su)) - np.outer(g, g)
        total += value
        score += g
        hessian += hh
    return total, score, -(hessian + hessian.T) / 2


def ph(raw, covariance):
    p = np.exp(-raw[-1])
    jac = np.zeros((len(raw), len(raw)))
    jac[:-1, :-1] = -p * np.eye(len(raw) - 1)
    jac[:-1, -1] = p * raw[:-1]
    jac[-1, -1] = -1
    return np.r_[-p * raw[:-1], -raw[-1]], jac @ covariance @ jac.T, jac


def independent_fit(design, lower, upper):
    """SciPy optimization in an independently standardized nonsingular chart."""
    x = design[:, 1:]
    location, scale = x.mean(0), x.std(0)
    xx = np.column_stack((np.ones(len(design)), (x - location) / scale))
    representative = np.array([lo if lo else hi / 2 for lo, hi in zip(lower, upper)])
    time_center = np.log(representative).mean()
    unit = np.exp(time_center)
    ll = np.array(lower) / unit
    uu = [None if v is None else v / unit for v in upper]
    start = np.r_[np.linalg.lstsq(xx, np.log(representative / unit), rcond=None)[0], 0.0]
    fitted = minimize(
        lambda v: -geometry(v, xx, ll, uu)[0],
        start,
        jac=lambda v: -geometry(v, xx, ll, uu)[1],
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 2000},
    )
    raw = fitted.x.copy()
    raw[1:-1] /= scale
    raw[0] += time_center - location @ raw[1:-1]
    exact = sum(lo == hi for lo, hi in zip(lower, upper))
    return raw, float(-fitted.fun - exact * time_center)


def prediction(raw, covariance, x, times):
    design = np.column_stack((np.ones(len(x)), x))
    p = np.exp(-raw[-1])
    s, h, sj, hj, lj = [], [], [], [], []
    for row in design:
        for time in times:
            if time == 0:
                s.append(1.0)
                h.append(0.0)
                sj.append(np.zeros(len(raw)))
                hj.append(np.zeros(len(raw)))
                lj.append(np.zeros(len(raw)))
            else:
                z = p * (np.log(time) - row @ raw[:-1])
                hz = np.exp(z)
                sv = np.exp(-hz)
                dz = np.r_[-p * row, -z]
                s.append(sv)
                h.append(hz)
                sj.append(-sv * hz * dz)
                hj.append(hz * dz)
                lj.append(dz)
    jac = np.array(sj + hj)
    return np.array(s + h), jac, jac @ covariance @ jac.T, covariance @ jac.T, np.array(lj)


def original_r(model, rscript=None):
    rscript = rscript or shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Actual Rscript reference is unavailable; no live-author gate can pass.")
    source, spec = model["source"], model["spec"]
    rows = source["values"]
    nvars = len(spec["x"])
    with tempfile.TemporaryDirectory(prefix="weibull-r-oracle-") as directory:
        base = Path(directory)
        path = base / "data.csv"
        with path.open("w") as handle:
            handle.write(",".join(["lo", "hi", *[f"x{i + 1}" for i in range(nvars)]]) + "\n")
            for row in rows:
                handle.write(",".join("" if v is None else repr(v) for v in row) + "\n")
        formula = "Surv(lo,hi,type='interval2')~" + (
            "+".join(f"x{i + 1}" for i in range(nvars)) or "1"
        )
        text = f"""library(survival)
d<-read.csv('{path.as_posix()}')
f<-survreg({formula}, data=d,dist='weibull',robust=FALSE,
 control=survreg.control(maxiter=1000,rel.tolerance=1e-12))
write.table(c(f$coefficients,log(f$scale)),file='{(base / "theta.txt").as_posix()}',row.names=FALSE,col.names=FALSE)
write.table(f$var,file='{(base / "cov.txt").as_posix()}',row.names=FALSE,col.names=FALSE)
write.table(f$loglik[2],file='{(base / "ll.txt").as_posix()}',row.names=FALSE,col.names=FALSE)
writeLines(paste(as.character(packageVersion('survival')),R.version.string,sep=' | '),'{(base / "version.txt").as_posix()}')
"""
        script = base / "run.R"
        script.write_text(text)
        result = subprocess.run([rscript, str(script)], capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError("Actual original survival::survreg failed: " + result.stderr)
        return dict(
            theta=np.loadtxt(base / "theta.txt"),
            covariance=np.loadtxt(base / "cov.txt"),
            log_likelihood=float(np.loadtxt(base / "ll.txt")),
            version=(base / "version.txt").read_text().strip(),
            stderr=result.stderr,
            invocation_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )


def normalized_error(actual, expected, covariance=None):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if covariance is None:
        return float(np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))))
    se = np.sqrt(np.diag(covariance))
    scale = se[:, None] * se[None, :]
    return float(np.max(np.abs(actual - expected) / scale))


def audit(document, *, run_r=True):
    checks, references = [], []

    def check(case, field, error, tolerance):
        checks.append(
            dict(
                case=case,
                field=field,
                error=float(error),
                tolerance=tolerance,
                passed=bool(np.isfinite(error) and error <= tolerance),
            )
        )

    for i, model in enumerate(document["models"]):
        raw = np.array(model["result"]["aft_parameters"])
        covariance = np.array(model["result"]["aft_covariance"])
        values = model["source"]["values"]
        design = np.column_stack((np.ones(len(values)), np.array([row[2:] for row in values])))
        lower, upper = [row[0] for row in values], [row[1] for row in values]
        ll, score, info = geometry(raw, design, lower, upper)
        oracle_cov = np.linalg.inv(info)
        check(
            i,
            "independent_original_likelihood",
            abs(ll - model["endpoint"]["log_likelihood"]),
            1e-8,
        )
        check(
            i,
            "independent_full_OIM_covariance",
            normalized_error(covariance, oracle_cov, oracle_cov),
            1e-8,
        )
        info_units = np.sqrt(np.diag(info))
        check(
            i,
            "independent_original_full_information",
            float(
                np.max(
                    np.abs(np.array(model["result"]["aft_information"]) - info)
                    / (info_units[:, None] * info_units[None, :])
                )
            ),
            1e-8,
        )
        check(i, "independent_original_score_quadratic", float(score @ oracle_cov @ score), 1e-10)
        pc, cc, jj = ph(raw, oracle_cov)
        check(
            i,
            "independent_full_PH_transform",
            normalized_error(model["result"]["ph_covariance"], cc, cc),
            1e-8,
        )
        check(i, "PH_parameters", normalized_error(model["result"]["ph_parameters"], pc), 1e-10)
        check(i, "PH_Jacobian", normalized_error(model["result"]["ph_jacobian"], jj), 1e-10)
        fitted, fitted_ll = independent_fit(design, lower, upper)
        check(i, "independent_optimizer_likelihood", abs(fitted_ll - ll), 1e-7)
        check(
            i,
            "independent_optimizer_endpoint",
            float(np.linalg.norm((fitted - raw) / np.sqrt(np.diag(covariance)))),
            1e-5,
        )
        if run_r:
            r = original_r(model)
            check(
                i,
                "actual_survreg_endpoint",
                float(np.linalg.norm((r["theta"] - raw) / np.sqrt(np.diag(covariance)))),
                1e-5,
            )
            check(i, "actual_survreg_likelihood", abs(r["log_likelihood"] - ll), 1e-7)
            check(
                i,
                "actual_survreg_full_covariance",
                normalized_error(r["covariance"], covariance, covariance),
                1e-6,
            )
            references.append(
                {
                    key: val.tolist() if isinstance(val, np.ndarray) else val
                    for key, val in r.items()
                }
            )
    for i, query in enumerate(document.get("predictions", [])):
        model = query["target"]
        raw = np.array(model["result"]["aft_parameters"])
        covariance = np.array(model["result"]["aft_covariance"])
        x = np.array(query["source"]["values"])
        estimates, jac, full, cross, logj = prediction(raw, covariance, x, query["times"])
        check(i, "query_values", normalized_error(query["result"]["estimates"], estimates), 1e-10)
        check(
            i,
            "query_full_cross_covariance",
            normalized_error(query["result"]["covariance"], full),
            1e-9,
        )
        check(
            i,
            "parameter_query_cross_covariance",
            normalized_error(query["result"]["parameter_query_covariance"], cross),
            1e-9,
        )
        check(i, "query_full_Jacobian", normalized_error(query["result"]["jacobian"], jac), 1e-9)
        check(
            i,
            "log_hazard_Jacobian",
            normalized_error(query["result"]["log_hazard_jacobian"], logj),
            1e-9,
        )
    return dict(
        oracle="independent NumPy analytic derivatives + SciPy optimizer"
        + (" + actual survival::survreg" if run_r else ""),
        actual_R_executed=run_r,
        production_imported=False,
        torch_imported=False,
        checks=checks,
        references=references,
        passed=all(v["passed"] for v in checks),
        check_count=len(checks),
    )


def cached_author_check(model, fixture):
    """Compare only an exactly pinned reference's complete original numeric case."""
    data = dict(columns=model["source"]["columns"], values=model["source"]["values"])
    data_hash = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if data_hash != fixture["reference_data_sha256"]:
        raise ValueError("Cached original R fixture requires its complete exact numeric input case")
    raw = np.array(model["result"]["aft_parameters"])
    covariance = np.array(model["result"]["aft_covariance"])
    reference = fixture["actual_original_R"]
    errors = {
        "endpoint": float(
            np.linalg.norm((np.array(reference["theta"]) - raw) / np.sqrt(np.diag(covariance)))
        ),
        "full_covariance": normalized_error(reference["covariance"], covariance, covariance),
        "log_likelihood": abs(reference["log_likelihood"] - model["endpoint"]["log_likelihood"]),
    }
    return dict(
        errors=errors,
        passed=errors["endpoint"] <= 1e-5
        and errors["full_covariance"] <= 1e-6
        and errors["log_likelihood"] <= 1e-7,
        cached_actual_reference=True,
        new_actual_R_run=False,
        version=reference["version"],
        reference_data_sha256=data_hash,
    )


def read_document(path):
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            try:
                value = json.loads(line)
                if isinstance(value, dict) and "models" in value:
                    return value
            except json.JSONDecodeError:
                pass
    raise ValueError("No complete saved Weibull model/query JSON document found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-r", action="store_true")
    args = parser.parse_args()
    document = read_document(args.input)
    receipt = audit(document, run_r=not args.no_r)
    receipt["input_sha256"] = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": receipt["passed"], "checks": receipt["check_count"]}))
    raise SystemExit(0 if receipt["passed"] else 1)
