"""Independent development references for the original teaching examples."""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import integrate, optimize
import statsmodels.api as sm
import verify_teaching_labs as shared


def compare_model(payload, fit, terms, positions=None):
    """Compare every coefficient, covariance entry, inference and saved prediction."""
    order = [terms.index(row["term"]) for row in payload["coefficients"]]
    interval = np.asarray(fit.conf_int())
    for row, i in zip(payload["coefficients"], order):
        for key, values in [
            ("estimate", fit.params),
            ("std_error", fit.bse),
            ("statistic", fit.tvalues),
            ("p_value", fit.pvalues),
            ("ci_low", interval[:, 0]),
            ("ci_high", interval[:, 1]),
        ]:
            shared.compare(float(np.asarray(values)[i]), row[key], f"reference.{row['term']}.{key}")
    shared.compare(
        np.asarray(fit.cov_params())[order][:, order].tolist(),
        payload["covariance_matrix"],
        "reference.full_covariance",
    )
    shared.require(int(fit.nobs) == payload["nobs"], "Independent sample count differs")
    if positions is not None:
        shared.require(
            payload["sample_positions"] == positions, "Independent retained positions differ"
        )
    for row in payload["predictions"]:
        i = payload["sample_positions"].index(row["row"])
        shared.compare(float(np.asarray(fit.predict())[i]), row["fitted"], "reference.fitted")
        shared.compare(
            float(np.asarray(fit.resid if hasattr(fit, "resid") else fit.resid_response)[i]),
            row["residual"],
            "reference.residual",
        )
    return {
        "statsmodels_full_coefficients_covariance_inference": True,
        "statsmodels_fitted_and_residual_state": True,
        "independent_sample_count": True,
    }


def microeconomics(folder, reference):
    meta = json.loads((folder / "dataset.json").read_text())
    data = pd.read_excel(folder / meta["filename"], sheet_name="Data")
    n = int(folder.name[:2])
    s = reference["summary"]
    checks = {}

    def equal(label, value, key):
        shared.compare(float(value), s[key], label)
        checks[label] = True

    if n == 1:
        h = 1e-4
        x = s["chosen_x"]
        derivative = (2 * np.sqrt(1600 - (x + h) ** 2) - 2 * np.sqrt(1600 - (x - h) ** 2)) / (2 * h)
        equal("central_difference_opportunity_cost", -derivative, "marginal_y_cost_per_x")
    elif n == 2:
        equal("A_technology_ratio", data.labor_x.iloc[0] / data.labor_y.iloc[0], "A_y_cost_per_x")
        equal("B_technology_ratio", data.labor_x.iloc[1] / data.labor_y.iloc[1], "B_y_cost_per_x")
    elif n == 3:
        root = optimize.brentq(lambda p: 120 - 2 * p - (-20 + 2 * p), 10, 60)
        equal("scipy_market_clearing_root", root, "equilibrium_price")
    elif n == 4:
        optimum = optimize.minimize_scalar(
            lambda p: -p * (120 - 2 * p), bounds=(1, 59), method="bounded"
        )
        equal("scipy_revenue_maximum", -optimum.fun, "maximum_revenue")
    elif n == 5:
        optimum = optimize.minimize_scalar(
            lambda x: -(x**0.4) * ((120 - 3 * x) / 2) ** 0.6,
            bounds=(0.001, 39.999),
            method="bounded",
            options={"xatol": 1e-12},
        )
        equal("scipy_maximum_utility", -optimum.fun, "utility_index")
    elif n == 6:
        equal("constant_share_high_income_demand", 0.4 * 150 / 3, "high_income_x")
    elif n == 7:
        u = s["original_utility"]
        # Expenditure minimization over the isoquant is a separate optimization.
        result = optimize.minimize_scalar(
            lambda x: 6 * x + 2 * (u / x**0.4) ** (1 / 0.6), bounds=(0.01, 80), method="bounded"
        )
        equal("scipy_compensated_minimum_expenditure", result.fun, "hicks_compensated_income")
    elif n == 8:
        cost = data[["px", "py"]].to_numpy() @ data[["x", "y"]].to_numpy().T
        affordable = cost <= data.income.to_numpy()[:, None] + 1e-10
        reverse_strict = cost.T < data.income.to_numpy()[None, :] - 1e-10
        equal(
            "vectorized_strict_affordability_cycles",
            np.sum(affordable & reverse_strict),
            "warp_violations",
        )
    elif n == 9:
        result = optimize.minimize_scalar(
            lambda labor: -(20 * np.sqrt(labor) - 5 * labor - 10), bounds=(0, 100), method="bounded"
        )
        equal("scipy_profit_maximum", -result.fun, "profit_after_fixed_cost")
    elif n == 10:
        result = optimize.minimize_scalar(
            lambda q: 100 / q + 2 + 0.1 * q, bounds=(1, 100), method="bounded"
        )
        equal("scipy_minimum_average_cost", result.fun, "continuous_min_atc")
    elif n == 11:
        q = s["base_output"]
        result = optimize.minimize_scalar(
            lambda labor: 4 * labor + q * q / (4 * labor), bounds=(0.01, 100), method="bounded"
        )
        equal("scipy_conditional_cost_minimum", result.fun, "base_cost")
    elif n == 12:
        result = optimize.minimize_scalar(
            lambda q: -(12 * q - 100 - 2 * q - 0.1 * q * q), bounds=(0, 100), method="bounded"
        )
        equal("scipy_competitive_profit_maximum", -result.fun, "profit_at_price_12")
    elif n == 13:
        result = optimize.minimize_scalar(
            lambda q: -((100 - q) * q - 20 * q - 200), bounds=(0, 100), method="bounded"
        )
        equal("scipy_monopoly_profit_maximum", -result.fun, "monopoly_profit")
        equal(
            "scipy_lost_gains_integral",
            integrate.quad(lambda q: 80 - q, 40, 80)[0],
            "deadweight_loss",
        )
    elif n in (14, 15, 16):
        q = s.get("traded_quantity", s.get("taxed_quantity", s.get("subsidized_quantity")))
        total = integrate.quad(lambda x: 50 - x, 0, q)[0]
        equal(
            "scipy_net_gains_integral",
            total,
            "total_surplus" if n == 14 else "net_total_surplus" if n == 16 else "consumer_surplus",
        ) if n != 15 else equal("scipy_lost_gains_integral", 1250 - total, "deadweight_loss")
    elif n == 17:
        q = optimize.brentq(lambda x: 60 - 0.5 * x - (20 + 0.5 * x), 0, 100)
        equal("scipy_social_marginal_condition", q, "social_optimal_quantity")
        equal(
            "scipy_net_social_integral",
            integrate.quad(lambda x: 40 - x, 0, q)[0],
            "welfare_at_social_optimum",
        )
    elif n == 18:
        q = optimize.brentq(lambda x: 40 - x + 30 - 0.5 * x - 25, 0, 40)
        equal("scipy_public_good_root", q, "efficient_shared_quantity")
    elif n == 19:
        root = optimize.root(lambda q: [80 - 2 * q[0] - q[1], 80 - q[0] - 2 * q[1]], [20, 20])
        shared.require(root.success, "Independent best-response solver failed")
        equal("scipy_nash_best_response_system", root.x[0], "duopoly_output_per_firm")
    elif n == 20:
        X = sm.add_constant(np.log(data[["price", "income"]]).to_numpy())
        fit = sm.OLS(np.log(data.quantity), X).fit(cov_type="HC3", use_t=True)
        checks.update(
            compare_model(
                reference["models"]["demand"],
                fit,
                ["Intercept", "log_price", "log_income"],
                list(range(len(data))),
            )
        )
    shared.require(bool(checks), "No independent microeconomics reference exercised")
    return checks


def advanced_econometrics(folder, reference):
    """Rebuild each statistical design independently from the prepared rows."""
    from statsmodels.sandbox.regression.gmm import IV2SLS

    meta = json.loads((folder / "dataset.json").read_text())
    raw = pd.read_excel(folder / meta["filename"], sheet_name="Data")
    number = int(folder.name[:2])
    checks = {}
    for name, payload in reference["models"].items():
        data = raw.copy()
        y = "y"
        predictors = ["x", "z"]
        covariance = "HC3"
        if number == 1:
            if name == "outcome_on_controls":
                predictors = ["z"]
            if name == "exposure_on_controls":
                predictors = ["z"]
                y = "x"
        if number == 2:
            if name == "short":
                predictors = ["x"]
            if name == "z_on_x":
                predictors = ["x"]
                y = "z"
        if number == 3:
            predictors = ["x" if name == "latent_benchmark" else "observed_x"]
        if number == 5 and name == "without_flagged_row":
            data = data.iloc[1:].reset_index(drop=True)
        if number == 6:
            covariance = "nonrobust"
        if number == 7 and name == "clustered":
            covariance = "cluster"
        if number == 8 and name == "hac_bartlett_4":
            covariance = "HAC"
        if number == 11:
            data = raw.dropna(
                subset=["y", "x"] if name == "short_available" else ["y", "x", "z"]
            ).reset_index(drop=True)
            predictors = ["x", "z"] if name == "full_common" else ["x"]
        if number == 12:
            if name == "raw_polynomial":
                data["x2"] = data.x**2
                predictors = ["x", "x2", "z"]
            else:
                data["xc"] = data.x - data.x.mean()
                data["xc2"] = data.xc**2
                predictors = ["xc", "xc2", "z"]
        if number == 13:
            predictors = ["x", "z"]
            for unit in sorted(data.unit.unique())[1:]:
                column = f"unit_{unit}"
                data[column] = (data.unit == unit).astype(float)
                predictors.append(column)
            covariance = "cluster"
        if number == 14:
            data = raw.sort_values(["unit", "time"]).copy()
            data[["y", "x", "z"]] = data.groupby("unit")[["y", "x", "z"]].diff()
            data = data.dropna(subset=["y", "x", "z"]).reset_index(drop=True)
            covariance = "cluster"
        if number in (15, 16):
            covariance = "nonrobust"
            predictors = ["control", "x"]
            if name.startswith("weak"):
                data["y"] = raw.y - 1.2 * raw.x + 1.2 * raw.weak_x
                data["x"] = raw.weak_x
            if name in ("first_stage", "strong_first_stage", "weak_first_stage"):
                y = "x"
                predictors = ["control", "z"]
        if number == 19:
            horizon = int(name[1:])
            data["lead_y"] = data.y.shift(-horizon)
            data = data.iloc[1:].dropna(subset=["lead_y"]).reset_index(drop=True)
            y = "lead_y"
            covariance = "HAC"
        if number == 20:
            predictors = ["treat"] if name == "unadjusted" else ["treat", "x", "z"]
            if name == "heterogeneous":
                predictors.append("interaction")
        X = sm.add_constant(data[predictors].to_numpy())
        terms = ["Intercept"] + predictors
        if number in (15, 16) and name in ("two_stage_least_squares", "strong_iv", "weak_iv"):
            Z = sm.add_constant(data[["control", "z"]].to_numpy())
            fitted = IV2SLS(data.y.to_numpy(), X, Z).fit()
        elif number == 17:
            fitted = sm.Logit(data.y.to_numpy(), X).fit(disp=False, tol=1e-12, maxiter=100)
        elif number == 18:
            fitted = sm.GLM(
                data.y.to_numpy(),
                X,
                family=sm.families.Poisson(),
                offset=np.log(data.exposure.to_numpy()),
            ).fit(tol=1e-12, maxiter=100)
        else:
            model = (
                sm.WLS(data[y].to_numpy(), X, weights=data.weight.to_numpy())
                if number == 6 and name == "known_variance_wls"
                else sm.OLS(data[y].to_numpy(), X)
            )
            options = {}
            if covariance == "cluster":
                options = {
                    "cov_kwds": {
                        "groups": data["group" if number == 7 else "unit"].to_numpy(),
                        "use_correction": True,
                        "df_correction": True,
                    }
                }
            elif covariance == "HAC":
                options = {"cov_kwds": {"maxlags": 4, "use_correction": True}}
            fitted = model.fit(cov_type=covariance, use_t=True, **options)
        checks[name] = compare_model(payload, fitted, terms, list(range(len(data))))
        if number == 9:
            test = fitted.f_test(np.array([[0, 1, 0], [0, 0, 1]]))
            shared.compare(
                float(test.fvalue), reference["summary"]["joint_F"], "independent_joint_F"
            )
            shared.compare(
                float(test.pvalue), reference["summary"]["joint_p"], "independent_joint_p"
            )
            checks["joint_restriction_inference"] = True
        if number == 10:
            b = np.asarray(fitted.params)
            gradient = np.array([0, 1 / b[2], -b[1] / b[2] ** 2])
            se = np.sqrt(gradient @ fitted.cov_params() @ gradient)
            shared.compare(
                float(se), reference["summary"]["delta_se"], "independent_nonlinear_delta_se"
            )
            checks["nonlinear_delta_variance"] = True
        if number == 17:
            probability = fitted.predict(X)
            effect = fitted.params[1] * np.mean(probability * (1 - probability))
            shared.compare(
                float(effect),
                reference["summary"]["average_marginal_effect_x"],
                "independent_logit_ame",
            )
            checks["average_probability_derivative"] = True
        if number == 20 and name == "heterogeneous":
            contrast = np.array([0, 1, 0, 0, data.x.mean()])
            test = fitted.t_test(contrast)
            shared.compare(
                float(np.asarray(test.effect).item()),
                reference["summary"]["sample_average_effect"],
                "independent_average_effect",
            )
            shared.compare(
                float(np.asarray(test.sd).item()),
                reference["summary"]["average_effect_se"],
                "independent_average_effect_se",
            )
            shared.compare(
                np.asarray(test.conf_int())[0].tolist(),
                [
                    reference["summary"]["average_effect_ci_low"],
                    reference["summary"]["average_effect_ci_high"],
                ],
                "independent_average_effect_interval",
            )
            checks["average_effect_full_covariance_contrast"] = True
    shared.require(bool(checks), "No independent advanced reference exercised")
    return checks
