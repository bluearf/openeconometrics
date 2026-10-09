"""Bounded marginal Gaussian GLS with estimated repeated residual covariance.

There are no random effects.  Each subject contributes its observed principal
submatrix of a single occasion covariance.  ML profiles the fixed effects;
REML uses ``log|X'V^-1X| - log|X'X|``.  Thus the REML log likelihood is invariant
to nonsingular changes of fixed-effect coordinates (and differs from software
which omits the latter constant by ``+.5 log|X'X|``).

All differentiation is the exact Torch chain rule.  ML uncertainty is the inverse
*joint* observed information in fixed and covariance coordinates, including cross
blocks.  REML provides conditional GLS fixed uncertainty and profile covariance
information separately; it is neither KR nor Satterthwaite inference.
"""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mixed.extended_common import Objective, maximize, optimizer_record


def _inverse_information(information):
    matrix = (information + information.T) / 2
    diagonal = matrix.diagonal()
    if not bool(torch.isfinite(matrix).all()) or not bool((diagonal > 0).all()):
        raise AnalysisError("boundary_solution", "Observed information is not positive definite.")
    scale = diagonal.rsqrt()
    lower, status = torch.linalg.cholesky_ex(matrix * scale[:, None] * scale)
    if int(status) or float(lower.diagonal().min()) < 1e-6:
        raise AnalysisError("boundary_solution", "Observed information is singular or indefinite.")
    return torch.cholesky_inverse(lower) * scale[:, None] * scale


def _integer_vector(value, name):
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1 or tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise AnalysisError("invalid_repeated_geometry", f"{name} must be an integer vector.")
    return tensor.to(device="cpu", dtype=torch.int64)


class _Geometry:
    def __init__(self, x, y, groups, occasions, levels, structure, method, max_work):
        if (not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor)
                or x.device.type != "cpu" or y.device.type != "cpu"
                or x.dtype != torch.float64 or y.dtype != torch.float64):
            raise AnalysisError("invalid_repeated_geometry", "x and y must be CPU float64 tensors.")
        if x.ndim != 2 or y.ndim != 1 or len(y) != len(x):
            raise AnalysisError("invalid_repeated_geometry", "Expected an n by p design and n-vector response.")
        self.n, self.p = x.shape
        if not 2 <= self.n <= 256 or not 1 <= self.p <= 6 or self.n <= self.p:
            raise AnalysisError("repeated_limit", "Repeated GLS permits n <= 256 and p <= 6 with n > p.")
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise AnalysisError("invalid_repeated_geometry", "The admitted design and response must be finite.")
        if structure not in ("cs", "ar1", "diagonal", "unstructured") or method not in ("ML", "REML"):
            raise AnalysisError("invalid_option", "Unknown repeated covariance structure or likelihood method.")
        if isinstance(max_work, bool) or not isinstance(max_work, int) or not 1 <= max_work <= 2_000_000_000:
            raise AnalysisError("work_budget", "max_work must be a positive integer at most two billion.")
        self.groups = _integer_vector(groups, "groups")
        self.occasions = _integer_vector(occasions, "occasions")
        self.levels = _integer_vector(levels, "levels")
        self.q = len(self.levels)
        if not 2 <= self.q <= 6 or not bool((self.levels[1:] > self.levels[:-1]).all()):
            raise AnalysisError("invalid_repeated_geometry", "Two to six sorted distinct integer occasions are required.")
        if len(self.groups) != self.n or len(self.occasions) != self.n:
            raise AnalysisError("invalid_repeated_geometry", "Subject and occasion indices must match the sample.")
        subjects = self.groups.unique(sorted=True)
        if not 4 <= len(subjects) <= 64 or not torch.equal(subjects, torch.arange(len(subjects))):
            raise AnalysisError("invalid_repeated_geometry", "Four to sixty-four contiguous subject index codes are required.")
        if not bool(((self.occasions >= 0) & (self.occasions < self.q)).all()):
            raise AnalysisError("invalid_repeated_geometry", "Occasion indices are outside the declared occasion grid.")
        self.structure, self.method, self.max_work = structure, method, max_work
        self.rows, self.cols = torch.tril_indices(self.q, self.q)
        self.diagonal_theta = self.rows == self.cols
        self.k = self.q * (self.q + 1) // 2 if structure == "unstructured" else self.q if structure == "diagonal" else 2
        # Scaling and an exact fixed-space shift remove large fitted means from
        # the covariance optimization while retaining the original likelihood.
        self.x_scale = x.square().mean(0).sqrt()
        if not bool((self.x_scale > 0).all()):
            raise AnalysisError("singular_design", "The fixed design contains a zero column.")
        self.x = x / self.x_scale
        singular = torch.linalg.svdvals(self.x)
        if float(singular[-1]) <= float(singular[0]) * 1e-8:
            raise AnalysisError("singular_design", "The normalized fixed design is rank deficient or ill conditioned.")
        self.ols = torch.linalg.lstsq(self.x, y[:, None], driver="gels").solution[:, 0]
        residual = y - self.x @ self.ols
        self.y_scale = float((residual.square().sum() / (self.n - self.p)).sqrt())
        if not math.isfinite(self.y_scale) or self.y_scale <= torch.finfo(torch.float64).tiny:
            raise AnalysisError("boundary_solution", "The response has no positive residual variation.")
        self.y = residual / self.y_scale
        self.log_xx = torch.linalg.slogdet(self.x.T @ self.x)[1]
        patterns = {}
        pairs = set()
        for group in subjects.tolist():
            indices = torch.nonzero(self.groups == group).flatten()
            indices = indices[torch.argsort(self.occasions[indices])]
            observed = self.occasions[indices].tolist()
            if len(observed) != len(set(observed)):
                raise AnalysisError("invalid_repeated_geometry", "A subject has duplicate observations at one occasion.")
            patterns.setdefault(tuple(observed), []).append(indices)
            pairs.update((a, b) for a in observed for b in observed if a < b)
        if any(int((self.occasions == occasion).sum()) < 4 for occasion in range(self.q)):
            raise AnalysisError("covariance_identification", "Each occasion requires at least four subjects.")
        if structure == "unstructured":
            counts = {pair: 0 for pair in pairs}
            for observed, indices in patterns.items():
                for pair in counts:
                    if pair[0] in observed and pair[1] in observed:
                        counts[pair] += len(indices)
            if len(counts) != self.q * (self.q - 1) // 2 or min(counts.values()) < 4:
                raise AnalysisError("covariance_identification", "Every unstructured occasion pair requires four subjects.")
        if structure == "ar1":
            lags = [int(self.levels[b]) - int(self.levels[a]) for a, b in pairs]
            if not lags or math.gcd(*lags) != 1:
                raise AnalysisError("covariance_identification", "Signed AR1 requires greatest common observed integer lag one.")
        if structure == "cs" and not pairs:
            raise AnalysisError("covariance_identification", "CS correlation requires repeated observations.")
        self.patterns = [(torch.tensor(observed, dtype=torch.int64), torch.stack(indices))
                         for observed, indices in patterns.items()]
        self.gaps = (self.levels[:, None] - self.levels).abs()
        self.integer_gaps = self.gaps.flatten().tolist()
        self.work = 8 * sum(len(indices) * (len(observed) ** 3 + len(observed) * (self.p + 1) ** 2)
                            for observed, indices in self.patterns) + 8 * (self.p + self.k) ** 3
        self.objective = Objective(self.value, self.work, max_work)

    def covariance(self, theta):
        if self.structure == "diagonal":
            return torch.diag(torch.exp(2 * theta))
        if self.structure == "unstructured":
            entries = torch.where(self.diagonal_theta, torch.exp(theta), theta)
            lower = torch.zeros((self.q, self.q), dtype=torch.float64).index_put((self.rows, self.cols), entries)
            return lower @ lower.T
        variance = torch.exp(2 * theta[0])
        if self.structure == "cs":
            bound = -1 / (self.q - 1)
            rho = bound + (1 - bound) * torch.sigmoid(theta[1])
            return variance * (rho * torch.ones((self.q, self.q), dtype=torch.float64)
                               + (1 - rho) * torch.eye(self.q, dtype=torch.float64))
        rho = torch.tanh(theta[1])
        # Scalar integer powers preserve the exact polynomial Hessian at rho=0.
        # Torch's tensor-exponent derivative otherwise forms 0 * rho**(-1)
        # on diagonal and first-lag entries in its second derivative.
        powers = torch.stack([rho.pow(gap) for gap in self.integer_gaps]).reshape(self.q, self.q)
        return variance * powers

    def components(self, theta, beta=None):
        occasion_covariance = self.covariance(theta)
        if not bool(torch.isfinite(occasion_covariance).all()):
            return None
        bread = torch.zeros((self.p, self.p), dtype=torch.float64)
        score = torch.zeros(self.p, dtype=torch.float64)
        logdet = torch.zeros((), dtype=torch.float64)
        blocks = []
        for observed, indices in self.patterns:
            covariance = occasion_covariance[observed[:, None], observed]
            lower, status = torch.linalg.cholesky_ex(covariance)
            if int(status):
                return None
            xx, yy = self.x[indices], self.y[indices]
            solved_x = torch.cholesky_solve(xx, lower)
            solved_y = torch.cholesky_solve(yy[:, :, None], lower)[:, :, 0]
            bread = bread + (xx.transpose(1, 2) @ solved_x).sum(0)
            score = score + (xx * solved_y[:, :, None]).sum((0, 1))
            logdet = logdet + len(indices) * 2 * lower.diagonal().log().sum()
            blocks.append((xx, yy, lower))
        fixed_lower, status = torch.linalg.cholesky_ex(bread)
        if int(status):
            return None
        if beta is None:
            beta = torch.cholesky_solve(score[:, None], fixed_lower)[:, 0]
        quadratic = torch.zeros((), dtype=torch.float64)
        for xx, yy, lower in blocks:
            residual = yy - (xx @ beta[:, None])[:, :, 0]
            quadratic = quadratic + (residual * torch.cholesky_solve(residual[:, :, None], lower)[:, :, 0]).sum()
        logdet_fixed = 2 * fixed_lower.diagonal().log().sum()
        ml = -.5 * (self.n * math.log(2 * math.pi) + logdet + quadratic)
        reml = -.5 * ((self.n - self.p) * math.log(2 * math.pi) + logdet + logdet_fixed - self.log_xx + quadratic)
        return beta, torch.cholesky_inverse(fixed_lower), occasion_covariance, quadratic, ml, reml

    def value(self, theta):
        pieces = self.components(theta)
        if pieces is None:
            return theta.sum() * math.nan
        return pieces[4 if self.method == "ML" else 5]

    def raw_theta(self, theta):
        if self.structure == "unstructured":
            return torch.where(self.diagonal_theta, theta + math.log(self.y_scale), theta * self.y_scale)
        if self.structure == "diagonal":
            return theta + math.log(self.y_scale)
        return torch.stack((theta[0] + math.log(self.y_scale), theta[1]))

    def normalized_theta(self, theta):
        if self.structure == "unstructured":
            return torch.where(self.diagonal_theta, theta - math.log(self.y_scale), theta / self.y_scale)
        if self.structure == "diagonal":
            return theta - math.log(self.y_scale)
        return torch.stack((theta[0] - math.log(self.y_scale), theta[1]))

    def parameter_report(self, raw_theta):
        covariance = self.covariance(raw_theta)
        values = [covariance[i, i] for i in range(self.q)]
        if self.structure in ("cs", "ar1"):
            return torch.stack((covariance[0, 0],
                                -1 / (self.q - 1) + (1 + 1 / (self.q - 1)) * torch.sigmoid(raw_theta[1])
                                if self.structure == "cs" else torch.tanh(raw_theta[1])))
        if self.structure == "unstructured":
            values += [covariance[i, j] for i in range(self.q) for j in range(i)]
        return torch.stack(values)

    def outputs(self, theta):
        self.objective.charge(2 * (self.k + 1))
        with torch.enable_grad():
            hessian = torch.autograd.functional.hessian(self.value, theta)
            point = theta.detach().requires_grad_()
            gradient = torch.autograd.grad(self.value(point), point)[0].detach()
        theta_covariance = _inverse_information(-hessian)
        scaled_gradient = float(gradient @ theta_covariance @ gradient)
        if not math.isfinite(scaled_gradient) or scaled_gradient > 1e-8:
            raise AnalysisError("nonconvergence", "Saved covariance parameters are not a stationary likelihood fit.")
        pieces = self.components(theta)
        if pieces is None:
            raise AnalysisError("boundary_solution", "Estimated residual covariance is not positive definite.")
        beta, gls, covariance, quadratic, ml, reml = pieces
        sd = covariance.diagonal().sqrt()
        correlation = covariance / sd[:, None] / sd
        if (not bool(torch.isfinite(correlation).all()) or float(torch.linalg.eigvalsh(correlation)[0]) < 1e-8
                or float(sd.min() / sd.max()) < 1e-6):
            raise AnalysisError("boundary_solution", "Estimated residual covariance is at a numerical boundary.")
        beta_scale = self.y_scale / self.x_scale
        theta_scale = torch.ones(self.k, dtype=torch.float64)
        if self.structure == "unstructured":
            theta_scale = torch.where(self.diagonal_theta, 1.0, self.y_scale)
        joint_covariance = None
        beta_theta_covariance = None
        conditional = gls * beta_scale[:, None] * beta_scale
        fixed_covariance = conditional
        if self.method == "ML":
            def joint_value(parameters):
                components = self.components(parameters[self.p:], parameters[:self.p])
                return parameters.sum() * math.nan if components is None else components[4]
            self.objective.charge(2 * (self.p + self.k + 1))
            with torch.enable_grad():
                joint_hessian = torch.autograd.functional.hessian(joint_value, torch.cat((beta.detach(), theta)))
            joint_covariance = _inverse_information(-joint_hessian)
            joint_scale = torch.cat((beta_scale, theta_scale))
            joint_covariance = joint_covariance * joint_scale[:, None] * joint_scale
            fixed_covariance = joint_covariance[:self.p, :self.p]
            beta_theta_covariance = joint_covariance[:self.p, self.p:]
            theta_covariance = joint_covariance[self.p:, self.p:] / theta_scale[:, None] / theta_scale
        theta_covariance = theta_covariance * theta_scale[:, None] * theta_scale
        raw_theta = self.raw_theta(theta).detach()
        with torch.enable_grad():
            jacobian = torch.autograd.functional.jacobian(self.parameter_report, raw_theta)
        transformed_covariance = jacobian @ theta_covariance @ jacobian.T
        joint_raw_covariance = joint_covariance
        if joint_covariance is not None:
            joint_jacobian = torch.block_diag(torch.eye(self.p, dtype=torch.float64), jacobian)
            joint_covariance = joint_jacobian @ joint_covariance @ joint_jacobian.T
        terms = [f"variance[{int(level)}]" for level in self.levels]
        if self.structure in ("cs", "ar1"):
            terms = ["variance", "rho"]
        elif self.structure == "unstructured":
            terms += [f"covariance[{int(self.levels[i])},{int(self.levels[j])}]"
                      for i in range(self.q) for j in range(i)]
        residual_covariance = torch.zeros((self.n, self.n), dtype=torch.float64)
        for observed, indices in self.patterns:
            block = covariance[observed[:, None], observed] * self.y_scale ** 2
            for rows in indices:
                residual_covariance[rows[:, None], rows] = block
        return {
            "beta": ((self.ols + self.y_scale * beta) / self.x_scale).detach().tolist(),
            "covariance": fixed_covariance.detach().tolist(),
            "conditional_gls_covariance": conditional.detach().tolist(),
            "theta": raw_theta.tolist(), "theta_covariance": theta_covariance.detach().tolist(),
            "covariance_parameters": self.parameter_report(raw_theta).detach().tolist(),
            "covariance_terms": terms,
            "covariance_parameter_covariance": transformed_covariance.detach().tolist(),
            "occasion_covariance": (covariance * self.y_scale ** 2).detach().tolist(),
            "residual_covariance": residual_covariance.tolist(),
            "joint_covariance": None if joint_covariance is None else joint_covariance.detach().tolist(),
            "joint_raw_covariance": None if joint_raw_covariance is None else joint_raw_covariance.detach().tolist(),
            "beta_theta_covariance": None if beta_theta_covariance is None else beta_theta_covariance.detach().tolist(),
            "loglik_ml": float(ml) - self.n * math.log(self.y_scale),
            "loglik_reml": float(reml) - (self.n - self.p) * math.log(self.y_scale),
            "rss": float(quadratic),
            "inference": "ML full joint observed information; normal Wald fixed effects" if self.method == "ML"
                         else "REML conditional GLS plug-in fixed covariance; normal Wald; no KR/Satterthwaite",
            "covariance_inference": "full profile observed information and exact delta method",
            "reml_normalization": "logdet(X'V^-1X) minus logdet(X'X)",
            "geometry": {"x_scale": self.x_scale.tolist(), "y_scale": self.y_scale,
                         "ols_normalized_design": self.ols.tolist()},
            "stationarity": {"scaled_gradient": scaled_gradient, "gradient_normalized_theta": gradient.tolist()},
        }


def fit(x, y, groups, occasions, levels, structure, method, max_work):
    """Fit the admitted geometry and return JSON-safe full scientific outputs."""
    geometry = _Geometry(x, y, groups, occasions, levels, structure, method, max_work)
    if structure == "unstructured":
        initial = torch.zeros(geometry.k, dtype=torch.float64)
    else:
        initial = torch.zeros(geometry.k, dtype=torch.float64)
        if structure == "cs":
            initial[1] = -math.log(geometry.q - 1)  # rho zero
    starts = [initial]
    if structure in ("cs", "ar1"):
        for correlation in (-.65, -.25, .25, .65):
            if structure == "cs":
                lower = -1 / (geometry.q - 1)
                if correlation <= lower:
                    continue
                fraction = (correlation - lower) / (1 - lower)
                eta = math.log(fraction / (1 - fraction))
            else:
                eta = math.atanh(correlation)
            start = initial.clone()
            start[1] = eta
            starts.append(start)
    fitted, _ = maximize(geometry.objective, starts)
    output = geometry.outputs(fitted.theta)
    output["optimizer"] = optimizer_record(fitted, geometry.objective)
    return output


def replay(x, y, groups, occasions, levels, structure, method, theta, max_work):
    """Recompute outputs and stationarity at saved raw theta, without optimization."""
    geometry = _Geometry(x, y, groups, occasions, levels, structure, method, max_work)
    raw = torch.as_tensor(theta, dtype=torch.float64)
    if raw.ndim != 1 or len(raw) != geometry.k or not bool(torch.isfinite(raw).all()):
        raise AnalysisError("invalid_repeated_state", "Saved covariance theta has invalid dimensions or values.")
    return geometry.outputs(geometry.normalized_theta(raw))
