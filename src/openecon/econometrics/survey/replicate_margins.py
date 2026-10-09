"""Whole-PSU replication of nonlinear weighted empirical regression targets.

Every original and replicate model is fitted once. Its target uses that fit's
own sampling weights and physical support, so both coefficient and empirical
covariate uncertainty enter the saved joint target covariance.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from openecon.analysis_contracts import AnalysisError
from .common import FLOAT, MAX_TARGETS, finite, number
from .regression_common import cpu_call
from .regression_replication import _fit_replicate
from .replicate_margins_state import (
    SurveyReplicateMarginsResult, build_target_result, evaluate_targets, target_replay_work,
)


class _TargetTrace:
    """Private once-only target evaluator attached to the shared coefficient loop."""

    def __init__(self, *, family, target, profiles, variables, alpha, null):
        if not isinstance(family, str) or family not in {"logit", "probit", "poisson"}:
            raise AnalysisError(
                "invalid_survey_target", "Empirical target replication supports logit, probit, or poisson."
            )
        if not isinstance(target, str) or target not in {"mean", "ame"}:
            raise AnalysisError("invalid_survey_target", "target must be mean or ame.")
        if target == "mean" and variables is not None:
            raise AnalysisError("invalid_survey_target", "variables is an option only for target='ame'.")
        if target == "ame" and profiles is not None:
            raise AnalysisError("invalid_survey_target", "profiles is an option only for target='mean'.")
        self.family, self.target = family, target
        self.requested_profiles, self.requested_variables = profiles, variables
        self.alpha = number(alpha, "alpha", 1e-8, 1 - 1e-8)
        self.requested_null = null
        self.original = None
        self.replicas = []

    def _specification(self, sample):
        if self.target == "mean":
            profiles = self.requested_profiles
            if not isinstance(profiles, Mapping) or not 1 <= len(profiles) <= MAX_TARGETS:
                raise AnalysisError(
                    "invalid_survey_target", "Mean targets require 1..32 named partial profiles."
                )
            admitted = {}
            for label, settings in profiles.items():
                if (not isinstance(label, str) or not label.strip() or len(label) > 200
                        or not isinstance(settings, Mapping) or not settings
                        or len(settings) >= len(sample.regressors)
                        or any(not isinstance(name, str) or name not in sample.regressors
                               for name in settings)):
                    raise AnalysisError(
                        "invalid_survey_target",
                        "Each bounded named profile must fix fitted regressors and leave at least "
                        "one regressor at its observed empirical value.",
                    )
                admitted[label] = {name: number(value, "profile value")
                                   for name, value in settings.items()}
            self.profiles, self.variables = admitted, ()
            self.labels = tuple(admitted)
        else:
            variables = self.requested_variables
            if isinstance(variables, str):
                variables = [variables]
            if (not isinstance(variables, (list, tuple)) or not 1 <= len(variables) <= MAX_TARGETS
                    or any(not isinstance(name, str) or name not in sample.regressors
                           for name in variables)
                    or len(set(variables)) != len(variables)):
                raise AnalysisError(
                    "invalid_survey_target", "AME targets require distinct fitted continuous regressors."
                )
            self.profiles, self.variables = {}, tuple(variables)
            self.labels = tuple("AME:" + name for name in variables)
        if isinstance(self.requested_null, (list, tuple)):
            if len(self.requested_null) != len(self.labels):
                raise AnalysisError("invalid_survey_option", "One null per empirical target is required.")
            self.null = tuple(number(value, "target null") for value in self.requested_null)
        else:
            value = number(self.requested_null, "target null")
            self.null = (value,) * len(self.labels)

    def preflight(self, sample, count):
        """Reserve fitting, primitive storage and mandatory saved replay together."""
        self._specification(sample)
        n, n_used, width = len(sample.weights), int(sample.selected.sum()), len(sample.labels)
        targets, psus = len(self.labels), sample.design.validation.n_psu
        self.evaluation_work = n_used * width * targets * 8
        self.prepare_work = n * count * 8 + n_used * (width + 1) * 2 + count * psus * 2
        self.replay_work = target_replay_work(n, n_used, width, targets, count)
        work = self.prepare_work + (count + 1) * self.evaluation_work + self.replay_work
        return work, {
            "empirical target exact replica weights and primitive replay": n * count * 384,
            "empirical target original weights, groups and physical positions": n * 192,
            "empirical target selected covariates, responses and serialization":
                n_used * (width + 1) * 192,
            "empirical target PSU factors and validation": count * psus * 192,
            "empirical target replicas, joint covariance and saved replay":
                (count + 1) * targets * 192 + targets * targets * 512,
        }

    def prepare(self, sample, plan, ledger):
        """Retain exact admitted inputs before either fitting or target evaluation."""
        ledger.reserve(self.prepare_work)
        positions = torch.where(sample.selected)[0].tolist()
        self.X = sample.X[sample.selected]
        groups = sample.groups.tolist()
        first_rows = {}
        for row, psu in enumerate(groups):
            first_rows.setdefault(psu, row)
        first = [first_rows[psu] for psu in range(sample.design.validation.n_psu)]
        factors = finite(plan["weights"][:, first] / sample.weights[first], "Saved replicate PSU factors")
        self.primitive = {
            "X": self.X.tolist(), "y": sample.y[sample.selected].tolist(),
            "sample_positions": positions,
            "base_weights": sample.weights.tolist(), "groups": groups,
            "psu_factors": factors.tolist(), "replicate_weights": plan["weights"].tolist(),
            "sample_input_sha256": sample.metadata["sample_input_sha256"],
            "replicate_weight_sha256": plan["metadata"]["replicate_weight_sha256"],
        }
        self.coefficient_labels = tuple(sample.labels)
        self.regressors, self.intercept = tuple(sample.regressors), sample.intercept

    def evaluate(self, sample, fitted, *, replicate_id, ledger):
        ledger.reserve(self.evaluation_work)
        values = evaluate_targets(
            self.X, fitted["beta"], sample.weights, self.primitive["sample_positions"],
            family=self.family, labels=self.coefficient_labels, target=self.target,
            profiles=self.profiles, variables=self.variables, regressors=self.regressors,
            intercept=self.intercept,
        )
        values = finite(torch.as_tensor(values, dtype=FLOAT), "Empirical replicated target")
        if values.shape != (len(self.labels),):
            raise AnalysisError("invalid_survey_target", "The admitted empirical target dimensions changed.")
        if replicate_id is None:
            self.original = values
        else:
            self.replicas.append(values)

    def finish(self, source_result, plan, workspace, ledger):
        if self.original is None or len(self.replicas) != len(plan["ids"]):
            raise AnalysisError("survey_replicate_failure", "Empirical target replication is incomplete.")
        # Restoring the state checks the exact actual weights, original data
        # hash and every target value, then replays the full joint covariance.
        ledger.reserve(self.replay_work)
        work = {
            **ledger.record(),
            "primitive_preparation_work_units": self.prepare_work,
            "target_evaluation_work_units": (len(plan["ids"]) + 1) * self.evaluation_work,
            "target_replay_work_units": self.replay_work,
            "accounting": ledger.record()["accounting"] +
            " + exact target primitives + original/replica target evaluations + mandatory primitive replay",
        }
        return build_target_result(
            source_result, target=self.target, profiles=self.profiles, variables=self.variables,
            labels=self.labels, estimates=self.original.tolist(),
            replicate_estimates=torch.stack(self.replicas).tolist(), primitive=self.primitive,
            alpha=self.alpha, null=self.null, workspace=workspace, work=work,
        )


@cpu_call
def survey_margins_replicate(
    data, design, outcome, regressors, *, family, method, target="mean", profiles=None,
    variables=None, intercept=True, domain=None, missing="raise", alpha=0.05, null=0.0,
    max_iter=100, tolerance=1e-9, replicates=None, replicate_weights=None,
    centering="original", rho=None, justification=None, scale=None, rscales=None, df=None,
) -> SurveyReplicateMarginsResult:
    """Joint nonlinear empirical means or continuous AMEs with complete PSU refits.

    ``family`` is logit, probit, or poisson. Mean targets require a mapping of
    profile labels to nonempty partial regressor substitutions; each profile
    leaves at least one fitted regressor at its observed value. AME targets
    require explicit distinct continuous ``variables`` and reject profiles.

    Each original and replicate estimate averages over its own positive
    sampling weights and admitted physical support. The joint covariance
    includes both fitted coefficient uncertainty and uncertainty in the
    weighted empirical covariate distribution. It retains all off-diagonal
    entries, the declared replication method's centering and scales, and
    design-t inference. It does not provide future-outcome intervals or
    claim validated Stata parity.

    BRR/Fay, delete-one PSU jackknife and supplied whole-PSU bootstrap share
    the coefficient replication contract. Fay defaults to rho=0.5; bootstrap
    needs supplied weights, justification and variance scale. All fits and
    targets are evaluated once. Complete work and saved primitive replay
    storage are checked before the full replica weight allocation. Ordinary
    replica fit or target failures are returned together; safety budgets
    stop execution before the next operation. The typed JSON state retains
    exact target replicas and their original data/weight replay inputs.
    """
    trace = _TargetTrace(family=family, target=target, profiles=profiles, variables=variables,
                         alpha=alpha, null=null)
    return _fit_replicate(
        data, design, outcome, regressors, family=family, method=method, intercept=intercept,
        domain=domain, missing=missing, alpha=trace.alpha, null=0.0,
        max_iter=max_iter, tolerance=tolerance, replicates=replicates,
        replicate_weights=replicate_weights, centering=centering, rho=rho,
        justification=justification, scale=scale, rscales=rscales, df=df, _target_trace=trace,
    )
