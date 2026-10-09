"""Generate deterministic, source-backed current scope; --check refuses drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
JSON_PATH = ROOT / "docs/econometrics/capabilities.generated.json"
MD_PATH = ROOT / "docs/capabilities.md"
SCOPE_DOCUMENTS = (
    "README.md",
    "docs/architecture.md",
    "docs/streaming.md",
    "docs/econometrics/coverage.md",
    "desktop/README.md",
    "docs/agent-integration.md",
)
SCOPE_BEGIN = "<!-- BEGIN source-generated capability scope -->"
SCOPE_END = "<!-- END source-generated capability scope -->"


def scope_block(value: dict, document: str) -> str:
    """One source contract shared by entry-point guides, without release claims."""
    target = os.path.relpath("docs/capabilities.md", Path(document).parent).replace(os.sep, "/")
    prediction = value["postestimation"]["saved_prediction"]
    return "\n".join(
        [
            SCOPE_BEGIN,
            f"Current source: **{len(value['estimators'])} registered fit names**, "
            f"**{len(value['streaming']['estimators'])} Dataset fit routes**, "
            f"**{prediction['estimator_count']} common saved predict/margins adapters**. "
            f"The [generated method/option inventory]({target}) states conditions, exclusions and devices.",
            "",
            "Fit routes, common prediction adapters and family-specific helpers/forecasts have separate contracts. "
            "Source implementation does not establish independent scientific validation, installed-package "
            "verification or public shipment for a method/option. Those require their own dated, "
            "source-pinned evidence; historical measurements retain their original scope.",
            SCOPE_END,
        ]
    )


def scope_document(text: str, value: dict, document: str) -> str:
    if text.count(SCOPE_BEGIN) != 1 or text.count(SCOPE_END) != 1:
        raise ValueError(f"{document}: require exactly one generated capability scope block")
    before, remaining = text.split(SCOPE_BEGIN)
    _, after = remaining.split(SCOPE_END)
    return before + scope_block(value, document) + after


def unsupported_scope_claims(text: str) -> list[str]:
    """Catch forbidden universal route claims outside generated/historical scope.

    This is a narrow wording guard, not a semantic correctness proof. Generated
    blocks and their source comparison are the primary metadata drift guard.
    Dated measurements are preserved, including their original module counts.
    """
    text = re.sub(re.escape(SCOPE_BEGIN) + r".*?" + re.escape(SCOPE_END), "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text.replace("`", ""))
    pattern = (
        r"\b(?:All|Every)\s+(?:\d+\s+)?(?:currently\s+)?(?:registered\s+)?"
        r"(?:estimator names|estimators|model names|models)\b"
        r"[^.!?]{0,160}\b(?:Dataset|replayable|streaming|GPU|CUDA|parity|predict/margins)\b"
    )
    return re.findall(pattern, text, flags=re.I)


def snapshot(source_ref: str) -> dict:
    from openecon.analysis_contracts import capabilities

    cap = capabilities()
    source_files = sorted(
        {
            "src/openecon/analysis_contracts.py",
            "src/openecon/dataset.py",
            "src/openecon/dataset_prepare.py",
            "src/openecon/source_readers.py",
            "src/openecon/prediction_capabilities.py",
            "src/openecon/econometrics/registry.py",
            "src/openecon/econometrics/streaming_registry.py",
            "src/openecon/linear_ols/spec.py",
            "src/openecon/resources.py",
            "src/openecon/engines/separation.py",
            "src/openecon/survey.py",
            "pyproject.toml",
            "desktop/src-tauri/tauri.conf.json",
            "packages/openecon-charts/pyproject.toml",
            *[
                str(path.relative_to(ROOT))
                for path in (ROOT / "src/openecon/econometrics").glob("*/__init__.py")
            ],
            *[
                str(path.relative_to(ROOT))
                for path in (ROOT / "src/openecon/econometrics/mi").glob("*.py")
            ],
            *[
                str(path.relative_to(ROOT))
                for path in (ROOT / "src/openecon/econometrics/control_function").glob("*.py")
            ],
        }
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    charts = tomllib.loads((ROOT / "packages/openecon-charts/pyproject.toml").read_text())
    native = json.loads((ROOT / "desktop/src-tauri/tauri.conf.json").read_text())
    return {
        "schema": 1,
        "source_ref": source_ref,
        "source_fingerprints": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files
        },
        "versions": {
            "sdk": project["project"]["version"],
            "native": native["version"],
            "charts": charts["project"]["version"],
        },
        "estimators": cap["estimators"],
        "families": cap["families"],
        "auxiliary_exports": cap["auxiliary_exports"],
        "streaming": cap["streaming"],
        "postestimation": cap["postestimation"],
        "multiple_testing": cap["multiple_testing"],
        "measurement_reliability": cap["measurement_reliability"],
        "missing_data": cap["missing_data"],
        "control_function": cap["control_function"],
        "timeseries_workflows": cap["timeseries_workflows"],
        "state_space_paths": cap["state_space_paths"],
        "causal_design": cap["causal_design"],
        "conjoint": cap["conjoint"],
        "finite_regression": cap["finite_regression"],
        "conditional_regression": cap["conditional_regression"],
        "fractional_memory": cap["fractional_memory"],
        "meta_analysis": cap["meta_analysis"],
        "repeated_residual_gls": cap["repeated_residual_gls"],
        "inference_extensions": cap["inference_extensions"],
        "native_analysis_extensions": cap["native_analysis_extensions"],
        "prospective_planning": cap["prospective_planning"],
        "irt": cap["irt"],
        "categorical_scaling_loglinear": cap["categorical_scaling_loglinear"],
        "twostep_clustering": cap["twostep_clustering"],
        "multivariate_options": cap["multivariate_options"],
        "survey_design": cap["survey_design"],
        "survey_inference": cap["survey_inference"],
        "survey_regression": cap["survey_regression"],
        "survey_two_stage": cap["survey_two_stage"],
        "survey_four_stage": cap["survey_four_stage"],
        "survey_fully_stratified_three_stage": cap["survey_fully_stratified_three_stage"],
        "prospective_power_designs": cap["prospective_power_designs"],
        "temporal_disaggregation": cap["temporal_disaggregation"],
        "survival_extensions": cap["survival_extensions"],
        "binary_mediation": cap["binary_mediation"],
        "nested_choice": cap["nested_choice"],
        "multinomial_probit_choice": cap["multinomial_probit_choice"],
        "rank_ordered_choice": cap["rank_ordered_choice"],
        "evidence_scope": {
            "claim_policy": "implementation metadata; not a scientific, installed or public-release receipt",
            "per_estimator": {
                name: {
                    "source_fit": True,
                    "source_dataset_fit": name in cap["streaming"]["estimators"],
                    "source_common_prediction": name
                    in cap["postestimation"]["saved_prediction"]["estimators"],
                    "option_contract": f"estimators.{name}",
                    "independent_scientific_validation": "requires method/option-specific reference and dated receipt",
                    "installed_package_verification": "requires exact package/source hash and method/option execution receipt",
                    "public_shipment": "requires accessible release/source manifest and matching method/option package pin",
                }
                for name in sorted(cap["estimators"])
            },
            "helpers": "separately catalogued procedure contracts; not additional registered fit or common prediction routes",
        },
        "execution": cap["execution"],
        "precision": cap["precision"],
        "input": cap["input"],
        "dataset_preparation": cap["dataset_preparation"],
        "workspace": cap["workspace"],
        "stata_parity_validated": cap["stata_parity_validated"],
        "cuda_hardware_validated": cap["network"]["cuda_hardware_validated"],
        "dependencies": {
            "runtime": project["project"]["dependencies"],
            "optional": project["project"]["optional-dependencies"],
            "dev_oracles": project["dependency-groups"]["dev"],
        },
    }


def markdown(value: dict) -> str:
    estimates = value["estimators"]
    replay = value["streaming"]
    prediction = value["postestimation"]["saved_prediction"]
    versions = value["versions"]
    missing_data = value["missing_data"]
    lines = [
        "# Current OpenEconometrics capability inventory",
        "",
        "Generated by `scripts/generate_capability_docs.py`; check with `--check`. "
        "The JSON companion retains complete options, conditions and resource contracts.",
        "",
        f"Source reference: `{value['source_ref']}`. Content fingerprints in "
        "[the generated inventory](econometrics/capabilities.generated.json) identify the exact metadata used. "
        "The reference is a source snapshot, not an installation or public-release claim.",
        "",
        f"SDK `{versions['sdk']}` · native source `{versions['native']}` · charts `{versions['charts']}`.",
        "",
        f"{len(estimates)} registered estimator names; {len(replay['estimators'])} Dataset fitting routes; "
        f"{prediction['estimator_count']} common saved predict/margins adapters; "
        f"{prediction['remaining_estimator_count']} names without a common adapter.",
        "",
        "A registered route is implemented within its documented option and target domains. "
        "It does not establish unrestricted support, large-data performance on every model, "
        "or numerical parity. Shared prediction is distinct from family-specific forecasts and prediction helpers.",
        "",
        "## Status and evidence",
        "",
        "- **Implemented** means a registry or capability route exists; inspect its conditions below and in the JSON.",
        "- **Partial** means the documented supported domain has explicit exclusions; it is not blanket method coverage.",
        "- **Planned** methods in historical plans are not counted as implemented unless present in the registry.",
        "- **Parity tested** requires an identified external result and sample/options/inference comparison. "
        "The global `stata_parity_validated` flag remains `false`; selected independent or published-fixture tests do not change it.",
        "- **Independent scientific validation** requires an identified reference, sample, options, covariance and inference comparison.",
        "- **Installed-package verification** requires execution from a package pinned by source and artifact hashes; source tests alone do not qualify.",
        "- **Public shipment** requires an accessible source/release artifact matching that package pin; current source versions do not update older installers.",
        "",
        "The JSON `evidence_scope.per_estimator` separates these proof requirements for each fit name and links "
        "its exact option contract. It asserts source routes only, rather than inferring evidence coverage from "
        "registration. Family guides and dated receipts may establish narrower method/option claims. "
        "Helpers, diagnostics and forecasts in the sections below are separate from registered fitting and common prediction.",
        "",
        "Selected validation modules: `tests/test_prediction_capabilities.py`, "
        "`tests/test_streaming_capability_conditions.py`, `tests/test_econ_core.py`, "
        "`tests/test_runtime_without_scipy.py`, `tests/test_capability_docs.py`. "
        "This inventory is not a test-run receipt; dated evidence records retain their original bounded scope.",
        "",
        "## Source helper registrations",
        "",
        f"{len(value['auxiliary_exports'])} manifest helper registrations, separate from registered fit names "
        "and common predict/margins adapters. The JSON `auxiliary_exports` records each source family/entry. "
        "Presence establishes registration only; read the helper's method/option guide and dated evidence "
        "before making scientific, installed-package or public-release claims.",
        "",
        "## Runtime, devices and dependencies",
        "",
        "Local source allocation guards and preparation contracts are published in the generated JSON "
        "and [Dataset preparation guide](dataset-preparation.md). Parser RSS, per-stage buffers, "
        "SQLite disk/cache and caller-function allocations have distinct scopes; pandas ordering, "
        "typed keys, missing tags and reshape policies are explicit.",
        "",
        f"Execution contract: {value['execution']}. Precision: `{value['precision']}`.",
        "",
        "Streaming device records: `" + json.dumps(replay["devices"], sort_keys=True) + "`.",
        "",
        replay["metal_precision"]
        + ". CUDA hardware validation remains "
        + (
            "recorded."
            if value["cuda_hardware_validated"]
            else "unverified in this source snapshot."
        ),
        "",
        "OpenEconometrics implements its estimators and separation certificates with native Torch kernels. "
        "SciPy, statsmodels and other reference estimators are development oracles, not required estimation engines. "
        "The frozen desktop packaging excludes SciPy. pandas, PyTorch and import/export dependencies retain "
        "NumPy dependencies; the environment is not NumPy-free. PyTorch may also bring NetworkX transitively; "
        "OpenEconometrics network algorithms do not use it as their analytical engine.",
        "",
        "Direct runtime declarations: `" + "`, `".join(value["dependencies"]["runtime"]) + "`.",
        "",
        "Development declarations: `" + "`, `".join(value["dependencies"]["dev_oracles"]) + "`.",
        "",
        "## Multinomial probit choice",
        "",
        "Four bounded CPU float64 helpers fit normalized Gaussian choice likelihoods over two or three explicitly declared alternatives, "
        "restore their complete numerical state, and compute availability-aware probability and own/cross attribute targets. "
        "Three-alternative models jointly estimate physical utility coefficients and two normalized difference-covariance parameters, "
        "or condition on a fixed declared covariance. Full observed information, whole-case HC0 and respondent-cluster CR0 preserve all cross blocks. "
        "Saved queries retain full joint delta covariance, positive-attribute elasticities and fixed-weight averages. "
        "The [method guide](econometrics/mprobit.md) defines numerical integration, identification, admitted input and work budgets, "
        "and complete optimizer-free replay. Four-plus alternatives, fit weights, correlated panel likelihood, GPU/Dataset execution "
        "and licensed vendor parity remain unsupported; source metadata is separate from scientific or installed acceptance.",
        "",
        "## Nested choice",
        "",
        "`nlogit` jointly estimates shared attribute coefficients and finite interior dissimilarities for two-level disjoint nests. "
        "OIM, whole-case HC0 and respondent-cluster CR0 retain full beta/dissimilarity cross covariance. "
        "`nlogit_restore`, `nlogit_predict` and `nlogit_margins` replay complete state and provide new-set probabilities, "
        "analytical substitution, elasticities and fixed weighted averages with joint delta inference. "
        "The [method guide](econometrics/nested-choice.md) defines normalization, boundaries, support and resource limits; "
        "broader choice and licensed-reference acceptance remain separate.",
        "",
        "## Rank-ordered choice",
        "",
        "`rologit` fits strict full/partial rankings with explicit availability and unbalanced choice sets. "
        "OIM, complete choice-case HC0 and respondent-cluster CR0 have distinct inference contracts. "
        "`rologit_restore`, `rologit_predict` and `rologit_margins` validate portable saved state, "
        "predict new choice sets and calculate analytical substitution with full parameter uncertainty. "
        "[Method guide](econometrics/rank-ordered-choice.md) describes rank coding, query limits, "
        "fixed case-weighted targets and unsupported ties/fit weights/nested ranking.",
        "",
        "## Unidimensional IRT",
        "",
        "`irt_rasch`, `irt_2pl`, known-guessing `irt_3pl`, `irt_grm`, `irt_pcm` and `irt_rsm` fit "
        "resident standard-normal marginal likelihoods with full observed information. Rasch/PCM/RSM "
        "fix discrimination at one. `irt_information` returns conditional category/item/test curves; "
        "`irt_score` returns EAP and posterior SD. `irt_restore` checks complete portable state. "
        "[Method guide](econometrics/irt.md) states identification, sample and work limits. "
        "Saved descriptive residual/Q3 diagnostics and observed-score binary MH DIF have separate "
        "[bounded contracts](econometrics/next-eight-survey-measurement.md). Free guessing, latent/multigroup "
        "or polytomous/nonuniform DIF, calibrated fit tests, robust inference and vendor parity remain excluded.",
        "",
        "## Saved analytical extensions",
        "",
        "Binary logit/probit mediators paired with Gaussian/logit/probit/Poisson outcomes have eight "
        "separately validated model domains. Exact two-point standardization over retained controls "
        "includes exposure–mediator interaction and full joint OIM/HC0 delta covariance. "
        "[Binary mediation guide](econometrics/binary-mediation.md) states fixed-population targets, "
        "causal labeling assumptions, saved replay and explicit unsupported domains.",
        "",
        "Literal penalty factors and forced controls extend continuous regularized prediction. "
        "Saved Gaussian local derivatives retain fixed training bandwidths without uncertainty claims. "
        "Single-shock proxy SVAR requires exact external-key/source alignment and returns point responses only. "
        "Saved mixed scalar Satterthwaite contrasts and full-covariance event/horizon normal-limit bands "
        "declare different inference laws. Fixed-design iid Gaussian OLS CUSUM/CUSUMSQ uses complete "
        "conditional projection simulation rather than borrowed recursive or asymptotic bounds. "
        "[Inference guide](econometrics/next-eight-inference-stability.md), "
        "[proxy guide](econometrics/proxy-svar.md), "
        "[regularization guide](econometrics/next-eight-regularized-prediction.md) and "
        "[smoothing/multivariate matrix](econometrics/next-eight-smoothing-multivariate.md) state domains, "
        "workspace plans and substantive open stages. None establishes blanket vendor parity.",
        "",
        "## Multivariate option extensions",
        "",
        "Frequency-weight PCA/EFA, summary covariance/correlation input, minres/alpha/SAS image-covariance extraction, "
        "Anderson–Rubin scoring, complete and identified partial targets, Crawford–Ferguson and geomin rotations, "
        "and saved supplementary CA projections have bounded contracts. Fixed one-factor IID principal-factor "
        "bootstrap supplies marginal percentile intervals with complete replicate covariance; rotated, selected-factor "
        "or dependent-data inference and SPSS generalized-image parity are excluded. "
        "Saved scalar repeated-measure cell x between-design contrasts use exact Student t under "
        "independent Gaussian subjects/common cell covariance, without requiring sphericity. "
        "The JSON records weights, training-moment requirements, device/workspace/work limits and exclusions. "
        "See [the multivariate guide](econometrics/multivariate.md); this is not universal FACTOR/PCA/CA parity.",
        "Frequency reliability/KMO, resident frequency LDA/QDA and declared group summaries, frequency/joint-summary CCA "
        "with saved scores, and a wide complete-subject RM adapter with validated saved contrasts extend the option matrix. "
        "Gaussian frequency tests require original independent observations; RM contrasts retain full covariance with "
        "marginal t and joint Hotelling F degrees of freedom. Tied canonical axes, missing policies and unsupported designs "
        "are explicit in [the weighted/matrix guide](econometrics/multivariate-weight-matrix.md).",
        "Forward/backward/bidirectional Wilks LDA screening supports a fixed complete-case sample and integer frequencies; "
        "its reference p-values describe selection rules, not post-selection inference. One-way frequency and group-summary "
        "MANOVA save full moments. General saved MANOVA and repeated-measures L B M = C hypotheses retain complete "
        "target covariance, marginal t intervals and explicit exact/approximate/Roy-upper-bound test labels. "
        "See [selection and general contrasts](econometrics/multivariate-selection-contrasts.md).",
        "",
        "## Repeated residual Gaussian GLS",
        "",
        "`repeated_gls` fits negative-capable compound symmetry, signed integer-gap AR1, occasion-specific diagonal "
        "and full unstructured residual covariance, each by ML or REML. Independent subjects may have absent occasions; "
        "all selected outcomes remain complete and every declared residual block is retained. ML uses full joint observed "
        "information; REML reports plug-in fixed GLS and separate covariance-parameter information. Saved replay, joint "
        "population means and named fixed contrasts retain complete covariance. The [guide](econometrics/repeated-residual-gls.md) "
        "states CPU float64, work/geometry limits and asymptotic inference. Random-effect combinations, KR/Satterthwaite, "
        "weights and Dataset routes remain excluded; these are separate helpers rather than additional ModelSpec fit names.",
        "",
        "## Independent-study meta-analysis",
        "",
        "`meta_effectsize` prepares MD, Hedges g/LS, log OR/RR and Fisher z. `meta_pool` and `meta_regress` "
        "supply common/DL/ML/REML independent-study models with z/HKSJ/modified-HKSJ inference. "
        "`meta_predict` restores complete coefficient covariance for conditional means and approximate latent-effect intervals. "
        "`meta_diagnostics` retains all planned leave-one-out/cumulative/subgroup refits and descriptive Egger/forest/funnel output. "
        "These summary procedures are separately catalogued helpers, not additional ModelSpec estimator routes. "
        "The [method guide](meta-analysis.md) states assumptions, numerical limits and bounded validation; "
        "The separate [dependent-effects guide](econometrics/dependent-meta.md) documents labeled full sampling "
        "covariance GLS, one effect-level or shared-study ML/REML variance, study CR0/CR1, saved joint contrasts "
        "and mean/latent-effect covariance, and whole-study deletion diagnostics. Multiple variance components, "
        "unstructured random covariance, selection/bias models and licensed vendor comparisons remain open.",
        "",
        "## Mixed-data TwoStep clustering",
        "",
        "[TwoStep clustering](econometrics/twostep.md) adds a bounded native CPU float64 CF tree, complete "
        "likelihood merge hierarchy, global-minimum BIC/AIC or fixed counts, descriptive profiles/silhouette, "
        "saved typed-map assignment and physical-row ARI stability. Small-leaf noise and order dependence "
        "are explicit. IBM automatic change/jump selection, adaptive rebuild/noise reinsertion, licensed "
        "vendor parity, weights and GPU/Dataset execution are outside this scope.",
        "",
        "## Categorical scaling and loglinear workflows",
        "",
        "Eight bounded CPU float64 [categorical procedures](econometrics/categorical.md) add nominal/ordinal optimal-scaling regression, "
        "categorical PCA, MCA, multiset nonlinear homogeneity analysis, nonmetric MDS, structural-zero hierarchical IPF "
        "and general Poisson cell-design ML. Adaptive transformations are descriptive local solutions; loglinear fits "
        "retain full observed-information inference under independent Poisson cells. Full maps, sample, convergence "
        "and cell designs persist through complete summary state; saved projections and nested LR verify their domains.",
        "Additional [categorical options](econometrics/categorical-outcomes.md) add nominal/ordinal response scaling and "
        "full-refit IID fixed-query prediction bootstrap; [saved rotations](econometrics/categorical-rotation.md) add "
        "varimax and oblique promax with coherent pattern/structure/component covariance and supplementary scores. "
        "[Conditional sampling and interaction selection](econometrics/categorical-sampling-selection.md) add direct "
        "fixed-total multinomial/product-multinomial joint ML and greedy hierarchical AIC/BIC with every candidate retained. "
        "Bootstrap intervals are empirical, rotation is descriptive and selected-model inference is not selection-adjusted.",
        "",
        "## Measurement reliability",
        "",
        "Eight resident CPU float64 [measurement procedures](econometrics/measurement-reliability.md) "
        "cover two-step polychoric/polyserial correlations, one-factor model omega-total, balanced ICC, "
        "Cohen/Fleiss agreement, Krippendorff alpha and Gwet AC1/AC2. Optional complete subject jackknife "
        "refits nuisance parameters and records covariance and approximate t inference. Complete JSON "
        "artifacts retain tables, category order, sample positions and diagnostics. IRT, joint ordinal ML, "
        "exact/vendor intervals, weights and GPU/Dataset inputs are outside this bounded scope.",
        "",
        "## Conditional exact regression",
        "",
        "Eight [conditional-regression procedures](econometrics/conditional-regression.md) provide one-target "
        "Bernoulli/Poisson CMLE, inclusive exact intervals, probability-ordered tests and complete conditional "
        "training response moments. Intercept and up to three integer nuisance statistics are conditioned out. "
        "Full bounded enumeration and exposure/factorial base measures persist; no joint vector or nuisance "
        "estimation, Wald/mean uncertainty, mid-p/MUE, new-row prediction, weights or licensed vendor parity.",
        "",
        "## Bounded finite regression",
        "",
        "Eight resident CPU float64 [finite-regression procedures](econometrics/finite-regression.md) cover full-model Firth logistic, "
        "saved predictions, same-model penalized profile intervals and LR tests, complete fixed-margin common odds, "
        "fixed-exposure exact Poisson rate and exhaustive small-sample LTS with descriptive saved predictions. "
        "Complete ordered artifacts retain full covariance, samples, all conditional support/candidates and convergence traces. "
        "Firth Wald/profile inference is asymptotic; exact APIs have explicit tail and boundary conventions. "
        "Multivariable exact models, weighted/categorical/streaming routes, large-n FAST-LTS and vendor parity remain open.",
        "",
        "## Continuous-first-stage control functions",
        "",
        "Eight conditional-mean domains use one continuous endogenous regressor and an OLS first stage. "
        "Full nonsymmetric observed stacked equations include generated-control uncertainty in HC0 or whole-cluster CR0, "
        "with correction factor one and asymptotic normal/Wald inference. All eight fit domains support bounded Dataset replay. "
        "Resident v1 states retain bounded source rows; compact stream.v2 states validate unchanged estimation sources, "
        "sample and joint algebra without refitting. `cf_restore` explicitly rebinds portable JSON states; "
        "`cf_predict` computes observed-D conditional means with full joint covariance. "
        "Gamma/inverse-Gaussian use fixed working dispersion one; fractional logit is quasi-Bernoulli. "
        "Instrument exclusion and nonlinear control sufficiency remain caller assumptions. No weak-IV, finite-cluster, "
        "structural treatment-effect, estimated-dispersion ML or blanket vendor parity claim. "
        "See [the method scope](econometrics/control-functions.md).",
        "",
        "## Missing data and multiple imputation",
        "",
        "Eight implemented method stages cover Gaussian observed-data EM (`mvnorm_em`), the common-covariance "
        "Little diagnostic (`little_mcar`), explicit proper-NIW MVN data augmentation (`mi_mvn`), verified "
        "monotone Gaussian regression (`mi_monotone`), and FCS normal, PMM type 1 and binary proper-prior "
        "logistic Metropolis updates (`mi_chained`). `mi_pool` retains Rubin full covariance and marginal "
        "Barnard-Rubin inference; `mi_test` adds D1 linear joint tests with Li1991 or positive-moment "
        "Reiter2007 calibration. The separate `missing_patterns` helper describes observed missingness only. "
        "These helpers do not add ModelSpec estimator routes.",
        "",
        "Eight additional stages add proper-prior Poisson, cumulative ordered-logit and nominal multinomial-logit "
        "FCS (`mi_discrete`); fixed single-target Gaussian mean, binary log-odds and Poisson log-mean "
        "pattern-mixture sensitivity (`mi_delta`); declared deterministic affine/product/power derivation "
        "after MI (`mi_passive`); and full-covariance pooled linear functionals with contrast-specific "
        "Rubin/Barnard-Rubin inference against explicit nulls (`mi_lincom`). Sensitivity parameters remain "
        "external assumptions. Passive transforms do not feed back into FCS or establish substantive compatibility. "
        "Typed extended result classes validate the complete saved operation-specific state.",
        "",
        f"Resident CPU float64 generation is bounded by {missing_data['budgets']['generation_rows']:,} rows, "
        f"{missing_data['budgets']['generation_columns']} columns and "
        f"{missing_data['budgets']['generation_imputations'][1]} imputations; pooling admits "
        f"{missing_data['budgets']['pooling_imputations'][0]}..{missing_data['budgets']['pooling_imputations'][1]} "
        f"imputations and at most {missing_data['budgets']['pooling_terms']} terms. Diagnostics have "
        f"{missing_data['budgets']['diagnostic_columns']} columns, "
        f"{missing_data['budgets']['diagnostic_patterns']:,} patterns and declared work/workspace limits, "
        "rather than a fixed row ceiling. Generation preserves the original sample/index and observed cells; "
        "diagnostics record informative and all-missing row positions. Immutable full recorded JSON state "
        "has checksums for corruption detection, including completions, pooling covariance and joint-test parents.",
        "",
        "EM checks likelihood and parameter convergence; finite MVN/FCS schedules establish no sampler convergence "
        "or independent posterior draws. Little non-rejection does not prove MCAR. Weights, Dataset/replay, "
        "CUDA/Metal, arbitrary passive expressions/FCS feedback, general multivariable MNAR, pooled likelihood-ratio tests "
        "and licensed vendor parity remain excluded. See the "
        "[Gaussian generation guide](econometrics/mi-generation.md), "
        "[FCS guide](econometrics/mi-chained.md) and "
        "[pooling guide](econometrics/mi-pooling.md); the JSON states priors, defaults and exact boundaries.",
        "",
        "## Full-profile conjoint",
        "",
        "Eight resident CPU float64 [conjoint procedures](econometrics/conjoint.md) generate complete declared profiles "
        "and regular binary fractions, audit rank/orthogonality/aliases, fit individual scored utilities with full covariance, "
        "predict conditional fitted means, validate held-out profiles, average individual-normalized importance and simulate "
        "first-choice/positive BTL/temperature-logit shares. Complete JSON preserves coding, samples and every table. "
        "Rank/sequence/CBC, arbitrary mixed-level array search, group/market sampling intervals, weights and GPU/Dataset are outside scope.",
        "",
        "## Prospective power designs",
        "",
        "Eight bounded planning helpers cover normal t means/pairs, pooled independent t means, balanced Gaussian ANOVA, "
        "fixed-design regression F, equal-cluster ICC normal design effects and Pearson goodness-of-fit/independence approximations. "
        "They solve power, verified integer sample size and detectable departure using prespecified population assumptions, "
        "retain full settings/scenarios, and consume no observations. Exact conditional t/F laws and approximate design laws "
        "are labelled separately. [Method guide](econometrics/power-designs.md) records budgets and exclusions. "
        "These eight helpers exclude survival/sequential and broader options. "
        "Separate known-SD paired/independent, Fisher/arcsine, Schoenfeld log-rank, conditional McNemar "
        "and random-width assurance methods are documented in [prospective planning](econometrics/planning.md).",
        "",
        "## Declared mixed-frequency, ETS and state-space workflows",
        "",
        "Release-aware exponential-Almon and unrestricted MIDAS preserve validated origin/lag calendars. "
        "Bounded ARIMA and additive ETS selection record same-sample likelihood criteria, free-parameter counts "
        "and candidate failures; declared rolling/recursive workflows refit inside each origin. "
        "General proper-prior Gaussian state-space models retain observation masks, known equation/exogenous paths "
        "and full filtered/forecast covariance; saved RTS states, lag-one and disturbance moments condition on "
        "the observations. Forecast uncertainty conditions on the selected fitted model. Exact diffuse, "
        "singular innovation geometry, estimated equation schedules, dynamic factors, DSGE and external "
        "seasonal engines remain separate stages. "
        "[Method contracts](econometrics/audited-eight-timeseries-2026-10-07.md) "
        "record public routes, independent checks and replay boundaries; vendor parity remains false.",
        "",
        "## Causal balance, sensitivity and treatment targets",
        "",
        "Implemented procedures: `" + "`, `".join(value["causal_design"]["procedures"]) + "`.",
        "",
        "These resident CPU float64 procedures provide entropy/CEM preprocessing, fixed-weight observed balance, "
        "paired sign/signed-rank and homoskedastic OLS partial-R2 sensitivity, RR E-values, fixed external bias-box "
        "Gaussian unions, consistency-only Manski/monotone-selection Lee identification bounds, "
        "declared paired/complete/stratified sharp-null randomization, and declared independent-arm "
        "marginal CDF, quantile and common-horizon KM restricted-mean survival targets. "
        "Observational extensions add estimated-propensity Hájek CDF inference, whole-row refitted quantile bootstrap, "
        "cross-fitted AIPW CDFs, known-nuisance HT survival/exponential-censor RMST and complete-horizon RMST augmentation, "
        "plus formal observed-control OVB bounds and max-coordinate robustness witnesses. "
        "`causal_design_save` and `causal_design_load` preserve all tables, dtypes, sample positions and scientific state "
        "with checksums. Assignment and predeclaration assumptions remain explicit user declarations; "
        "preprocessing has no effect covariance, and fixed-weight diagnostics do not include estimated-weight uncertainty. "
        "[Method contracts](econometrics/causal-design.md) document numerical support, pointwise inference and budgets; "
        "[Observational target contracts](econometrics/causal-observational-targets.md) distinguish nuisance estimation, "
        "both-consistent AIPW inference, fixed external survival laws and unsupported designs. "
        "[Sensitivity and assignment contracts](econometrics/causal-sensitivity.md) distinguish fixed assumptions, "
        "identification bounds and sampling uncertainty. "
        "Dataset/GPU execution, automatic causal identification from balance and vendor parity are excluded.",
        "",
        "## Temporal disaggregation",
        "",
        "Eight bounded native CPU float64 procedures provide original first/second-difference additive/proportional Denton, "
        "first-difference Denton-Cholette, fixed-rho Chow-Lin, Fernandez and fixed-rho Litterman. "
        "Explicit complete annual-to-quarter/month and quarter-to-month calendars preserve sum/mean/first/last constraints; "
        "complete input, release, aggregation and covariance state is saved. Denton is descriptive; GLS uncertainty "
        "conditions on its prespecified Gaussian covariance shape and exact low-frequency observations. "
        "[Method contracts](econometrics/temporal-disaggregation.md) specify boundaries; estimated rho, ragged calendars, "
        "vintage forecast evaluation and licensed vendor parity remain open.",
        "",
        "## Interval survival and competing risks",
        "",
        "Eight bounded native CPU float64 stages provide intercept-only exponential, Weibull, lognormal and loglogistic "
        "interval-censored ML, Turnbull support-location bounds, Aalen-Johansen cumulative incidence, cause-specific "
        "Nelson-Aalen cumulative hazard, and independent two-group prespecified fixed-time CIF contrasts. "
        "Complete input, parameter and curve covariance state is saved; Turnbull reports identification bounds without invented sampling uncertainty. "
        "[Method contracts](econometrics/survival-extensions.md) specify endpoint, risk-set and inference assumptions. "
        "Covariate interval regression, Fine-Gray, Gray tests, recurrent events, frailty and licensed vendor parity remain open.",
        "",
        "## Declared hypothesis families",
        "",
        "`multipletests` implements eight FWER/FDR adjustments with explicit dependence assumptions. "
        "`stepdown` consumes scientifically justified joint-null draws for Romano-Wolf maxT or Westfall-Young minP. "
        "`simultaneous_ci` uses a compatible full covariance and normal-limit simulation with recorded Monte Carlo precision. "
        "These CPU float64 summary procedures retain order, labels and missing policies; they do not invent a family, "
        "fit a model, generate an arbitrary bootstrap, or accept Dataset/GPU inputs. "
        "[Method contracts](econometrics/multiple-testing.md) specify assumptions, persistence and preallocation budgets.",
        "",
        "Eight separately declared [finite-sample distribution-free procedures](econometrics/distribution-free-joint.md) "
        "generate row-sign/two-group assignment stepdown nulls and conservative entire-CDF, lower-quantile, "
        "binomial, multinomial-simplex and bounded-mean confidence families. Their public postest manifest and "
        "offline editor signatures preserve distinct iid/invariance/support contracts; they do not infer them from data. "
        "Full summary state, including orbit statistics and unbounded endpoint flags, is saved independently of previews.",
        "",
        "## Survey declaration",
        "",
        "`survey_design` validates a bounded resident single-stage sampling declaration and persists its geometry. "
        "It supports positive sampling weights, typed nested PSU/strata IDs, integer population-count FPC and "
        "explicit census singleton admission. It produces no survey estimates, covariance or inference. "
        "[Declaration contract](econometrics/survey-design.md) and "
        "[open implementation stages](econometrics/roadmaps/survey.md) describe its limits.",
        "",
        "## Survey inference",
        "",
        "Eight resident CPU float64 survey procedures provide means, totals, ratios, fixed-category proportions "
        "with full Taylor covariance, balanced BRR/Fay, stratified PSU jackknife and supplied justified design-bootstrap weights. "
        "Complete design geometry survives domain/missing exclusions; typed saved results retain full covariance and replica replay. "
        "[Method contracts](econometrics/survey-inference.md) specify centering, df, FPC and budgets. "
        "Numeric linear/logit/probit/Poisson coefficient refits additionally support BRR/Fay, stratified PSU jackknife and supplied design-bootstrap "
        "with complete covariance and method-aware saved replay. [Regression replication contracts](econometrics/survey-regression-replication.md) "
        "specify the sixteen family/method gates and conditional postestimation. "
        "[Empirical target replication](econometrics/survey-replicate-margins.md) adds joint partial-profile means "
        "and continuous AMEs for logit/probit/Poisson, refitting and reweighting each replica. General multistage, arbitrary-weight DEFF, "
        "calibrated unit-varying replicates and licensed vendor parity remain separate gates.",
        "",
        "Nine separate [two-stage SRSWOR APIs](econometrics/survey-two-stage.md) declare exact nested PSU/SSU geometry "
        "and estimate means, totals, ratios, proportions, WLS, logit, probit and Poisson with both stage FPC terms. "
        "Weights are derived from mandatory stage population counts. Complete geometry survives domain/missing exclusions; "
        "distinct typed saved states replay primitive targets/scores and both covariance contributions. First-stage PSU-minus-strata "
        "df is an explicit reference convention; positive variance with df0 has no tests or CI. Three or more stages, PPS and "
        "calibration remain outside this bounded contract.",
        "",
        "Nine separate [three-stage SRSWOR APIs with strata at both lower stages](econometrics/survey-fully-stratified-three-stage.md) "
        "declare positive SSU-cell frames within sampled PSUs and positive TSU-cell frames within sampled SSUs. "
        "Exact cell population counts derive weights; means, totals, ratios, proportions, WLS, logit, probit and Poisson "
        "retain three covariance contributions with independently centered terminal strata and each parent cell's sampling fraction. "
        "Complete physical geometry, both typed frames and primitive targets/scores survive domain/missing exclusions and saved replay. "
        "First-stage PSU-minus-strata df is a reference convention; positive variance with df0 has SE but no tests or CI. "
        "Four or more selections, PPS, arbitrary supplied weights, calibration, Dataset/GPU and licensed vendor parity remain outside this scope.",
        "",
        "## Estimator support matrix",
        "",
        "| Estimator | Family | Dataset route | Common saved prediction | Covariance | Weights | Options |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, item in sorted(estimates.items()):
        weights = item.get("weights", [])
        if isinstance(weights, bool):
            weights = ["see core contract"] if weights else []
        fields = [
            f"`{name}`",
            item.get("family", "core"),
            "implemented" if name in replay["estimators"] else "unavailable",
            "conditional adapter"
            if name in prediction["estimators"]
            else "family-specific or planned",
            ", ".join(item["covariances"]),
            ", ".join(weights) or "none",
            ", ".join(sorted(item.get("options", {}))) or "see core contract",
        ]
        lines.append("| " + " | ".join(field.replace("|", "\\|") for field in fields) + " |")
    lines += [
        "",
        "## Dataset fitting conditions",
        "",
        f"Shared bounds: {replay['max_parameters']} parameters; no total row ceiling; "
        f"{replay['working_memory_budget_bytes']} bytes of planned working buffers. "
        "These are not total process RSS guarantees. Reader, disk, model width and repeated I/O constraints remain.",
        "",
    ]
    for name, algorithm in sorted(replay["algorithms"].items()):
        lines += [
            f"### `{name}`",
            "",
            algorithm + ".",
            "",
            "Conditions: `"
            + json.dumps(replay["conditions"].get(name, {}), sort_keys=True, ensure_ascii=False)
            + "`.",
            "",
        ]
    lines += [
        "## Common saved prediction",
        "",
        prediction["coverage_scope"] + ".",
        "",
        "Supported names: `" + "`, `".join(prediction["estimators"]) + "`.",
        "",
        "Remaining common adapters: `" + "`, `".join(prediction["remaining_estimators"]) + "`.",
        "",
        "Dataset evaluation requires explicit input and writes complete indexed output on disk. "
        "The stored 400-row fit/chart preview is separate from evaluation. Options such as panel FD, "
        "group effects, count cutoffs and multi-equation outcomes retain their explicit restrictions.",
        "",
        "See [prediction domains](econometrics/prediction-domains.md), "
        "[family coverage](econometrics/coverage.md) and [streaming contracts](streaming.md).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-ref")
    args = parser.parse_args()
    if args.check:
        if not JSON_PATH.exists() or not MD_PATH.exists():
            print("Capability documents missing. Run scripts/generate_capability_docs.py.")
            return 1
        reference = json.loads(JSON_PATH.read_text())["source_ref"]
    else:
        reference = (
            args.source_ref
            or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        )
    value = snapshot(reference)
    outputs = {
        JSON_PATH: json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        MD_PATH: markdown(value),
    }
    for name in SCOPE_DOCUMENTS:
        path = ROOT / name
        try:
            outputs[path] = scope_document(path.read_text(), value, name)
        except (OSError, ValueError) as error:
            print(f"Capability documentation drift: {error}")
            return 1
        claims = unsupported_scope_claims(outputs[path])
        if claims:
            print(f"Unsupported universal scope claim in {name}: {claims}")
            return 1
    if args.check:
        changed = [
            (str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
            for path, text in outputs.items()
            if path.read_text() != text
        ]
        if changed:
            print("Capability documentation drift: " + ", ".join(changed))
            return 1
        print("Capability documentation matches source.")
        return 0
    for path, text in outputs.items():
        path.write_text(text)
    print("Generated source-backed capability documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
