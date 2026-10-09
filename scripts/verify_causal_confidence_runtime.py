"""Owned frozen verification of eight confidence, multiarm and individual-effect extensions."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import verify_causal_targets_runtime as engine

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/causal_confidence_eight.py"
MODULES = tuple(dict.fromkeys(engine.MODULES + (
    "openecon.econometrics.causal_design.assignment_extensions",
    "openecon.econometrics.causal_design.neyman",
    "openecon.econometrics.causal_design.identification_extensions",
    "openecon.econometrics.causal_design.confidence_sets",
    "openecon.econometrics.causal_design.multiarm_neyman",
    "openecon.econometrics.causal_design.effect_distribution",
)))
NAMES = ('randomization_confidence_set', 'paired_randomization_confidence_set', 'cluster_randomization_confidence_set', 'bernoulli_neyman_ate', 'multiarm_neyman_ate', 'stratified_multiarm_neyman_ate', 'treatment_effect_cdf_bounds', 'treatment_effect_quantile_bounds')
MARKER = "CAUSAL_CONFIDENCE_EIGHT_OK "

@contextmanager
def configured_engine():
    """Keep earlier-batch verifier imports unchanged outside this invocation."""
    old = engine.EXAMPLE, engine.MODULES, engine.NAMES, engine.MARKER
    engine.EXAMPLE, engine.MODULES, engine.NAMES, engine.MARKER = EXAMPLE, MODULES, NAMES, MARKER
    try:
        yield
    finally:
        engine.EXAMPLE, engine.MODULES, engine.NAMES, engine.MARKER = old


def scoped(function):
    @wraps(function)
    def call(*args, **kwargs):
        with configured_engine():
            return function(*args, **kwargs)
    return call


digest = engine.digest
fingerprint = engine.fingerprint
canonical = engine.canonical
header = scoped(engine.header)
proof_from_execution = scoped(engine.proof_from_execution)
source_identity = scoped(engine.source_identity)
artifact_files = scoped(engine.artifact_files)
verify = scoped(engine.verify)

if __name__ == "__main__":
    with configured_engine():
        engine.main()
