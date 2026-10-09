"""Owned frozen verification of eight assignment and identification extensions."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import verify_causal_targets_runtime as engine

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/causal_assignment_eight.py"
MODULES = tuple(dict.fromkeys(engine.MODULES + (
    "openecon.econometrics.causal_design.assignment_extensions",
    "openecon.econometrics.causal_design.neyman",
    "openecon.econometrics.causal_design.identification_extensions",
)))
NAMES = ('cluster_randomization', 'bernoulli_randomization', 'neyman_ate', 'stratified_neyman_ate', 'cluster_neyman_ate', 'paired_neyman_ate', 'manski_ate_inference', 'stratified_lee_bounds')
MARKER = "CAUSAL_ASSIGNMENT_EIGHT_OK "

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
