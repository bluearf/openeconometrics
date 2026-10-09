"""Compare complete source, wheel, frozen and native assignment artifacts."""
import verify_causal_targets_artifacts as engine
from verify_causal_assignment_runtime import NAMES

if __name__ == "__main__":
    engine.NAMES = NAMES
    raise SystemExit(engine.main())
