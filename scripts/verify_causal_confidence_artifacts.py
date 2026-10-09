"""Compare complete source, wheel, frozen and native confidence/multiarm/distribution artifacts."""
import verify_causal_targets_artifacts as engine
from verify_causal_confidence_runtime import NAMES

if __name__ == "__main__":
    engine.NAMES = NAMES
    raise SystemExit(engine.main())
