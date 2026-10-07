"""Optional application dependencies, checked only at their entry points."""
from importlib.util import find_spec


def require_extra(extra: str, *modules: str) -> None:
    missing = [name for name in modules if find_spec(name) is None]
    if missing:
        raise ImportError(
            f"This operation needs the OpenEconometrics '{extra}' extra "
            f"(missing: {', '.join(missing)}). Install with pip install 'openecon[{extra}]'."
        )
