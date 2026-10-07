"""Parameter layout of one ``arch`` specification (no tensors).

The reported parameter vector is ordered as Stata prints it:

    mean equation   x_t b                      terms of the design ("Intercept" first)
    ARCHM           psi                        ARCHM:sigma2 | ARCHM:sigma | ARCHM:lnsigma2
    ARMA            rho_j, theta_k             ARMA:L<j>.ar, ARMA:L<k>.ma
    HET             l (variance regressors)    HET:<name>, ..., HET:Intercept
    ARCH            a_i, g_i, b_j, omega       ARCH:L<i>.arch ... ARCH:Intercept
    POWER           phi                        POWER:power            (power ARCH only)
    distribution    ln(df - 2) | ln(shape)     /lndfm2 | /lnshape

With variance regressors the variance constant is ``HET:Intercept`` and there is no
``ARCH:Intercept``, as in Stata. The names of the ARCH terms depend on the model:

    arch, garch, igarch   L.arch            L.garch
    gjr / tarch           L.arch  L.tarch   L.garch
    egarch                L.earch L.earch_a L.egarch
    parch                 L.parch           L.pgarch
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from openecon.analysis_contracts import AnalysisError

MAX_LAG = 60             # largest admissible lag of any ARCH, GARCH, AR or MA term
_KINDS = {"arch": "garch", "garch": "garch", "igarch": "garch", "gjr": "gjr", "tarch": "gjr",
          "egarch": "egarch", "parch": "parch"}
_TERM_NAMES = {            # (ARCH term, asymmetry term, GARCH term)
    "garch": ("arch", None, "garch"),
    "gjr": ("arch", "tarch", "garch"),
    "egarch": ("earch", "earch_a", "egarch"),
    "parch": ("parch", None, "pgarch"),
}
_ARCHM_TERMS = {"variance": "ARCHM:sigma2", "sd": "ARCHM:sigma", "log": "ARCHM:lnsigma2"}
_DIST_TERMS = {"t": "/lndfm2", "ged": "/lnshape"}
_LABELS = {"arch": "ARCH", "garch": "GARCH", "gjr": "GJR-GARCH", "tarch": "GJR-GARCH",
           "egarch": "EGARCH", "parch": "PARCH", "igarch": "IGARCH"}
_DIST_LABELS = {"normal": "normal", "t": "Student t", "ged": "GED"}


@dataclass(frozen=True)
class Index:
    """Positions of each parameter block in the reported vector."""

    x: list[int]
    m: list[int]         # ARCH-in-mean coefficient (0 or 1 entry)
    ar: list[int]
    ma: list[int]
    z: list[int]         # slopes of the variance regressors
    c: int               # variance constant (omega, or the HET constant)
    a: list[int]         # ARCH terms
    g: list[int]         # asymmetry terms (tarch, earch_a)
    b: list[int]         # GARCH terms
    p: list[int]         # power (0 or 1 entry)
    d: list[int]         # distribution parameter (0 or 1 entry)
    k: int

    @property
    def mean(self) -> list[int]:
        """Parameters of the conditional mean other than the ARCH-in-mean coefficient."""
        return [*self.x, *self.ar, *self.ma]


@dataclass(frozen=True)
class Layout:
    model: str
    dist: str
    arch_lags: tuple[int, ...]
    garch_lags: tuple[int, ...]
    ar_lags: tuple[int, ...] = ()
    ma_lags: tuple[int, ...] = ()
    archm: str | None = None
    k_x: int = 0
    k_z: int = 0

    @property
    def kind(self) -> str:
        """The recursion: garch (arch, garch, igarch), gjr, egarch or parch."""
        return _KINDS[self.model]

    @property
    def restricted(self) -> bool:
        """IGARCH: the last GARCH coefficient is 1 minus the other ARCH and GARCH terms."""
        return self.model == "igarch"

    @property
    def asymmetric(self) -> bool:
        return self.kind in {"gjr", "egarch"}

    @property
    def max_lag(self) -> int:
        return max(1, *self.arch_lags, *self.garch_lags, *self.ar_lags, *self.ma_lags)

    @cached_property
    def index(self) -> Index:
        position = 0

        def take(count: int) -> list[int]:
            nonlocal position
            block = list(range(position, position + count))
            position += count
            return block

        x, m = take(self.k_x), take(int(self.archm is not None))
        ar, ma = take(len(self.ar_lags)), take(len(self.ma_lags))
        z = take(self.k_z)
        c = take(1)[0] if self.k_z else -1
        a = take(len(self.arch_lags))
        g = take(len(self.arch_lags) if self.asymmetric else 0)
        b = take(len(self.garch_lags))
        if c < 0:
            c = take(1)[0]
        p = take(int(self.kind == "parch"))
        d = take(int(self.dist != "normal"))
        return Index(x, m, ar, ma, z, c, a, g, b, p, d, position)

    @property
    def k(self) -> int:
        return self.index.k

    @property
    def k_free(self) -> int:
        return self.k - int(self.restricted)

    def terms(self, outcome: str, x_terms: list[str],
              z_terms: list[str]) -> tuple[list[str], list[str | None]]:
        """Unique term names and their equation labels, in the order of ``index``."""
        ix = self.index
        names: list[str] = [""] * ix.k
        equations: list[str | None] = [None] * ix.k
        arch_name, asymmetry_name, garch_name = _TERM_NAMES[self.kind]

        def put(positions: list[int], labels: list[str], equation: str | None) -> None:
            for position, label in zip(positions, labels, strict=True):
                names[position], equations[position] = label, equation

        put(ix.x, x_terms, outcome)
        put(ix.m, [_ARCHM_TERMS[self.archm]] if self.archm else [], "ARCHM")
        put(ix.ar, [f"ARMA:L{lag}.ar" for lag in self.ar_lags], "ARMA")
        put(ix.ma, [f"ARMA:L{lag}.ma" for lag in self.ma_lags], "ARMA")
        put(ix.z, z_terms, "HET")
        put([ix.c], ["HET:Intercept" if self.k_z else "ARCH:Intercept"],
            "HET" if self.k_z else "ARCH")
        put(ix.a, [f"ARCH:L{lag}.{arch_name}" for lag in self.arch_lags], "ARCH")
        if ix.g:
            put(ix.g, [f"ARCH:L{lag}.{asymmetry_name}" for lag in self.arch_lags], "ARCH")
        put(ix.b, [f"ARCH:L{lag}.{garch_name}" for lag in self.garch_lags], "ARCH")
        put(ix.p, ["POWER:power"] if ix.p else [], "POWER")
        put(ix.d, [_DIST_TERMS[self.dist]] if ix.d else [], None)
        return names, equations

    def label(self) -> str:
        """Short model name, e.g. ``GARCH(1,1)``: (ARCH terms, GARCH terms)."""
        orders = f"({len(self.arch_lags)})" if self.model == "arch" \
            else f"({len(self.arch_lags)},{len(self.garch_lags)})"
        text = _LABELS[self.model] + orders
        if self.archm:
            text += "-in-mean"
        if self.ar_lags or self.ma_lags:
            text += " with ARMA disturbances"
        return text

    def title(self) -> str:
        return f"{self.label()} regression, {_DIST_LABELS[self.dist]} errors"

    def describe(self) -> dict[str, Any]:
        """JSON record of the specification, stored in ``result.extra['model']``."""
        return {"model": self.model, "recursion": self.kind, "label": self.label(),
                "dist": self.dist, "arch_lags": list(self.arch_lags),
                "garch_lags": list(self.garch_lags), "ar_lags": list(self.ar_lags),
                "ma_lags": list(self.ma_lags), "archm": self.archm,
                "mean_terms": self.k_x, "variance_regressors": self.k_z}


def lag_list(value: Any, name: str) -> list[int] | None:
    """``None``, an order ``q`` (lags 1..q) or a list of lags, as a sorted list of lags.

    Used by the convenience function: ``arch=2`` means lags 1 and 2 and ``ar=[1, 4]``
    the two lags 1 and 4 (Stata's numlist). ``0`` and ``[]`` mean no such terms.
    """
    if value is None:
        return None
    problem = AnalysisError(
        "invalid_lags", f"{name} must be a non-negative order (for example {name}=1) or a list "
        f"of distinct positive lags (for example {name}=[1, 4]), none above {MAX_LAG}.")
    if isinstance(value, (bool, float, str, bytes)):
        raise problem
    try:
        order = operator.index(value)
    except TypeError:
        order = None
    if order is not None:
        if not 0 <= order <= MAX_LAG:
            raise problem
        return list(range(1, order + 1))
    if not hasattr(value, "__iter__"):
        raise problem
    lags = []
    for item in value:
        if isinstance(item, (bool, float, str, bytes)):
            raise problem
        try:
            lags.append(operator.index(item))
        except TypeError:
            raise problem from None
    return _checked(lags, name)


def _checked(lags: list[int], name: str) -> list[int]:
    if any(not isinstance(lag, int) or isinstance(lag, bool) or not 1 <= lag <= MAX_LAG
           for lag in lags) or len(set(lags)) != len(lags):
        raise AnalysisError(
            "invalid_lags", f"{name} must list distinct positive lags (for example "
            f"{name}=[1, 4]), none above {MAX_LAG}.")
    return sorted(lags)


def read_layout(spec: Any, k_x: int, k_z: int) -> Layout:
    """Validate the lag, model and distribution options of an arch specification."""
    options = spec.options
    model = options.get("model", "garch")
    arch_lags = _checked(list(options.get("arch", [1])), "arch")
    garch_given = options.get("garch")
    if model == "arch":
        if garch_given:
            raise AnalysisError("invalid_spec", "model='arch' has no GARCH terms; use "
                                "model='garch' for a GARCH model or drop the garch option.")
        garch_lags: list[int] = []
    else:
        garch_lags = _checked(list([1] if garch_given is None else garch_given), "garch")
    if not arch_lags:
        raise AnalysisError("invalid_spec", "The variance equation needs at least one ARCH "
                            "term (arch=1): without it the GARCH coefficients are not identified.")
    if model == "igarch" and not garch_lags:
        raise AnalysisError("invalid_spec", "model='igarch' needs at least one GARCH term: the "
                            "ARCH and GARCH coefficients are restricted to sum to one.")
    return Layout(
        model=model, dist=options.get("dist", "normal"), arch_lags=tuple(arch_lags),
        garch_lags=tuple(garch_lags), ar_lags=tuple(_checked(list(options.get("ar") or []), "ar")),
        ma_lags=tuple(_checked(list(options.get("ma") or []), "ma")),
        archm=options.get("archm"), k_x=k_x, k_z=k_z)
