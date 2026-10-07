"""General linear model with sum-to-zero (effect) coding: the engine of ANOVA and MANOVA.

Coding. A factor with L levels contributes L - 1 columns: level l < L is the unit
vector e_l and the last level is -1 in every column, so the columns of a factor sum
to zero over its levels. Interaction columns are row-wise Kronecker products of
the factor blocks; covariates enter as they are and multiply the factor blocks in
factor-by-covariate terms. With every cell of an interaction observed the design
has full column rank and "dropping the columns of a term" is the Type III
hypothesis of SAS and SPSS (equal-weighted marginal means), also for unbalanced
data. The model must respect marginality: an interaction is accepted only together
with every effect it contains (a#b#x needs a#x, b#x and x), because only then do
the sum-to-zero columns span the same model as the cell indicators of SPSS, SAS
and Stata.

Reduction to a small problem. One blocked Householder QR of [X, Y] (n rows) gives
its triangular factor R with R'R = [X, Y]'[X, Y]. Every least-squares fit on any
subset of the columns of X is then a fit on the corresponding columns of R, a
(k+m)-row problem solved by QR again (``linalg.least_squares``): no normal
equations, O(n k^2) work once and O(k^3) per hypothesis.

Sums of squares (and SSCP matrices when Y has several columns). The hypothesis
matrix of a term T inside a model M containing it is the Wald form

    H = B_T' [ (X_M'X_M)^{-1}_TT ]^{-1} B_T  =  SSR(M without T) - SSR(M),

computed from the coefficients, so that small effects lose no digits to the
subtraction of two residual sums of squares. The model M depends on the type:

    Type I    M = the terms up to and including T, in model order (sequential);
    Type II   M = every term that does not contain T (T adjusted for all effects
              that do not contain it; U contains T when both have the same
              covariates and the factors of T are a proper subset of those of U);
    Type III  M = the full model (T adjusted for everything, the SPSS default).

Covariates with a large level. The design is built with covariates centred at
their means, x - xbar, whenever the model is hierarchical in its covariates (for
every term F#x1#..#xr it also has F with each subset of those covariates, which
includes every model whose covariates enter only linearly). The centred and the
raw designs then span the same space, X_raw = X_c T with a sparse matrix T made
of products of the means, so the raw coefficients and (X'X)^{-1} follow exactly,

    b_raw = T^{-1} b_c,   (X_raw'X_raw)^{-1} = T^{-1} (X_c'X_c)^{-1} T^{-T},

where T^{-1} is T with the signs of the means reversed. Every hypothesis is still
the hypothesis on the RAW coefficients (the intercept and a factor in a model
with factor-by-covariate slopes are tested at covariate = 0, as SPSS and Stata
do), but a covariate such as 1e8 + noise costs no digits and is not mistaken for
a constant. A model that is not hierarchical in its covariates is fitted on the
raw covariates.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call
from openecon.engines import linalg

_BLOCK_ELEMENTS = 1 << 22
_MAX_COLUMNS = 2000
_SEPARATORS = ("#", "*", ":")


@dataclass(frozen=True)
class Term:
    """A model effect: the product of some factors and some covariates."""

    name: str
    factors: tuple[str, ...] = ()
    covariates: tuple[str, ...] = ()

    def contains(self, other: Term) -> bool:
        """SPSS's containment: same covariates and a proper superset of the factors."""
        return set(self.covariates) == set(other.covariates) \
            and set(other.factors) < set(self.factors)


INTERCEPT = Term("Intercept")
# Row labels of the ANOVA tables that a factor or covariate must not be called.
RESERVED = ("Intercept", "corrected_model", "error", "total", "corrected_total")


def _key(term: Term) -> tuple[frozenset[str], frozenset[str]]:
    return frozenset(term.factors), frozenset(term.covariates)


def term_name(names: Sequence[str]) -> str:
    return "#".join(names)


def parse_interaction(entry: Any, factors: Sequence[str], covariates: Sequence[str]) -> Term:
    if isinstance(entry, str):
        names = [entry]
        for separator in _SEPARATORS:
            names = [piece.strip() for name in names for piece in name.split(separator)]
    elif isinstance(entry, (list, tuple)) and all(isinstance(name, str) for name in entry):
        names = list(entry)
    else:
        raise AnalysisError("invalid_spec", "Each interaction is a list of column names, for "
                            "example interactions=[['a', 'b']], or a string such as 'a#b'.")
    unknown = [name for name in names if name not in factors and name not in covariates]
    if unknown or len(names) < 2 or len(set(names)) != len(names):
        raise AnalysisError("invalid_spec", f"Interaction {entry!r} must combine at least two "
                            "different columns listed in factors or covariates.")
    inside = tuple(name for name in factors if name in names)
    numeric = tuple(name for name in covariates if name in names)
    return Term(term_name([*inside, *numeric]), inside, numeric)


def build_terms(factors: Sequence[str], covariates: Sequence[str], interactions: Any,
                slopes: bool = False) -> list[Term]:
    """Intercept, covariates, main effects, interactions (lowest order first), slopes."""
    terms = [INTERCEPT, *(Term(name, (), (name,)) for name in covariates),
             *(Term(name, (name,)) for name in factors)]
    if interactions == "full":
        for order in range(2, len(factors) + 1):
            terms.extend(Term(term_name(combo), tuple(combo))
                         for combo in itertools.combinations(factors, order))
    elif isinstance(interactions, (list, tuple)):
        terms.extend(parse_interaction(entry, factors, covariates) for entry in interactions)
    elif interactions not in ("none", None):
        raise AnalysisError("invalid_option", "interactions must be 'full', 'none' or a list of "
                            "interactions such as [['a', 'b']].")
    if slopes:
        terms.extend(Term(term_name([factor, name]), (factor,), (name,))
                     for name in covariates for factor in factors)
    keys = [(frozenset(term.factors), frozenset(term.covariates)) for term in terms]
    if len(set(keys)) != len(keys):
        raise AnalysisError("invalid_spec", "A model term is listed more than once.")
    # Marginality. SPSS, SAS and Stata build an interaction from indicator columns of
    # its cells, which span the interaction AND every effect contained in it; products
    # of sum-to-zero columns span the interaction contrasts only. The two agree exactly
    # when the contained effects are in the model, so a model that omits them is refused
    # instead of being fitted with a different meaning.
    present = set(keys)
    for term in terms:
        absent = [term_name([*combo, *term.covariates])
                  for size in range(len(term.factors))
                  for combo in itertools.combinations(term.factors, size)
                  if (frozenset(combo), frozenset(term.covariates)) not in present]
        if absent:
            raise AnalysisError(
                "invalid_spec", f"The interaction {term.name} needs the effects it contains in "
                f"the model: add {', '.join(absent)} to interactions. Models that omit "
                "lower-order terms of an interaction are not supported.")
    names = [term.name for term in terms[1:]]
    clashes = sorted({name for name in names if name in RESERVED or names.count(name) > 1})
    if clashes:
        raise AnalysisError("invalid_spec", f"The term name(s) {', '.join(clashes)} are ambiguous "
                            f"in the ANOVA table (its rows include {', '.join(RESERVED)}, and "
                            "interactions are named with '#'); rename the column(s) in the data.")
    return terms


def check_sequential_order(terms: Sequence[Term]) -> None:
    """Type I sums of squares need every term after the effects it contains.

    Entered earlier, the cell-indicator columns of SPSS, SAS and Stata would absorb
    the contained effect, which sum-to-zero columns cannot reproduce.
    """
    seen: set[tuple[frozenset[str], frozenset[str]]] = set()
    for term in terms:
        late = [other.name for other in terms if term.contains(other) and _key(other) not in seen]
        if late:
            raise AnalysisError(
                "invalid_spec", f"With ss_type=1 the interaction {term.name} must come after "
                f"the effects it contains: list {', '.join(late)} before it in interactions.")
        seen.add(_key(term))


def kron(a: Tensor, b: Tensor) -> Tensor:
    """Kronecker product of two matrices (or two vectors) by broadcasting.

    ``torch.kron`` relies on views and fails for some stride layouts of n-by-1
    factors returned by LAPACK routines; this form has no such restriction.
    """
    if a.ndim == 1:
        return (a[:, None] * b[None, :]).reshape(-1)
    return (a[:, None, :, None] * b[None, :, None, :]).reshape(
        a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])


def effect_coding(levels: int) -> Tensor:
    """The L-by-(L-1) sum-to-zero coding matrix [I; -1']."""
    return torch.cat([torch.eye(levels - 1, dtype=torch.float64),
                      -torch.ones((1, levels - 1), dtype=torch.float64)])


class LinearModel:
    """Sum-to-zero general linear model of Y [n, m] reduced to its triangular factor."""

    def __init__(self, terms: Sequence[Term], codes: dict[str, Tensor],
                 levels: dict[str, list[Any]], covariates: dict[str, Tensor], y: Tensor,
                 shift: Tensor | None = None):
        # ``y`` may be supplied minus a constant per column (``shift``): every fit has an
        # intercept, so only the intercept and predictions depend on it, and they add
        # it back. This keeps full precision when the outcome has a large level.
        self.shift = torch.zeros(y.shape[1], dtype=torch.float64) if shift is None else shift
        self.terms = list(terms)
        self.codes, self.levels, self.covariates = codes, levels, covariates
        self.n, self.m = y.shape
        self._keys = {_key(term): term for term in self.terms}
        self.centred = bool(covariates) and self._closed(self.terms)
        self.centres = {name: float(values.mean()) if self.centred else 0.0
                        for name, values in covariates.items()}
        for name, labels in levels.items():
            if len(labels) < 2:
                raise AnalysisError("single_level", f"Factor '{name}' has only one level in the "
                                    "sample; remove it from the model.")
        self.codings = {name: effect_coding(len(labels)) for name, labels in levels.items()}
        self.columns: dict[str, list[int]] = {}
        position = 0
        for term in self.terms:
            width = 1
            for name in term.factors:
                width *= len(levels[name]) - 1
            self.columns[term.name] = list(range(position, position + width))
            position += width
        self.k = position
        if self.k > _MAX_COLUMNS:
            raise AnalysisError("design_too_large", f"The model has {self.k} parameters; factorial "
                                f"designs are limited to {_MAX_COLUMNS}. Remove interactions or "
                                "factors with very many levels.")
        if self.n - self.k < 1:
            raise AnalysisError("no_residual_df", f"The model has {self.k} parameters for "
                                f"{self.n} observations, leaving no degrees of freedom for the "
                                "error term. Remove interactions or add observations.")
        self._check_cells()
        self.factor = self._compress(y)
        # X_raw = X_c @ recentre and X_c = X_raw @ uncentre (identities without centring).
        self.recentre, self.uncentre = self._transform(1.0), self._transform(-1.0)
        self.full = self._fit(list(range(self.k)))
        self.rank = self.k
        self.df_error = self.n - self.rank
        self.beta = self.full.beta if self.full.beta.ndim == 2 else self.full.beta[:, None]
        residual = self.full.resid if self.full.resid.ndim == 2 else self.full.resid[:, None]
        self.error = linalg.symmetrize(residual.T @ residual)       # E, the residual SSCP

    def _lower(self, term: Term) -> list[tuple[Term | None, tuple[str, ...]]]:
        """For every proper subset S of the covariates of ``term``: (the term with the same
        factors and covariates S, or None when the model lacks it; the covariates left out)."""
        out = []
        for size in range(len(term.covariates)):
            for subset in itertools.combinations(term.covariates, size):
                rest = tuple(name for name in term.covariates if name not in subset)
                out.append((self._keys.get((frozenset(term.factors), frozenset(subset))), rest))
        return out

    def _closed(self, members: Sequence[Term]) -> bool:
        """True when ``members`` is hierarchical in its covariates (see the module docstring)."""
        inside = {_key(term) for term in members}
        return all(lower is not None and _key(lower) in inside
                   for term in members for lower, _ in self._lower(term))

    def _transform(self, sign: float) -> Tensor:
        """T(sign * centres): the columns of a term F#x1..xr in terms of the columns built
        with x - sign * centre, i.e. sum over subsets S of prod_{j not in S} (sign c_j) F#x_S."""
        matrix = torch.eye(self.k, dtype=torch.float64)
        if not self.centred:
            return matrix
        for term in self.terms:
            for lower, rest in self._lower(term):
                coefficient = 1.0
                for name in rest:
                    coefficient *= sign * self.centres[name]
                matrix[self.columns[lower.name], self.columns[term.name]] = coefficient
        return matrix

    def _check_cells(self) -> None:
        """Every cell of each interaction must be observed (no Type IV hypotheses)."""
        for term in self.terms:
            if len(term.factors) < 2:
                continue
            index = torch.zeros(self.n, dtype=torch.int64)
            cells = 1
            for name in term.factors:
                size = len(self.levels[name])
                index = index * size + self.codes[name]
                cells *= size
            observed = int(torch.unique(index).numel())
            if observed < cells:
                raise AnalysisError(
                    "empty_cells", f"{cells - observed} of the {cells} cells of "
                    f"{' x '.join(term.factors)} have no observations. Type I-III sums of "
                    "squares need every cell of an interaction (Type IV is not implemented): "
                    "remove the interaction (interactions='none' or a list) or combine levels.")

    def _block(self, low: int, high: int) -> Tensor:
        pieces = []
        for term in self.terms:
            part = torch.ones((high - low, 1), dtype=torch.float64)
            for name in term.factors:
                coded = self.codings[name][self.codes[name][low:high]]
                part = (part[:, :, None] * coded[:, None, :]).reshape(high - low, -1)
            for name in term.covariates:
                part = part * (self.covariates[name][low:high, None] - self.centres[name])
            pieces.append(part)
        return torch.cat(pieces, dim=1)

    def _compress(self, y: Tensor) -> Tensor:
        width = self.k + self.m
        factor = torch.empty((0, width), dtype=torch.float64)
        step = max(width, _BLOCK_ELEMENTS // width)
        for low in range(0, self.n, step):
            high = min(low + step, self.n)
            block = torch.cat([self._block(low, high), y[low:high]], dim=1)
            factor = torch.linalg.qr(torch.cat([factor, block]), mode="r").R
        return factor

    def _fit(self, columns: list[int], design: Tensor | None = None) -> linalg.LeastSquares:
        """Least squares of the outcomes on ``columns`` of the (centred) design, or on
        ``design``, a matrix with the cross products of those columns in another basis."""
        if design is None:
            design = self.factor[:, columns]
        fit = kernel_call(linalg.least_squares, design.contiguous(),
                          self.factor[:, self.k:].contiguous(), drop_collinear=True)
        if fit.omitted:
            names = sorted({term.name for term in self.terms
                            if any(columns[i] in self.columns[term.name] for i in fit.omitted)})
            raise AnalysisError(
                "collinear_design", "The design matrix is rank deficient in: "
                f"{', '.join(names)}. Sums of squares are not defined for a confounded effect: "
                "remove it, or check for empty cells, constant covariates and covariates that "
                "are constant within the cells of a factor.")
        return fit

    def hypothesis(self, term: Term, ss_type: int = 3) -> Tensor:
        """The [m, m] hypothesis SSCP matrix of ``term`` (its sum of squares when m = 1)."""
        if ss_type == 3:
            members = self.terms
        elif ss_type == 2:
            members = [other for other in self.terms if not other.contains(term)]
        else:
            members = self.terms[:self.terms.index(term) + 1]
        columns = [column for other in members for column in self.columns[other.name]]
        target = set(self.columns[term.name])
        positions = [i for i, column in enumerate(columns) if column in target]
        if not self.centred or self._closed(members):
            fit = self.full if len(columns) == self.k else self._fit(columns)
            beta = fit.beta if fit.beta.ndim == 2 else fit.beta[:, None]
            inverse = fit.xtx_inv
            if self.centred:
                # Rows of T^{-1} for the tested coefficients: raw = T^{-1} centred.
                back = self.uncentre[columns][:, columns][positions]
                block, middle = back @ beta, back @ inverse @ back.T
            else:
                block, middle = beta[positions], inverse[positions][:, positions]
        else:
            # The sub-model lacks a lower-order covariate term, so its raw columns span a
            # different space from its centred ones: fit the raw columns X_c T[:, M].
            fit = self._fit(columns, self.factor[:, :self.k] @ self.recentre[:, columns])
            beta = fit.beta if fit.beta.ndim == 2 else fit.beta[:, None]
            block, middle = beta[positions], fit.xtx_inv[positions][:, positions]
        if term.name == INTERCEPT.name:
            block = block + self.shift[None, :]
        middle = linalg.symmetrize(middle)
        solved = kernel_call(linalg.cholesky_solve, middle, block, code="collinear_design",
                             what=f"covariance of the coefficients of {term.name}")
        return linalg.symmetrize(block.T @ solved)

    def df(self, term: Term) -> int:
        return len(self.columns[term.name])

    def marginal_vector(self, cell: dict[str, int], means: dict[str, float]) -> Tensor:
        """L with L'beta = the estimated marginal mean of the cell ``{factor: level index}``.

        Factors not in ``cell`` are averaged with equal weights (their effect codes
        average to zero) and covariates are set to ``means``. The vector applies to
        the fitted coefficients (``self.beta``, ``self.full.xtx_inv``), whose
        covariates are measured from ``self.centres``.
        """
        vector = torch.zeros(self.k, dtype=torch.float64)
        for term in self.terms:
            if any(name not in cell for name in term.factors):
                continue
            part = torch.ones(1, dtype=torch.float64)
            for name in term.factors:
                part = kron(part, self.codings[name][cell[name]])
            for name in term.covariates:
                part = part * (means[name] - self.centres[name])
            vector[self.columns[term.name]] = part
        return vector
