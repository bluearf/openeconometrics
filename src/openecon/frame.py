"""OpenEconometrics's pandas-compatible frame with dependency-free LaTeX exports."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from openecon.latex import Latex, to_latex


class DataFrame(pd.DataFrame):
    """A pandas DataFrame whose LaTeX exporter does not need Jinja2.

    Frame operations such as ``head``, slicing and ``copy`` keep this class.
    All data, dtypes, index values and attrs retain pandas semantics.
    """

    @property
    def _constructor(self):
        return DataFrame

    def describe(self, percentiles=None, include=None, exclude=None) -> DataFrame:
        """Keep the direct LaTeX API on pandas' descriptive statistics result."""
        return as_frame(super().describe(percentiles=percentiles, include=include, exclude=exclude))

    def to_latex(self, buf=None, *, index: bool = True, caption=None, label=None,
                 float_format=None, precision: int = 4, na_rep: str = "",
                 header=True, columns=None, column_format=None, longtable: bool | None = None,
                 booktabs: bool = True, bold_rows: bool = False,
                 encoding: str = "utf-8", notes=None, font_size: str = "small",
                 max_width: str | None = r"\linewidth") -> Latex | None:
        """Return complete, escaped LaTeX, or write it to a file/buffer."""
        return to_latex(self, buf, index=index, caption=caption, label=label,
                        float_format=float_format, precision=precision, na_rep=na_rep,
                        header=header, columns=columns, column_format=column_format,
                        longtable=longtable, booktabs=booktabs, bold_rows=bold_rows,
                        encoding=encoding, notes=notes, font_size=font_size, max_width=max_width)

    @property
    def latex(self) -> Latex:
        """Complete LaTeX table source with the row index included."""
        source = self.to_latex()
        assert source is not None
        return source


def as_frame(frame: Any) -> DataFrame:
    """Wrap imported data without changing values, dtypes, labels or metadata."""
    if isinstance(frame, DataFrame):
        return frame
    result = DataFrame(frame, copy=False)
    result.attrs = deepcopy(getattr(frame, "attrs", {}))
    return result
