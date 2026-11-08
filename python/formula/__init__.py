from collections.abc import Sequence

from . import _formula
from ._formula import FittedFormula, FittedFormulas, Formulas, Role

__all__ = ["FittedFormula", "FittedFormulas", "Formulas", "Role", "parse"]


def parse(formulas: str | Sequence[str]) -> Formulas:
    if isinstance(formulas, str):
        return _formula.parse(formulas)

    if isinstance(formulas, (bytes, bytearray)):
        raise TypeError("expected a formula string or an iterable of strings")

    return _formula.parse_many(formulas)
