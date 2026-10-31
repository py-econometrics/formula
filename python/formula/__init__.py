from collections.abc import Iterable

from . import _formula
from ._formula import Specification

__all__ = ["Specification", "parse"]


def parse(formulas: str | Iterable[str]) -> Specification:
    if isinstance(formulas, str):
        return _formula.parse(formulas)

    if isinstance(formulas, (bytes, bytearray)):
        raise TypeError("expected a formula string or an iterable of strings")

    return _formula.parse_many(list(formulas))
