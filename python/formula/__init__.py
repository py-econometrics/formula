from collections.abc import Sequence

from . import _formula
from ._formula import Model, Models, Role, Specification

__all__ = ["Models", "Model", "Role", "Specification", "parse"]


def parse(formulas: str | Sequence[str]) -> Specification:
    if isinstance(formulas, str):
        return _formula.parse(formulas)

    if isinstance(formulas, (bytes, bytearray)):
        raise TypeError("expected a formula string or an iterable of strings")

    return _formula.parse_many(formulas)
