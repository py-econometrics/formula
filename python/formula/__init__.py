from collections.abc import Sequence

from . import _formula
from ._formula import (
    Factor,
    FactorContribution,
    FittedFormula,
    FittedFormulas,
    Formula,
    FormulaColumn,
    Formulas,
    Role,
    TransformCall,
)

__all__ = [
    "Factor",
    "FactorContribution",
    "FittedFormula",
    "FittedFormulas",
    "Formula",
    "FormulaColumn",
    "Formulas",
    "Role",
    "TransformCall",
    "parse",
]


def parse(formulas: str | Sequence[str]) -> Formulas:
    if isinstance(formulas, str):
        return _formula.parse(formulas)

    if isinstance(formulas, (bytes, bytearray)):
        raise TypeError("expected a formula string or an iterable of strings")

    return _formula.parse_many(formulas)
