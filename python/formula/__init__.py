from collections.abc import Sequence

from . import _formula
from ._formula import (
    Factor,
    FactorContribution,
    FittedFormula,
    FittedFormulas,
    FittedTerm,
    FixedEffect,
    FixedEffectEncoding,
    Formula,
    FormulaColumn,
    Formulas,
    Role,
    TransformCall,
    TransformState,
)

__all__ = [
    "Factor",
    "FactorContribution",
    "FittedFormula",
    "FittedFormulas",
    "FittedTerm",
    "FixedEffect",
    "Formula",
    "FormulaColumn",
    "Formulas",
    "FixedEffectEncoding",
    "Role",
    "TransformCall",
    "TransformState",
    "parse",
]


def parse(formulas: str | Sequence[str]) -> Formulas:
    if isinstance(formulas, str):
        return _formula.parse(formulas)

    if isinstance(formulas, (bytes, bytearray)):
        raise TypeError("expected a formula string or an iterable of strings")

    return _formula.parse_many(formulas)
