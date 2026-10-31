"""Type declarations for the native extension."""

from collections.abc import Sequence
from typing import ClassVar

import polars

class Specification:
    def materialize(self, data: polars.DataFrame) -> Models: ...

class Model:
    def frame(self, role: Role) -> polars.DataFrame | None: ...

class Models:
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Model: ...

class Role:
    Dependent: ClassVar[Role]
    Independent: ClassVar[Role]
    FixedEffect: ClassVar[Role]

def parse(formulas: str) -> Specification: ...
def parse_many(formulas: Sequence[str]) -> Specification: ...
