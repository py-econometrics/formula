"""Conversion of Polars frames between Python and the Rust extension.

`_roundtrip` converts a frame to a Rust `DataFrame` and back without evaluating
a formula, so these tests exercise only the conversion. The extension defines it
only when built with the `testing` Cargo feature, which `pixi run develop` enables.
"""

import datetime as dt
import decimal
import inspect
import math
from collections.abc import Callable

import polars
import polars.datatypes
import pytest
from polars.testing import assert_frame_equal

from formula import _formula

_roundtrip: Callable[[polars.DataFrame], polars.DataFrame] = _formula._roundtrip  # ty: ignore[unresolved-attribute]


def _floats(dtype: type[polars.DataType] | polars.DataType) -> polars.Series:
    return polars.Series([1.5, None, math.nan, math.inf, -math.inf], dtype=dtype)


def _strings(dtype: type[polars.DataType] | polars.DataType) -> polars.Series:
    return polars.Series(["a", None, "b"], dtype=dtype)


_TIMESTAMPS = [dt.datetime(2020, 1, 1, 12), None, dt.datetime(2021, 6, 1)]
_DURATIONS = [dt.timedelta(days=1), None, dt.timedelta(microseconds=1)]

# Each sample contains at least one null.
SUPPORTED: dict[str, polars.Series] = {
    **{
        str(dtype): polars.Series([1, None, 2], dtype=dtype)
        for dtype in (
            polars.Int8,
            polars.Int16,
            polars.Int32,
            polars.Int64,
            polars.Int128,
            polars.UInt8,
            polars.UInt16,
            polars.UInt32,
            polars.UInt64,
            polars.UInt128,
        )
    },
    "Float16": _floats(polars.Float16),
    "Float32": _floats(polars.Float32),
    "Float64": _floats(polars.Float64),
    "Boolean": polars.Series([True, None, False]),
    "String": _strings(polars.String),
    "Binary": polars.Series([b"a", None, b"\x00"]),
    "Null": polars.Series([None, None]),
    "Date": polars.Series([dt.date(2020, 1, 1), None]),
    "Time": polars.Series([dt.time(1, 2, 3), None]),
    **{
        f"Datetime({unit}, {zone})": polars.Series(
            _TIMESTAMPS, dtype=polars.Datetime(unit, zone)
        )
        for unit in ("ns", "us", "ms")
        for zone in (None, "Europe/Vienna")
    },
    **{
        f"Duration({unit})": polars.Series(_DURATIONS, dtype=polars.Duration(unit))
        for unit in ("ns", "us", "ms")
    },
    "Decimal(38, 2)": polars.Series(
        [decimal.Decimal("1.25"), None], dtype=polars.Decimal(38, 2)
    ),
    "Decimal(5, 0)": polars.Series(
        [decimal.Decimal(12345), None], dtype=polars.Decimal(5, 0)
    ),
    "Categorical": _strings(polars.Categorical),
    "Categorical(named)": _strings(
        polars.Categorical(polars.Categories("colors", "formula-tests"))
    ),
    "Categorical(UInt8)": _strings(
        polars.Categorical(polars.Categories("small", physical=polars.UInt8))
    ),
    # An Enum dtype includes its unused categories and their order, so comparing
    # dtypes checks that both survive the round trip.
    "Enum": _strings(polars.Enum(["b", "unused", "a"])),
    "Enum(70000 categories)": _strings(
        polars.Enum(["a", "b", *(f"c{index}" for index in range(70_000))])
    ),
    "List(Int64)": polars.Series([[1, None], None, []]),
    "Array(Int64, 2)": polars.Series(
        [[1, None], None], dtype=polars.Array(polars.Int64, 2)
    ),
    "Struct": polars.Series([{"a": 1, "b": "x"}, None, {"a": None, "b": None}]),
}


class _Point(polars.BaseExtension):
    def __init__(self) -> None:
        super().__init__("formula-tests.point", polars.Struct({"x": polars.Float64}))


polars.register_extension_type("formula-tests.point", _Point)

# Dtypes that Python can export but the extension does not convert faithfully.
UNSUPPORTED: dict[type[polars.DataType], tuple[polars.Series, str]] = {
    polars.Object: (
        polars.Series([object(), None], dtype=polars.Object),
        "arrives in Rust as Binary holding the objects' memory addresses",
    ),
    polars.Extension: (
        polars.Series([{"x": 1.0}, None]).ext.to(_Point()),
        "the extension is built without the dtype-extension feature",
    ),
}

# Polars releases before 2.0 have no Map dtype.
if hasattr(polars, "Map"):
    UNSUPPORTED[polars.Map] = (
        polars.Series(
            [[{"key": "a", "value": 1}], None],
            dtype=polars.Map(polars.String, polars.Int64),
        ),
        "Rust Polars 0.55 has no Map dtype and imports it as List(Struct)",
    )

# Requesting `Unknown` for a column gives `Null`, so no column has this dtype.
NOT_COLUMN_DTYPES: set[type[polars.DataType]] = {polars.Unknown}


def _concrete_dtype_classes() -> set[type[polars.DataType]]:
    classes = {
        cls
        for _, cls in inspect.getmembers(polars.datatypes, inspect.isclass)
        if issubclass(cls, polars.DataType)
    }
    # A class that another listed class derives from, such as `IntegerType`, is abstract.
    return {
        cls
        for cls in classes
        if not any(other is not cls and issubclass(other, cls) for other in classes)
    }


def test_every_dtype_is_classified():
    classified = (
        {type(series.dtype) for series in SUPPORTED.values()}
        | set(UNSUPPORTED)
        | NOT_COLUMN_DTYPES
    )

    unclassified = _concrete_dtype_classes() - classified

    assert not unclassified, (
        f"classify these dtypes in {__name__}: {sorted(cls.__name__ for cls in unclassified)}"
    )


def _list(series: polars.Series) -> polars.Series:
    return polars.Series(
        series.name, [series.to_list(), None], dtype=polars.List(series.dtype)
    )


def _array(series: polars.Series) -> polars.Series:
    return polars.Series(
        series.name,
        [series.to_list(), None],
        dtype=polars.Array(series.dtype, len(series)),
    )


def _struct(series: polars.Series) -> polars.Series:
    return series.to_frame().to_struct(series.name)


VARIANTS: dict[str, Callable[[polars.Series], polars.Series]] = {
    "plain": lambda series: series,
    "empty": lambda series: series.clear(),
    "chunked": lambda series: polars.concat([series, series], rechunk=False),
    "sliced": lambda series: series.slice(1),
    "in List": _list,
    "in Array": _array,
    "in Struct": _struct,
}


@pytest.mark.parametrize("variant", VARIANTS.values(), ids=VARIANTS.keys())
@pytest.mark.parametrize("series", SUPPORTED.values(), ids=SUPPORTED.keys())
def test_roundtrip_preserves_column(series, variant):
    frame = variant(series.alias("x")).to_frame()

    assert_frame_equal(_roundtrip(frame), frame)


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(series, marks=pytest.mark.xfail(strict=True, reason=reason))
        for series, reason in UNSUPPORTED.values()
    ],
    ids=[cls.__name__ for cls in UNSUPPORTED],
)
def test_roundtrip_unsupported_dtype(series):
    frame = series.alias("x").to_frame()

    assert_frame_equal(_roundtrip(frame), frame)


def test_roundtrip_preserves_columns_with_different_chunks():
    frame = polars.DataFrame(
        [
            polars.concat(
                [polars.Series("a", [1]), polars.Series("a", [2])], rechunk=False
            ),
            polars.Series("b", ["x", "y"]),
        ]
    )

    assert_frame_equal(_roundtrip(frame), frame)


@pytest.mark.xfail(
    strict=True,
    reason="pyo3-polars infers the height from the columns, so a frame without columns gets height 0",
)
def test_roundtrip_preserves_height_without_columns():
    frame = polars.DataFrame({"x": [1, 2, 3]}).drop("x")

    assert _roundtrip(frame).shape == (3, 0)
