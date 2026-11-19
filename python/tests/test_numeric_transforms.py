"""pow(), center() and scale(), checked against direct calculations."""

import polars as pl
import pytest
from polars.testing import assert_series_equal

import formula


@pytest.fixture
def data():
    return pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [1, 2, 3, 6]})


def predictor(source, data, new=None):
    model = formula.parse(f"y ~ {source} - 1").fit(data)[0]
    if new is not None:
        model = model.transform(new)
    return model.select(formula.Role.Independent).frame().to_series()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("pow(x, 2)", pl.col("x").cast(pl.Float64).pow(2)),
        ("pow(x, 0.5)", pl.col("x").cast(pl.Float64).sqrt()),
        ("center(x)", pl.col("x") - pl.col("x").mean()),
        ("scale(x)", (pl.col("x") - pl.col("x").mean()) / pl.col("x").std()),
        ("pow(center(x), 2)", (pl.col("x") - pl.col("x").mean()).pow(2)),
    ],
)
def test_values_match_direct_calculations(data, source, expected):
    actual = predictor(source, data)
    assert_series_equal(actual, data.select(expected.alias(source)).to_series())


@pytest.mark.parametrize("source", ["center(x)", "scale(x)"])
def test_prediction_reuses_the_fitted_statistics(data, source):
    new = pl.DataFrame({"y": [0.0, 0.0], "x": [3, 100]})
    mean, deviation = data["x"].mean(), data["x"].std()
    scale = deviation if source == "scale(x)" else 1.0
    expected = [(3 - mean) / scale, (100 - mean) / scale]
    assert predictor(source, data, new).to_list() == pytest.approx(expected)


@pytest.mark.parametrize("source", ["scale(x)", "center(x)"], ids=["constant", "empty"])
def test_degenerate_inputs_are_rejected(source):
    data = pl.DataFrame({"y": [1.0, 2.0], "x": [2, 2]})
    if source == "center(x)":
        data = data.head(0)
    with pytest.raises(ValueError, match=source.split("(")[0]):
        formula.parse(f"y ~ {source}").fit(data)
