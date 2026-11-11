import math

import polars
import pytest
from polars.testing import assert_series_equal

import formula


@pytest.fixture
def data():
    return polars.DataFrame(
        {"y": [1.0, 2.0, 3.0], "g": ["a", "b", "c"], "x": [2.0, 3.0, 4.0]}
    )


def test_selected_column_provenance_survives_prediction_and_owner_deletion(data):
    model = formula.parse("y ~ C(g, reference='a')*x").fit(data)[0]
    selected = model.select_terms("x:C(g, reference='a')", formula.Role.Independent)
    prediction = selected.transform(
        polars.DataFrame({"g": ["c", "b", "new"], "x": [10.0, 20.0, 30.0]})
    )
    columns = prediction.columns(role=formula.Role.Independent)
    del model, selected, prediction

    assert len(columns) == 2
    for column, level, values in zip(
        columns, ["b", "c"], [[0.0, 20.0, None], [10.0, 0.0, None]], strict=True
    ):
        assert isinstance(column, formula.FormulaColumn)
        assert isinstance(column.term, tuple)
        assert len(column.term) == 2
        term_call = column.term[0].transform
        assert term_call is not None
        assert term_call.name == "C"
        assert column.term[1].name == "x"
        assert isinstance(column.contributions, tuple)
        assert len(column.contributions) == 2
        categorical_contribution, variable_contribution = column.contributions
        categorical = categorical_contribution.factor
        variable = variable_contribution.factor
        assert isinstance(categorical, formula.Factor)
        assert categorical.name is None
        assert categorical.value is None
        call = categorical.transform
        assert isinstance(call, formula.TransformCall)
        assert call.name == "C"
        assert call.parameters == {"reference": "a"}
        assert len(call.inputs) == 1
        assert call.inputs[0].name == "g"
        assert categorical_contribution.key == level
        assert variable.name == "x"
        assert variable.value is None
        assert variable.transform is None
        assert variable_contribution.key is None
        assert_series_equal(column.data, polars.Series(column.data.name, values))


@pytest.mark.parametrize(
    ("values", "dtype", "key_type"),
    [
        ([True, False], polars.Boolean, bool),
        ([1, 2], polars.Int32, int),
        ([1.5, 2.5], polars.Float32, float),
        (["a", "b"], polars.String, str),
        (["a", "b"], polars.Categorical, str),
        (["a", "b"], polars.Enum(["a", "b"]), str),
        ([b"a", b"b"], polars.Binary, bytes),
        ([[1], [2]], polars.List(polars.Int64), polars.Series),
    ],
)
def test_categorical_keys_convert_to_python_values(values, dtype, key_type):
    data = polars.DataFrame(
        {"y": [1.0, 2.0], "g": polars.Series("g", values, dtype=dtype)}
    )
    model = formula.parse("y ~ C(g) - 1").fit(data)[0]
    columns = model.columns(formula.Role.Independent)
    assert len(columns) == 2
    for column, expected in zip(columns, values, strict=True):
        assert len(column.contributions) == 1
        key = column.contributions[0].key
        assert type(key) is key_type
        if isinstance(key, polars.Series):
            assert_series_equal(key, polars.Series("", expected))
        else:
            assert key == expected


def test_nested_transform_inputs_and_parameters():
    data = polars.DataFrame({"y": [1.0, 2.0], "x": [1.0, math.e]})
    model = formula.parse("y ~ poly(log(x), degree=2, raw=true) - 1").fit(data)[0]
    columns = model.columns(formula.Role.Independent)
    assert len(columns) == 2
    for column, basis in zip(columns, [1, 2], strict=True):
        assert len(column.contributions) == 1
        contribution = column.contributions[0]
        factor = contribution.factor
        key = contribution.key
        assert type(key) is int
        assert key == basis
        polynomial = factor.transform
        assert polynomial is not None
        assert polynomial.name == "poly"
        assert polynomial.parameters == {"degree": 2, "raw": True}
        assert type(polynomial.parameters["degree"]) is int
        assert type(polynomial.parameters["raw"]) is bool
        assert len(polynomial.inputs) == 1
        logarithm = polynomial.inputs[0].transform
        assert logarithm is not None
        assert logarithm.name == "log"
        assert logarithm.parameters == {}
        assert len(logarithm.inputs) == 1
        assert logarithm.inputs[0].name == "x"
        assert logarithm.inputs[0].transform is None
        assert column.data.to_list() == pytest.approx([0.0, 1.0])


@pytest.mark.parametrize(("literal", "expected"), [("2", 2), ("2.0", 2.0)])
def test_literal_transform_inputs_preserve_python_types(data, literal, expected):
    model = formula.parse(f"y ~ add(x, {literal}) - 1").fit(data)[0]
    column = model.columns(formula.Role.Independent)[0]
    contribution = column.contributions[0]
    factor = contribution.factor
    assert contribution.key is None
    call = factor.transform
    assert call is not None
    assert call.name == "add"
    assert call.parameters == {}
    assert len(call.inputs) == 2
    variable, constant = call.inputs
    assert variable.name == "x"
    assert variable.value is None
    assert variable.transform is None
    assert constant.name is None
    assert constant.transform is None
    assert type(constant.value) is type(expected)
    assert constant.value == expected
    assert column.data.to_list() == [4.0, 5.0, 6.0]


def test_provenance_properties_are_read_only_and_containers_are_copies(data):
    model = formula.parse("y ~ C(g, reference='a') - 1").fit(data)[0]
    column = model.columns(formula.Role.Independent)[0]
    contribution = column.contributions[0]
    factor = contribution.factor
    call = factor.transform
    assert call is not None
    for target, attribute, value in [
        (column, "data", None),
        (column, "term", ()),
        (column, "contributions", ()),
        (contribution, "factor", None),
        (contribution, "key", "changed"),
        (factor, "name", "changed"),
        (factor, "value", 10),
        (factor, "transform", None),
        (call, "name", "changed"),
        (call, "inputs", []),
        (call, "parameters", {}),
    ]:
        with pytest.raises(AttributeError):
            setattr(target, attribute, value)

    inputs = call.inputs
    inputs.clear()
    parameters = call.parameters
    parameters["reference"] = "changed"
    assert len(call.inputs) == 1
    assert call.inputs[0].name == "g"
    assert call.parameters == {"reference": "a"}


def test_intercept_and_zero_column_provenance():
    training = polars.DataFrame({"y": [1.0, 2.0], "g": ["a", "a"]})
    model = formula.parse("y ~ C(g, reference='a')").fit(training)[0]
    columns = model.columns(formula.Role.Independent)
    assert len(columns) == 1
    assert columns[0].term == ()
    assert columns[0].contributions == ()
    assert columns[0].data.to_list() == [1.0, 1.0]
    selected = model.select_terms("C(g, reference='a')", formula.Role.Independent)
    assert selected.columns(formula.Role.Independent) == []
    prediction = polars.DataFrame({"g": ["a", "new", "a"]})
    for data in [prediction, prediction.head(0)]:
        transformed = selected.transform(data)
        assert transformed.columns(formula.Role.Independent) == []
        assert (
            transformed.select_terms(
                "C(g, reference='a')", formula.Role.Independent
            ).columns(formula.Role.Independent)
            == []
        )


@pytest.mark.xfail(
    strict=True,
    reason="Polars DataFrame export loses zero-column height: formula#96",
)
def test_zero_column_frame_export_preserves_height():
    data = polars.DataFrame({"y": [1.0, 2.0], "g": ["a", "a"]})
    selected = (
        formula.parse("y ~ C(g, reference='a')")
        .fit(data)[0]
        .select_terms("C(g, reference='a')", formula.Role.Independent)
    )
    assert selected.frame().shape == (data.height, 0)


def test_columns_reject_missing_or_invalid_roles(data):
    model = formula.parse("y ~ x").fit(data)[0]
    with pytest.raises(KeyError, match="FixedEffect"):
        model.columns(formula.Role.FixedEffect)
    with pytest.raises(TypeError):
        model.columns("Independent")  # ty: ignore[invalid-argument-type]


def test_varying_slope_provenance_survives_owner_deletion():
    data = polars.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [4.0, 5.0, 6.0],
            "firm": ["a", "b", "a"],
            "year": [2020, 2020, 2021],
            "time": polars.Series([1, 2, 3], dtype=polars.Int32),
            "exposure": [0.5, 1.0, 1.5],
        }
    )
    model = formula.parse("y ~ x | firm:year[log(time), exposure]").fit(data)[0]
    selected = model.select_terms(
        "firm:year[log(time), exposure]", formula.Role.FixedEffect
    )
    predicted = selected.transform(data.select("firm", "year", "time", "exposure"))
    columns = predicted.columns(role=formula.Role.FixedEffect)
    del model, selected, predicted

    assert len(columns) == 6
    for index, column in enumerate(columns):
        (contribution,) = column.contributions
        role = "group" if index % 2 == 0 else "loading"
        assert contribution.key == role
        assert type(contribution.key) is str
        assert column.data.dtype == (
            polars.UInt32 if role == "group" else polars.Float64
        )
        (term_factor,) = column.term
        term_call = term_factor.transform
        assert term_call is not None
        assert term_call.name == "G"

        call = contribution.factor.transform
        assert call is not None
        assert call.name == "G"
        assert call.parameters == {}
        group, loading = call.inputs
        encoding = group.transform
        assert encoding is not None
        assert encoding.name == "F"
        assert [factor.name for factor in encoding.inputs] == ["firm", "year"]

        if index < 2:
            assert type(loading.value) is int
            assert loading.value == 1
        elif index < 4:
            transform = loading.transform
            assert transform is not None
            assert transform.name == "log"
            assert [factor.name for factor in transform.inputs] == ["time"]
        else:
            assert loading.name == "exposure"
