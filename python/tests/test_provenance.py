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


@pytest.mark.parametrize(
    "dtype", [polars.String, polars.Categorical, polars.Enum(["a", "b", "c"])]
)
@pytest.mark.parametrize(
    ("argument", "parameter", "remaining"),
    [
        ("'a'", "a", ["b", "c"]),
        ("['a']", ["a"], ["b", "c"]),
        ("['a', 'b']", ["a", "b"], ["c"]),
        ("['a', 'a']", ["a", "a"], ["b", "c"]),
        ("['a', 'b', 'c']", ["a", "b", "c"], []),
    ],
)
def test_categorical_omit_scalar_and_list(argument, parameter, remaining, dtype):
    training = polars.DataFrame(
        {"y": [1.0, 2.0, 3.0], "g": polars.Series(["a", "b", "c"], dtype=dtype)}
    )
    model = formula.parse(f"y ~ C(g, omit={argument}) - 1").fit(training)[0]
    (term,) = model.terms(formula.Role.Independent)
    call = term.factors[0].call
    assert call is not None
    assert call.parameters == {"omit": parameter}
    assert [column.contributions[0].key for column in term.columns] == remaining
    assert [column.data.name for column in term.columns] == [
        f"C(g, omit={argument})[`{level}`]" for level in remaining
    ]

    prediction = model.select(formula.Role.Independent).transform(
        polars.DataFrame({"g": polars.Series(["a", "b", "c", None], dtype=dtype)})
    )
    columns = prediction.columns(formula.Role.Independent)
    assert len(columns) == len(remaining)
    for column, level in zip(columns, remaining, strict=True):
        assert column.data.to_list() == [
            value == level for value in ["a", "b", "c"]
        ] + [None]


def test_categorical_bin_parameters_convert_to_a_dict(data):
    model = formula.parse("y ~ C(g, bin={'ab': ['a', 'b']}) - 1").fit(data)[0]
    (term,) = model.terms(formula.Role.Independent)
    call = term.factors[0].call
    assert call is not None
    assert call.parameters == {"bin": {"ab": ["a", "b"]}}


@pytest.mark.parametrize("argument", ["['a', 'missing']", "['a', 1]"])
def test_categorical_omit_validates_every_list_element(data, argument):
    with pytest.raises(ValueError):
        formula.parse(f"y ~ C(g, omit={argument})").fit(data)


@pytest.mark.parametrize("keyword", ["degree", "raw"])
def test_polynomial_rejects_list_parameters(keyword):
    with pytest.raises(SyntaxError, match=keyword):
        formula.parse(f"y ~ poly(x, {keyword}=[1])")


def test_selected_term_and_columns_survive_prediction_and_owner_deletion(data):
    model = formula.parse("y ~ C(g, omit='a')*x").fit(data)[0]
    selected = model.select_terms("x:C(g, omit='a')", formula.Role.Independent)
    prediction = selected.transform(
        polars.DataFrame({"g": ["c", "b", "new"], "x": [10.0, 20.0, 30.0]})
    )
    terms = prediction.terms(role=formula.Role.Independent)
    assert isinstance(terms, list)
    (term,) = terms
    del model, selected, prediction, terms

    assert isinstance(term.factors, tuple)
    assert len(term.factors) == 2
    call = term.factors[0].call
    assert call is not None
    assert call.name == "C"
    assert term.factors[1].column == "x"
    columns = term.columns
    assert isinstance(columns, tuple)
    del term

    assert len(columns) == 2
    for column, level, values in zip(
        columns, ["b", "c"], [[0.0, 20.0, None], [10.0, 0.0, None]], strict=True
    ):
        assert isinstance(column.term, tuple)
        assert len(column.term) == 2
        term_call = column.term[0].call
        assert term_call is not None
        assert term_call.name == "C"
        assert column.term[1].column == "x"
        assert isinstance(column.contributions, tuple)
        assert len(column.contributions) == 2
        categorical_contribution, variable_contribution = column.contributions
        categorical = categorical_contribution.factor
        variable = variable_contribution.factor
        assert categorical.column is None
        assert categorical.literal is None
        call = categorical.call
        assert call is not None
        assert call.name == "C"
        assert call.parameters == {"omit": "a"}
        assert len(call.inputs) == 1
        assert call.inputs[0].column == "g"
        assert categorical_contribution.key == level
        assert variable.column == "x"
        assert variable.literal is None
        assert variable.call is None
        assert variable_contribution.key is None
        assert_series_equal(column.data, polars.Series(column.data.name, values))


@pytest.mark.parametrize(
    ("values", "dtype", "key_type"),
    [
        ([False, True], polars.Boolean, bool),
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
    data = polars.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, math.e, math.exp(2)]})
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
        polynomial = factor.call
        assert polynomial is not None
        assert polynomial.name == "poly"
        assert polynomial.parameters == {"degree": 2, "raw": True}
        assert type(polynomial.parameters["degree"]) is int
        assert type(polynomial.parameters["raw"]) is bool
        assert len(polynomial.inputs) == 1
        logarithm = polynomial.inputs[0].call
        assert logarithm is not None
        assert logarithm.name == "log"
        assert logarithm.parameters == {}
        assert len(logarithm.inputs) == 1
        assert logarithm.inputs[0].column == "x"
        assert logarithm.inputs[0].call is None
        assert column.data.to_list() == pytest.approx([0.0, 1.0, 2.0**basis])


@pytest.mark.parametrize(("literal", "expected"), [("2", 2), ("2.0", 2.0)])
def test_literal_transform_inputs_preserve_python_types(data, literal, expected):
    model = formula.parse(f"y ~ add(x, {literal}) - 1").fit(data)[0]
    column = model.columns(formula.Role.Independent)[0]
    contribution = column.contributions[0]
    factor = contribution.factor
    assert contribution.key is None
    call = factor.call
    assert call is not None
    assert call.name == "add"
    assert call.parameters == {}
    assert len(call.inputs) == 2
    variable, constant = call.inputs
    assert variable.column == "x"
    assert variable.literal is None
    assert variable.call is None
    assert constant.column is None
    assert constant.call is None
    assert type(constant.literal) is type(expected)
    assert constant.literal == expected
    assert column.data.to_list() == [4.0, 5.0, 6.0]


def test_provenance_properties_are_read_only_and_containers_are_copies(data):
    model = formula.parse("y ~ C(g, omit='a') - 1").fit(data)[0]
    term = model.terms(formula.Role.Independent)[0]
    column = term.columns[0]
    contribution = column.contributions[0]
    factor = contribution.factor
    call = factor.call
    assert call is not None
    for target, attribute, value in [
        (term, "factors", ()),
        (term, "columns", ()),
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
    parameters["omit"] = "changed"
    assert len(call.inputs) == 1
    assert call.inputs[0].column == "g"
    assert call.parameters == {"omit": "a"}

    exported = term.columns[0].data
    exported[0] = 99.0
    assert_series_equal(term.columns[0].data, column.data)


def test_zero_column_term_has_empty_python_tuple():
    training = polars.DataFrame({"y": [1.0, 2.0], "g": ["a", "a"]})
    model = formula.parse("y ~ C(g, omit='a')").fit(training)[0]
    intercept, categorical = model.terms(formula.Role.Independent)
    assert intercept.factors == ()
    assert categorical.columns == ()
    (factor,) = categorical.factors
    assert factor.call is not None
    assert factor.call.name == "C"


@pytest.mark.xfail(
    strict=True,
    reason="Polars DataFrame export loses zero-column height: formula#96",
)
def test_zero_column_frame_export_preserves_height():
    data = polars.DataFrame({"y": [1.0, 2.0], "g": ["a", "a"]})
    selected = (
        formula.parse("y ~ C(g, omit='a')")
        .fit(data)[0]
        .select_terms("C(g, omit='a')", formula.Role.Independent)
    )
    assert selected.frame().shape == (data.height, 0)


@pytest.mark.parametrize("accessor", ["columns", "terms"])
def test_columns_and_terms_reject_missing_or_invalid_roles(data, accessor):
    model = formula.parse("y ~ x").fit(data)[0]
    method = getattr(model, accessor)
    with pytest.raises(KeyError, match="FixedEffect"):
        method(formula.Role.FixedEffect)
    with pytest.raises(TypeError):
        method("Independent")


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
    (term,) = predicted.terms(role=formula.Role.FixedEffect)
    effect = term.fixed_effect
    assert effect is not None
    columns = predicted.columns(role=formula.Role.FixedEffect)
    del model, selected, predicted, term

    assert effect.encoding.levels.columns == ["firm", "year"]
    assert effect.codes.to_list() == [0, 1, 2]
    assert effect.intercept is True
    time, exposure = effect.loadings
    assert time.to_list() == pytest.approx([0.0, math.log(2), math.log(3)])
    assert exposure.to_list() == [0.5, 1.0, 1.5]

    assert len(columns) == 3
    for index, column in enumerate(columns):
        (contribution,) = column.contributions
        role = "group" if index == 0 else "loading"
        assert contribution.key == ("group" if index == 0 else ("loading", index))
        if index == 0:
            assert type(contribution.key) is str
        else:
            assert type(contribution.key) is tuple
            assert type(contribution.key[1]) is int
        assert column.data.dtype == (
            polars.UInt32 if role == "group" else polars.Float64
        )
        (term_factor,) = column.term
        term_call = term_factor.call
        assert term_call is not None

        call = contribution.factor.call
        assert call is not None
        assert call.parameters == {"intercept": True}
        assert type(call.parameters["intercept"]) is bool
        assert term_call.parameters == call.parameters
