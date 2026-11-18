"""i() designs, checked against explicit indicator calculations."""

import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import formula


@pytest.fixture
def data():
    return pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "a": ["c", "a", "b", "a", "b", "c"],
            "b": ["v", "u", "u", "v", "v", "u"],
            "x": [2.0, 3.0, 5.0, 7.0, 11.0, 13.0],
            "t": [1, -1, 0, -1, 0, 1],
            "flag": [True, False, True, False, True, False],
        }
    )


def design(source, data):
    return formula.parse(source).fit(data)[0].select(formula.Role.Independent)


@pytest.mark.parametrize(
    ("source", "levels"),
    [
        ("y ~ i(a) - 1", ["a", "b", "c"]),
        ("y ~ 0 + i(a)", ["a", "b", "c"]),
        ("y ~ i(a, ref='b') - 1", ["a", "c"]),
        ("y ~ i(a, ref = 'b') | b", ["a", "c"]),
    ],
)
def test_indicators_and_references(data, source, levels):
    expected = data.select(
        *[(pl.col("a") == level).alias(f"a::{level}") for level in levels]
    )
    assert_frame_equal(design(source, data).frame(), expected)


@pytest.mark.parametrize(
    ("source", "columns"),
    [
        # As for C(), structural coding drops the first level next to an intercept
        ("y ~ i(a)", ["Intercept", "a::b", "a::c"]),
        # or a fixed effect that spans the constant,
        ("y ~ i(a) | b", ["a::b", "a::c"]),
        # and absorbs i() entirely when the fixed effect is its own variable.
        ("y ~ i(a) | a", []),
        ("y ~ i(a, x) | a[[x]]", ["Intercept"]),
        # A slope keeps every level unless the slope itself is in the model.
        ("y ~ i(a, x)", ["Intercept", "a::a:x", "a::b:x", "a::c:x"]),
        ("y ~ x + i(a, x)", ["Intercept", "x", "a::b:x", "a::c:x"]),
        # Two factors follow `a:b`.
        (
            "y ~ i(a, b)",
            ["Intercept", "b::v", "a::b:b::u", "a::b:b::v", "a::c:b::u", "a::c:b::v"],
        ),
        # A bare variable and i() share a level space.
        ("y ~ a + i(a)", ["Intercept", "a[T.b]", "a[T.c]"]),
    ],
)
def test_structural_coding(data, source, columns):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", formula.RankDeficiencyWarning)
        frame = design(source, data).frame()
    assert frame.columns == columns


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.Float32, pl.Float64])
@pytest.mark.parametrize("reference", ["-1", "-1.0"])
def test_event_time_numeric_references(data, dtype, reference):
    data = data.with_columns(pl.col("t").cast(dtype))
    frame = design(f"y ~ i(t, flag, ref={reference}) - 1 | a + b", data).frame()
    suffix = ".0" if dtype in (pl.Float32, pl.Float64) else ""
    expected = data.select(
        *[
            ((pl.col("t") == level).cast(pl.Float64) * pl.col("flag")).alias(
                f"t::{level}{suffix}:flag"
            )
            for level in (0, 1)
        ]
    )
    assert_frame_equal(frame, expected)


@pytest.mark.parametrize("second", ["x", "flag", "mul(x, 2)"])
@pytest.mark.parametrize("reference", [None, "a"])
def test_numeric_and_boolean_slopes(data, second, reference):
    suffix = "" if reference is None else f", ref='{reference}'"
    source = f"y ~ 0 + i(a, {second}{suffix})"
    values = pl.col("x") * 2 if second == "mul(x, 2)" else pl.col(second)
    expected = data.select(
        *[
            ((pl.col("a") == level).cast(pl.Float64) * values).alias(
                f"a::{level}:{second}"
            )
            for level in ["a", "b", "c"]
            if level != reference
        ]
    )
    assert_frame_equal(design(source, data).frame(), expected)


@pytest.mark.parametrize(
    ("options", "left", "right"),
    [
        ("", ["a", "b", "c"], ["u", "v"]),
        (", ref='a'", ["b", "c"], ["u", "v"]),
        (", ref2='v'", ["a", "b", "c"], ["u"]),
        (", ref='c', ref2='u'", ["a", "b"], ["v"]),
    ],
)
def test_factor_interactions(data, options, left, right):
    frame = design(f"y ~ 0 + i(a, b{options})", data).frame()
    expected = data.select(
        *[
            ((pl.col("a") == a) & (pl.col("b") == b))
            .cast(pl.Float64)
            .alias(f"a::{a}:b::{b}")
            for a in left
            for b in right
        ]
    )
    assert_frame_equal(frame, expected)


def test_ref2_forces_numeric_second_variable_to_be_categorical(data):
    actual = design("y ~ 0 + i(a, t, ref='a', ref2=-1)", data).frame()
    expected = data.select(
        *[
            ((pl.col("a") == a) & (pl.col("t") == t))
            .cast(pl.Float64)
            .alias(f"a::{a}:t::{t}")
            for a in ["b", "c"]
            for t in [0, 1]
        ]
    )
    assert_frame_equal(actual, expected)


def test_binning_references_and_retained_mapping(data):
    source = "y ~ 0 + i(a, b, bin={'ab': ['a', 'b']}, bin2={'uv': 'u'}, ref='c')"
    model = design(source, data)
    expected = data.select(
        *[
            (pl.col("a").is_in(["a", "b"]) & (pl.col("b") == old))
            .cast(pl.Float64)
            .alias(f"a::ab:b::{new}")
            for old, new in [("u", "uv"), ("v", "v")]
        ]
    )
    assert_frame_equal(model.frame(), expected)
    # Neither the reference nor every old bin member needs to be in new data.
    assert_frame_equal(model.transform(data.slice(2, 1)).frame(), expected.slice(2, 1))
    assert_frame_equal(model.transform(data.head(0)).frame(), expected.head(0))


def test_bin_labels_may_differ_in_type_from_the_levels(data):
    frame = design("y ~ 0 + i(t, bin={'pre': [-1, 0]}, ref='pre')", data).frame()
    assert_frame_equal(frame, data.select((pl.col("t") == 1).alias("t::1")))


@pytest.mark.parametrize(
    "dtype", [pl.String, pl.Categorical, pl.Enum(["unused", "c", "b", "a"])]
)
def test_native_category_levels_and_prediction(data, dtype):
    data = data.with_columns(pl.col("a").cast(dtype))
    levels = ["c", "b", "a"] if isinstance(dtype, pl.Enum) else ["a", "b", "c"]
    model = design("y ~ i(a) - 1", data)
    expected = data.select(
        *[(pl.col("a") == level).alias(f"a::{level}") for level in levels]
    )
    assert_frame_equal(model.frame(), expected)
    assert_frame_equal(model.transform(data.tail(1)).frame(), expected.tail(1))


@pytest.mark.parametrize(
    "source", ["y ~ i(a) - 1", "y ~ i(a, x) - 1", "y ~ i(a, b) - 1"]
)
def test_missing_and_unknown_levels_propagate_to_every_indicator(data, source):
    model = design(source, data)
    prediction = data.head(3).with_columns(
        pl.Series("a", [None, "unseen", "a"]),
        pl.Series("b", ["u", "u", None]),
        pl.Series("x", [2.0, 3.0, None]),
    )
    frame = model.transform(prediction).frame()
    assert all(value is None for row in frame.head(2).rows() for value in row)
    if source == "y ~ i(a) - 1":
        assert frame.row(2) == (True, False, False)
    else:
        assert all(value is None for value in frame.row(2))


def test_levels_are_learned_before_missing_response_rows_are_removed(data):
    data = data.with_columns(
        pl.when(pl.col("a") == "a").then(None).otherwise(pl.col("y")).alias("y")
    )
    model = formula.parse("y ~ i(a, ref='a') - 1").fit(data)[0]
    assert model.frame().height == data.height
    assert model.select(formula.Role.Independent).frame().columns == ["a::b", "a::c"]


@pytest.mark.parametrize(
    ("source", "columns"),
    [
        ("y ~ i(a):x - 1", ["a::a:x", "a::b:x", "a::c:x"]),
        ("y ~ x + i(a, ref='a'):x - 1", ["x", "a::b:x", "a::c:x"]),
        ("y ~ i(a, ref='a')*x - 1", ["a::b", "a::c", "x", "a::b:x", "a::c:x"]),
        ("y ~ 0 + i(a) + i(b)", ["a::a", "a::b", "a::c", "b::v"]),
    ],
)
def test_surrounding_formula_operators(data, source, columns):
    model = design(source, data)
    assert model.frame().columns == columns
    assert_frame_equal(model.transform(data).frame(), model.frame())


def test_stepwise_alternatives_and_formula_lookup_keep_distinct_options(data):
    models = formula.parse(
        [
            "y ~ i(a) - 1",
            "y ~ 0 + i(a, ref='a')",
            "y ~ sw(i(a, x), i(a, x, ref='a')) | b",
        ]
    ).fit(data)
    assert len(models) == 4
    for model in models:
        assert_frame_equal(models[model.formula].frame(), model.frame())
        assert_frame_equal(model.transform(data).frame(), model.frame())
    widths = [model.select(formula.Role.Independent).frame().width for model in models]
    assert widths == [3, 2, 3, 2]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("y ~ i(a, ref='absent')", "`ref` value 'absent' is not among the fitted"),
        (
            "y ~ i(a, ref=1)",
            "`ref` dtype Int64 is incompatible with input dtype String",
        ),
        ("y ~ i(a, b, ref2='a')", "`ref2` value 'a' is not among the fitted"),
        ("y ~ i(a, bin={'new': ['missing']})", "`bin` value 'missing' is not among"),
        ("y ~ i(a, poly(x))", "expanding"),
        ("y ~ i(a, b, x)", "between 1 and 2"),
        ("y ~ i()", "between 1 and 2"),
    ],
)
def test_invalid_i_inputs(data, source, message):
    with pytest.raises(ValueError, match=message):
        formula.parse(source).fit(data)


@pytest.mark.parametrize(
    "source",
    [
        "y ~ i(a, keep='a')",
        "y ~ i(a, ref='a', ref='b')",
        "y ~ i(a, ref={'x': ['a']})",
        "y ~ i(a, bin='a')",
        "y ~ i(a, bin={})",
        "y ~ i(a, bin={'new': []})",
        "y ~ i(a, bin={'new': ['a']",
        "y ~ x | i(a)",
        "y ~ i(a, ref2='a')",
        "y ~ i(a, bin={'new': ['a'], 'other': ['a']})",
        "y ~ i(a, bin={'new': 'a', 'new': 'b'})",
    ],
)
def test_unsupported_or_ambiguous_i_syntax_is_rejected(source):
    with pytest.raises(SyntaxError):
        formula.parse(source)


def test_formulas_render_to_equivalent_syntax(data):
    source = "y ~ i(a, b, ref2=  'v', bin2={'uv':'u'}, bin={ 'ab' : ['a','b'] })"
    models = formula.parse(source).fit(data)
    rendered = str(models[0].formula)
    assert rendered == (
        "y ~ 1 + i(a, b, ref2='v', bin={'ab': ['a', 'b']}, bin2={'uv': ['u']})"
    )
    assert str(formula.parse(rendered)[0]) == rendered
    assert_frame_equal(models[rendered].frame(), models[0].frame())


def test_nan_is_missing_and_large_integer_references_are_exact(data):
    data = data.with_columns(pl.Series("t", [float("nan"), -1.0, 0.0, 1.0, 0.0, 1.0]))
    frame = design("y ~ 0 + i(t, ref=-1)", data).frame()
    assert frame.columns == ["t::0.0", "t::1.0"]
    assert frame.row(0) == (None, None)
    data = data.with_columns(pl.Series("t", [2**53, 2**53 + 1] * 3))
    frame = design(f"y ~ 0 + i(t, ref={2**53})", data).frame()
    assert frame.to_dict(as_series=False) == {f"t::{2**53 + 1}": [False, True] * 3}


def test_close_float_levels_are_not_merged_by_display_rounding(data):
    data = data.with_columns(pl.Series("t", [0.12345671, 0.12345672] * 3))
    frame = design("y ~ 0 + i(t, ref=0.12345671)", data).frame()
    assert frame.to_dict(as_series=False) == {"t::0.12345672": [False, True] * 3}
    with pytest.raises(ValueError, match="`ref` value 0.12345673 is not among"):
        design("y ~ i(t, ref=0.12345673)", data)


@pytest.mark.parametrize("source", ["y ~ i(a)", "y ~ i(a, x)", "y ~ i(a, b)"])
def test_empty_predictions_and_empty_fits_keep_row_counts(data, source):
    model = design(source, data)
    assert_frame_equal(model.transform(data.head(0)).frame(), model.frame().head(0))
    assert design(source, data.head(0)).frame().height == 0


def test_all_missing_fit_keeps_height_and_has_no_indicators(data):
    for source in ["y ~ i(a)", "y ~ i(a, x)", "y ~ i(a, b)"]:
        missing = data.with_columns(pl.lit(None, dtype=pl.String).alias("a"))
        frame = design(source, missing).frame()
        assert frame.shape == (data.height, 1)
        assert frame.columns == ["Intercept"]


def test_many_levels_retain_schema_and_mask_unknown_values():
    data = pl.DataFrame({"y": range(20), "t": range(20)})
    model = design("y ~ i(t) - 1", data)
    assert model.frame().columns == [f"t::{level}" for level in range(20)]
    prediction = model.transform(pl.DataFrame({"t": [None, 20, 0]})).frame()
    assert prediction.row(0) == (None,) * 20
    assert prediction.row(1) == (None,) * 20
    assert prediction.row(2) == (True,) + (False,) * 19


def test_prediction_rejects_changed_dtype(data):
    model = design("y ~ i(a, x)", data)
    with pytest.raises(ValueError, match="expected.*f64.*received.*i64"):
        model.transform(data.with_columns(pl.col("x").cast(pl.Int64)))


def test_provenance_and_parameters(data):
    model = formula.parse("y ~ 0 + i(a, b, ref='a', ref2='v')").fit(data)[0]
    columns = model.columns(formula.Role.Independent)
    assert [column.data.name for column in columns] == ["a::b:b::u", "a::c:b::u"]
    for column, level in zip(columns, ["b", "c"], strict=True):
        (contribution,) = column.contributions
        assert contribution.key == (level, "u")
        call = contribution.factor.transform
        assert call is not None
        assert call.name == "i"
        assert call.parameters == {"ref": "a", "ref2": "v"}
        assert [factor.name for factor in call.inputs] == ["a", "b"]


@pytest.mark.parametrize("reference", ["-1", "-1.0"])
def test_float_event_time_with_missing_values(data, reference):
    # Never-treated units often have no event time, which makes the column float.
    data = data.with_columns(
        pl.Series("t", [1.0, -1.0, None, -1.0, 0.0, 1.0]), pl.lit(1.0).alias("one")
    )
    frame = design(f"y ~ i(t, one, ref={reference}) | a", data).frame()
    assert frame.columns == ["t::0.0:one", "t::1.0:one"]
    assert frame["t::1.0:one"].to_list() == [1.0, 0.0, None, 0.0, 0.0, 1.0]


def test_inexact_references_name_no_level(data):
    with pytest.raises(ValueError, match="`ref` value 0.5 is not among the fitted"):
        design("y ~ i(t, ref=0.5)", data)


@pytest.mark.parametrize("ensure_full_rank", [True, False])
def test_repeated_fixest_names_are_rejected_clearly(data, ensure_full_rank):
    source = "y ~ 0 + i(a, bin={'ab': ['a', 'b']}) + i(a, bin={'x': ['a']})"
    with pytest.raises(ValueError, match="produces the column `a::.*` more than once"):
        formula.parse(source).fit(data, ensure_full_rank=ensure_full_rank)
