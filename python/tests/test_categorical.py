import polars
import pytest
from polars.testing import assert_frame_equal

import formula

Independent = formula.Role.Independent


@pytest.fixture
def data() -> polars.DataFrame:
    return polars.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [5.0, 6.0, 7.0, 8.0],
            "a": ["q", "p", "r", "p"],
            "b": ["v", "u", "u", "v"],
            "flag": [True, False, True, False],
        }
    )


def independent(model: formula.FittedFormula) -> polars.DataFrame:
    return model.select(Independent).frame()


@pytest.mark.parametrize(
    "dtype",
    [polars.String, polars.Categorical, polars.Enum(["p", "q", "r"])],
)
def test_bare_string_categorical_and_enum_columns_encode_like_c(data, dtype):
    data = data.with_columns(polars.col("a").cast(dtype))
    bare = independent(formula.parse("y ~ a - 1").fit(data)[0])
    explicit = independent(formula.parse("y ~ C(a) - 1").fit(data)[0])

    assert bare.columns == ["a[p]", "a[q]", "a[r]"]
    assert explicit.columns == ["C(a)[p]", "C(a)[q]", "C(a)[r]"]
    assert_frame_equal(bare, explicit.rename(lambda name: name.replace("C(a)", "a")))
    assert bare["a[p]"].to_list() == [False, True, False, True]


def test_bare_categorical_columns_are_attributed_to_the_bare_factor(data):
    model = formula.parse("y ~ a").fit(data)[0]
    terms = model.terms(Independent)
    assert [factor.name for factor in terms[1].factors] == ["a"]
    columns = model.columns(Independent)[1:]
    for column, level in zip(columns, ["q", "r"], strict=True):
        (contribution,) = column.contributions
        assert contribution.factor.name == "a"
        assert contribution.factor.transform is None
        assert contribution.key == level


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("y ~ C(a) - 1", ["C(a)[p]", "C(a)[q]", "C(a)[r]"]),
        (
            "y ~ C(a, omit='q') - 1",
            ["C(a, omit='q')[p]", "C(a, omit='q')[r]"],
        ),
        ("y ~ C(n) - 1", ["C(n)[2]", "C(n)[9]", "C(n)[10]"]),
        ("y ~ C(flag) - 1", ["C(flag)[False]", "C(flag)[True]"]),
    ],
)
def test_categorical_levels_are_sorted(data, source, expected):
    data = data.with_columns(n=polars.Series([10, 9, 2, 9]))
    assert independent(formula.parse(source).fit(data)[0]).columns == expected


def test_categorical_levels_sort_categoricals_by_label_and_enums_by_category():
    values = polars.Series("a", ["b", "c", "a"])
    for dtype, expected in [
        (polars.Categorical, ["a[a]", "a[b]", "a[c]"]),
        (polars.Enum(["c", "a", "b"]), ["a[c]", "a[a]", "a[b]"]),
    ]:
        data = polars.DataFrame({"y": [1.0, 2.0, 3.0], "a": values.cast(dtype)})
        model = formula.parse("y ~ a - 1").fit(data)[0]
        assert independent(model).columns == expected


@pytest.mark.parametrize(
    ("source", "column"),
    [
        ("y ~ flag", "flag"),
        ("y ~ flag:x", "flag"),
        ("a ~ x", "a"),
        ("y ~ x + [a ~ x]", "a"),
    ],
)
def test_bare_booleans_and_dependent_strings_are_rejected(data, source, column):
    with pytest.raises(ValueError, match=f"column `{column}`: expected a primitive"):
        formula.parse(source).fit(data)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("y ~ a:x - 1", ["a[p]:x", "a[q]:x", "a[r]:x"]),
        ("y ~ x:a - 1", ["x:a[p]", "x:a[q]", "x:a[r]"]),
        (
            "y ~ a*x - 1",
            ["a[p]", "a[q]", "a[r]", "x", "a[T.q]:x", "a[T.r]:x"],
        ),
        (
            "y ~ a:b - 1",
            [f"a[{a}]:b[{b}]" for a in "pqr" for b in "uv"],
        ),
        (
            "y ~ a:C(b) - 1",
            [f"a[{a}]:C(b)[{b}]" for a in "pqr" for b in "uv"],
        ),
    ],
)
def test_bare_categoricals_interact(data, source, expected):
    assert independent(formula.parse(source).fit(data)[0]).columns == expected


def test_bare_categorical_interaction_values(data):
    frame = independent(formula.parse("y ~ a:x + a:b - 1").fit(data)[0])
    assert frame["a[p]:x"].to_list() == [0.0, 6.0, 0.0, 8.0]
    assert frame["a[p]:b[v]"].to_list() == [0.0, 0.0, 0.0, 1.0]


def test_fixed_effects_and_call_arguments_use_the_raw_column(data):
    # The fixed effect `a` spans both main effects of `a`, so they have no columns.
    with pytest.warns(formula.RankDeficiencyWarning, match=r"terms a, C\(a"):
        model = formula.parse("y ~ a + C(a, omit='p') | a").fit(data)[0]
    assert independent(model).columns == []
    assert [len(term.columns) for term in model.terms(formula.Role.Independent)] == [
        0,
        0,
    ]
    (fixed_effect,) = model.fixed_effects
    assert fixed_effect.codes.dtype == polars.UInt32
    assert fixed_effect.codes.to_list() == [0, 1, 2, 1]


def test_transform_reuses_fitted_levels_and_coding(data):
    model = formula.parse("y ~ a + a:x - 1").fit(data)[0]
    prediction = polars.DataFrame(
        {"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0], "a": ["r", "new", None]}
    )
    frame = independent(model.transform(prediction))
    assert frame.columns == independent(model).columns
    assert frame["a[r]"].to_list() == [True, None, None]
    assert frame["a[p]"].to_list() == [False, None, None]
    assert frame["a[r]:x"].to_list() == [1.0, None, None]

    with pytest.raises(ValueError, match="input `a`: expected str, received i64"):
        model.transform(prediction.with_columns(a=polars.Series([1, 2, 3])))

    numeric = formula.parse("y ~ a - 1").fit(polars.DataFrame({"y": [1.0], "a": [1.0]}))
    with pytest.raises(ValueError, match="input `a`: expected f64, received str"):
        numeric.transform(polars.DataFrame({"y": [1.0], "a": ["p"]}))


def test_lookup_and_selection_use_the_bare_spelling(data):
    fits = formula.parse("y ~ a + sw0(x)").fit(data)
    assert independent(fits["y ~ a"]).columns == ["Intercept", "a[T.q]", "a[T.r]"]

    selected = fits["y ~ a + x"].select_terms("a", Independent)
    assert selected.frame().columns == ["a[T.q]", "a[T.r]"]
    prediction = selected.transform(polars.DataFrame({"a": ["r", "p"]})).frame()
    assert prediction["a[T.r]"].to_list() == [True, False]
    with pytest.raises(KeyError, match=r"C\(a\)"):
        fits["y ~ a"].select_terms("C(a)", Independent)
