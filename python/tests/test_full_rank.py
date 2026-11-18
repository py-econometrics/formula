import re
import warnings

import polars
import pytest
from polars.testing import assert_frame_equal

import formula

Independent = formula.Role.Independent
Dependent = formula.Role.Dependent


@pytest.fixture
def data() -> polars.DataFrame:
    rows = [(a, b) for a in "rpq" for b in "vu"] * 2
    return polars.DataFrame(
        {
            "y": [float(index) for index in range(len(rows))],
            "x": [float((index * 7) % 5) + 1.0 for index in range(len(rows))],
            "a": [a for a, _ in rows],
            "b": [b for _, b in rows],
            "f": ["k", "l", "l"] * (len(rows) // 3),
        }
    )


def columns(model: formula.FittedFormula, role=Independent) -> list[str]:
    return model.select(role).frame().columns


def fit_quietly(source: str, data: polars.DataFrame, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return formula.parse(source).fit(data, **options)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("y ~ a", ["Intercept", "a[T.q]", "a[T.r]"]),
        ("y ~ a - 1", ["a[p]", "a[q]", "a[r]"]),
        ("y ~ C(a)", ["Intercept", "C(a)[T.q]", "C(a)[T.r]"]),
        (
            "y ~ a*b",
            [
                "Intercept",
                "a[T.q]",
                "a[T.r]",
                "b[T.v]",
                "a[T.q]:b[T.v]",
                "a[T.r]:b[T.v]",
            ],
        ),
        ("y ~ x + a:x - 1", ["x", "a[T.q]:x", "a[T.r]:x"]),
        ("y ~ a:x", ["Intercept", "a[p]:x", "a[q]:x", "a[r]:x"]),
        ("y ~ C(a) | f", ["C(a)[T.q]", "C(a)[T.r]"]),
        ("y ~ 0 + C(a) | f", ["C(a)[T.q]", "C(a)[T.r]"]),
        ("y ~ 0 + C(a) | f[[x]]", ["C(a)[p]", "C(a)[q]", "C(a)[r]"]),
    ],
)
def test_predictors_are_coded_to_full_rank(data, source, expected):
    assert columns(fit_quietly(source, data)[0]) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "y ~ C(a, omit='q')",
            ["Intercept", "C(a, omit='q')[p]", "C(a, omit='q')[r]"],
        ),
        (
            "y ~ C(a, omit='q') - 1",
            [
                "C(a, omit='q')[p]",
                "C(a, omit='q')[r]",
            ],
        ),
    ],
)
def test_explicit_omissions_always_apply(data, source, expected):
    assert columns(fit_quietly(source, data)[0]) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "y ~ a:b",
            [
                "Intercept",
                "b[T.v]",
                "a[T.q]:b[u]",
                "a[T.q]:b[v]",
                "a[T.r]:b[u]",
                "a[T.r]:b[v]",
            ],
        ),
        (
            "y ~ b + a:b",
            [
                "Intercept",
                "b[T.v]",
                "a[T.q]:b[u]",
                "a[T.q]:b[v]",
                "a[T.r]:b[u]",
                "a[T.r]:b[v]",
            ],
        ),
        ("y ~ C(a):C(f) | f", [f"C(a)[T.{a}]:C(f)[{f}]" for a in "qr" for f in "kl"]),
    ],
)
def test_terms_emit_only_the_column_groups_not_yet_spanned(data, source, expected):
    assert columns(fit_quietly(source, data)[0]) == expected


@pytest.mark.parametrize(
    ("source", "expected", "term"),
    [
        ("y ~ a + C(a)", ["Intercept", "a[T.q]", "a[T.r]"], "C(a)"),
        ("y ~ C(f) | f", [], "C(f)"),
        ("y ~ x | f[[x]]", ["Intercept"], "x"),
        ("y ~ C(f):x | f[x]", [], "C(f):x"),
    ],
)
def test_fully_spanned_terms_warn_and_have_no_columns(data, source, expected, term):
    match = re.escape(f"term {term} is fully spanned")
    with pytest.warns(formula.RankDeficiencyWarning, match=match) as record:
        fitted = formula.parse(source).fit(data)
    assert len(record) == 1
    assert issubclass(formula.RankDeficiencyWarning, UserWarning)
    assert "has no columns" in str(record[0].message)
    model = fitted[0]
    assert columns(model) == expected
    # The term is kept without columns; transformation does not warn again.
    absorbed = model.terms(Independent)[-1]
    assert absorbed.columns == ()
    selected = model.select_terms(term, Independent)
    # Zero-column frames lose their height on export (formula#96).
    assert selected.frame().width == 0
    assert len(selected.terms(Independent)) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert columns(fitted.transform(data)[0]) == expected
        assert selected.transform(data.head(2)).frame().width == 0


def test_each_deficient_formula_warns_once(data):
    with pytest.warns(formula.RankDeficiencyWarning) as record:
        # The child formula is shared by both models.
        formula.parse("y ~ sw(a + C(a), a + C(a) + x) + [x ~ b + C(b)]").fit(data)
    messages = sorted(str(warning.message) for warning in record)
    assert len(messages) == 3
    assert messages[0].startswith("`x ~ 1 + b + C(b)`")


def test_opting_out_still_honors_explicit_omissions(data):
    for source, expected in [
        ("y ~ a", ["Intercept", "a[p]", "a[q]", "a[r]"]),
        ("y ~ a:b", ["Intercept"] + [f"a[{a}]:b[{b}]" for a in "pqr" for b in "uv"]),
        (
            "y ~ a + C(a)",
            ["Intercept", "a[p]", "a[q]", "a[r]", "C(a)[p]", "C(a)[q]", "C(a)[r]"],
        ),
        (
            "y ~ C(a, omit='q') - 1",
            ["C(a, omit='q')[p]", "C(a, omit='q')[r]"],
        ),
    ]:
        model = fit_quietly(source, data, ensure_full_rank=False)[0]
        assert columns(model) == expected


def test_one_term_is_coded_per_expression(data):
    fits = fit_quietly("y ~ sw(C(a), C(b)) + C(a):C(b)", data)
    assert columns(fits[0])[-3:] == [
        "C(a)[p]:C(b)[T.v]",
        "C(a)[q]:C(b)[T.v]",
        "C(a)[r]:C(b)[T.v]",
    ]
    assert columns(fits[1])[-4:] == [
        "C(a)[T.q]:C(b)[u]",
        "C(a)[T.q]:C(b)[v]",
        "C(a)[T.r]:C(b)[u]",
        "C(a)[T.r]:C(b)[v]",
    ]
    transformed = fits.transform(data.head(4))
    for index in range(2):
        assert_frame_equal(
            transformed[index].select(Independent).frame(),
            fits[index].select(Independent).frame().head(4),
        )


def test_dependent_categoricals_keep_every_level(data):
    model = fit_quietly("C(a) ~ C(a)", data)[0]
    assert columns(model, Dependent) == ["C(a)[p]", "C(a)[q]", "C(a)[r]"]
    assert columns(model) == ["Intercept", "C(a)[T.q]", "C(a)[T.r]"]


def test_selection_and_contributions_follow_the_fitted_coding(data):
    model = fit_quietly("y ~ x + a", data)[0]
    selected = model.select_terms("a", Independent)
    assert selected.frame().columns == ["a[T.q]", "a[T.r]"]
    prediction = selected.transform(polars.DataFrame({"a": ["p", "r"]})).frame()
    assert prediction["a[T.r]"].to_list() == [False, True]
    keys = [
        [
            (contribution.factor.name, contribution.key)
            for contribution in column.contributions
        ]
        for column in model.columns(Independent)
    ]
    assert keys == [[], [("x", None)], [("a", "q")], [("a", "r")]]


def test_column_groups_omit_factors_from_their_contributions(data):
    model = fit_quietly("y ~ a:b", data)[0]
    keys = [
        [
            (contribution.factor.name, contribution.key)
            for contribution in column.contributions
        ]
        for column in model.columns(Independent)
    ]
    assert keys == [
        [],
        [("b", "v")],
        [("a", "q"), ("b", "u")],
        [("a", "q"), ("b", "v")],
        [("a", "r"), ("b", "u")],
        [("a", "r"), ("b", "v")],
    ]
    # Every column group still belongs to the term `a:b`.
    for column in model.columns(Independent)[1:]:
        assert [factor.name for factor in column.term] == ["a", "b"]
    selected = model.select_terms("b:a", Independent)
    assert selected.frame().columns == columns(model)[1:]
    assert_frame_equal(
        selected.transform(data.head(3)).frame(), selected.frame().head(3)
    )


def test_one_term_takes_a_different_number_of_groups_per_expression(data):
    fits = fit_quietly("y ~ sw0(C(a)) + C(a):C(b)", data)
    assert columns(fits[0])[1:] == [
        "C(b)[T.v]",
        "C(a)[T.q]:C(b)[u]",
        "C(a)[T.q]:C(b)[v]",
        "C(a)[T.r]:C(b)[u]",
        "C(a)[T.r]:C(b)[v]",
    ]
    assert columns(fits[1])[3:] == [
        "C(a)[p]:C(b)[T.v]",
        "C(a)[q]:C(b)[T.v]",
        "C(a)[r]:C(b)[T.v]",
    ]
    transformed = fits.transform(data.head(4))
    for index in range(2):
        assert_frame_equal(
            transformed[index].select(Independent).frame(),
            fits[index].select(Independent).frame().head(4),
        )
