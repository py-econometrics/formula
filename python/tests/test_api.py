import math
import operator

import polars
import pytest
from polars.testing import assert_frame_equal

import formula


@pytest.fixture
def data() -> polars.DataFrame:
    return polars.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [4.0, None, 6.0],
            "z": [7.0, 8.0, 9.0],
            "firm": ["a", "b", "a"],
        }
    )


@pytest.mark.parametrize(
    ("formulas", "expected_columns"),
    [
        ("y ~ x - 1", ["x"]),
        (["y ~ x - 1", "y ~ z - 1"], ["x", "z"]),
        (("y ~ x - 1", "y ~ z - 1"), ["x", "z"]),
        ([], []),
    ],
)
def test_parse_and_fit(formulas, expected_columns, data):
    specification = formula.parse(formulas)
    models = specification.fit(data)

    assert len(models) == len(expected_columns)
    for index, column in enumerate(expected_columns):
        frame = models[index].select(formula.Role.Independent).frame()
        assert_frame_equal(frame, data.select(column))


def test_unfitted_formulas_support_indexing_and_iteration():
    formulas = formula.parse(["y ~ sw(x, z) - 1", "y ~ x - 1"])

    assert len(formulas) == 2
    assert [str(entry) for entry in formulas] == ["y ~ x - 1", "y ~ z - 1"]
    assert str(formulas[0]) == str(formulas[-2]) == "y ~ x - 1"
    assert str(formulas[1]) == str(formulas[-1]) == "y ~ z - 1"
    assert str(formulas[" y~x -1 "]) == "y ~ x - 1"

    first = formulas[0]
    del formulas
    assert str(first) == "y ~ x - 1"


def test_unfitted_formula_lookup_uses_formula_identity():
    formulas = formula.parse("y ~ x + z - 1")
    assert str(formulas["y ~ z + x - 1"]) == "y ~ x + z - 1"


def test_unfitted_formulas_report_invalid_keys():
    formulas = formula.parse("y ~ sw(x, z) - 1")
    for index in (2, -3):
        with pytest.raises(IndexError, match="formula index out of range"):
            formulas[index]

    with pytest.raises(KeyError, match="missing"):
        formulas["y ~ missing"]
    with pytest.raises(SyntaxError, match="exactly one formula"):
        formulas["y ~ sw(x, z)"]
    with pytest.raises(SyntaxError):
        formulas["y ~"]


def test_empty_unfitted_formulas_are_an_empty_collection():
    formulas = formula.parse([])
    assert len(formulas) == 0
    assert not formulas
    assert list(iter(formulas)) == []

    for index in (0, -1):
        with pytest.raises(IndexError, match="formula index out of range"):
            formulas[index]
    with pytest.raises(KeyError):
        formulas["y ~ x"]


def test_role_frames_preserve_rows_and_nulls(data):
    model = formula.parse("y ~ x | firm").fit(data)[0]

    dependent = model.select(formula.Role.Dependent).frame()
    independent = model.select(formula.Role.Independent).frame()
    assert_frame_equal(dependent, data.select("y"))
    assert_frame_equal(independent, data.select("x"))
    fixed_effects = model.select(formula.Role.FixedEffect).frame()
    assert fixed_effects.shape == (data.height, 1)
    assert fixed_effects.schema == {
        "G(firm, intercept=true)[group]": polars.UInt32,
    }

    frame = model.frame()
    assert_frame_equal(frame, data.select("y", "x").hstack(fixed_effects))
    assert_frame_equal(
        model.select([formula.Role.Dependent, formula.Role.FixedEffect]).frame(),
        data.select("y").hstack(fixed_effects),
    )
    keep = frame.select(polars.all_horizontal(polars.all().is_not_null())).to_series()
    assert keep.to_list() == [True, False, True]
    assert_frame_equal(independent.filter(keep), polars.DataFrame({"x": [4.0, 6.0]}))


@pytest.mark.parametrize("height", [0, 1, 3])
def test_combined_frame_preserves_input_height_on_fitting_and_transformation(
    data, height
):
    specification = formula.parse("y ~ x - 1")
    input_data = data.head(height)
    fitted = specification.fit(data)[0]
    for model in (
        specification.fit(input_data)[0],
        fitted.transform(input_data),
    ):
        assert_frame_equal(model.frame(), input_data.select("y", "x"))


def test_is_valid_marks_rows_without_dropping_them():
    data = polars.DataFrame(
        {"y": [1.0, None, 3.0], "x": [1.0, 2.0, 0.0], "z": [1.0, 2.0, 3.0]}
    )
    models = formula.parse("y ~ sw(log(x), z)").fit(data)

    assert models[0].is_valid().to_list() == [True, False, False]
    assert models.is_valid().to_list() == [True, False, False]
    assert models[1].frame().filter(models[1].is_valid()).height == 2
    with pytest.raises(ValueError):
        models[0].children(formula.Role.Independent).is_valid()


def test_model_indexing(data):
    models = formula.parse(["y ~ x - 1", "y ~ z - 1"]).fit(data)

    by_formula = models["y ~ x - 1"]
    assert by_formula.formula == models[0].formula == "y ~ x - 1"
    assert models[" y~x -1 "].formula == by_formula.formula
    independent = by_formula.select(formula.Role.Independent).frame()
    assert_frame_equal(independent, data.select("x"))

    first = models[-2].select(formula.Role.Independent).frame()
    last = models[-1].select(formula.Role.Independent).frame()
    assert_frame_equal(first, data.select("x"))
    assert_frame_equal(last, data.select("z"))
    for index in (2, -3):
        with pytest.raises(IndexError, match="model index out of range"):
            models[index]

    with pytest.raises(KeyError, match="y ~ missing"):
        models["y ~ missing"]
    with pytest.raises(SyntaxError, match="exactly one formula"):
        models["y ~ sw(x, z)"]


@pytest.mark.parametrize(
    "source",
    [
        "y ~ sw(x, z)",
        "y ~ x - 1 + 1",
        "y ~ x - 1 + 1 + z",
        "y ~ x + [x ~ sw(z, y)]",
        "y ~ [x ~ z - 1] - 1",
        "y ~ x + [x ~ z + [z ~ x]] + [z ~ x | firm]",
    ],
)
def test_expanded_models_can_be_looked_up_by_their_formula(source, data):
    models = formula.parse(source).fit(data)
    assert len({models[index].formula for index in range(len(models))}) == len(models)

    for index in range(len(models)):
        model = models[index]
        found = models[model.formula]
        expected = model.select(formula.Role.Independent).frame()
        actual = found.select(formula.Role.Independent).frame()
        assert_frame_equal(actual, expected)


def test_model_owns_its_data_and_returns_independent_frames(data):
    model = formula.parse("y ~ x - 1").fit(data)[0]
    frame = model.frame()
    frame.replace_column(1, polars.Series("x", [99.0, 99.0, 99.0]))

    assert_frame_equal(model.frame(), data.select("y", "x"))
    original = model.select(formula.Role.Independent).frame()
    assert_frame_equal(original, data.select("x"))


@pytest.mark.parametrize("as_child", [False, True])
def test_model_transform_reuses_fitted_categories(data, as_child):
    source = "y ~ C(firm) - 1"
    if as_child:
        source = f"y ~ x + [{source}]"
    models = formula.parse(source).fit(data)
    parent = models[0]
    model = parent.children(formula.Role.Independent)[0] if as_child else parent
    del parent, models
    training = model.select(formula.Role.Independent).frame()

    # Only one training category occurs in the new data; retain both columns.
    # A child owns its fitted state and no longer requires the parent's x column.
    transformed = model.transform(data.select("y", "firm").slice(1, 1))
    independent = transformed.select(formula.Role.Independent).frame()
    dependent = transformed.select(formula.Role.Dependent).frame()
    assert_frame_equal(independent, training.slice(1, 1))
    assert_frame_equal(dependent, data.select("y").slice(1, 1))

    original = model.select(formula.Role.Independent).frame()
    assert_frame_equal(original, training)
    selected = model.select(formula.Role.Independent)
    assert_frame_equal(
        selected.transform(data.select("firm").slice(1, 1)).frame(),
        training.slice(1, 1),
    )


def test_children_preserve_role_attachment_order_and_navigation(data):
    models = formula.parse(
        "[z ~ x - 1] ~ x + [x ~ z + [z ~ x - 1] - 1] + [z ~ x - 1] - 1"
    ).fit(data)
    model = models[0]
    children = model.children(formula.Role.Independent)
    assert len(children) == 2
    assert_frame_equal(
        children[0].select(formula.Role.Dependent).frame(), data.select("x")
    )
    assert_frame_equal(
        children[1].select(formula.Role.Dependent).frame(), data.select("z")
    )
    grandchild = children[0].children(formula.Role.Independent)[0]
    assert_frame_equal(grandchild.frame(), data.select("z", "x"))
    empty = children[1].children(formula.Role.Independent)
    assert len(empty) == 0
    with pytest.raises(IndexError):
        empty[0]
    assert_frame_equal(children["z ~ x - 1"].frame(), children[-1].frame())
    assert_frame_equal(children[children[0].formula].frame(), children[0].frame())
    with pytest.raises(KeyError, match="missing"):
        children["z ~ missing"]
    with pytest.raises(IndexError):
        children[2]
    with pytest.raises(SyntaxError):
        children["z ~"]
    with pytest.raises(SyntaxError, match="exactly one formula"):
        children["z ~ sw(x, z)"]
    assert_frame_equal(model.frame(), data.select("x"))

    # A role with children but no local terms is still selectable and navigable.
    dependent = model.select(formula.Role.Dependent)
    assert dependent.frame().width == 0
    assert_frame_equal(
        dependent.children(formula.Role.Dependent)[0].frame(), data.select("z", "x")
    )
    for target, missing in [
        (model, formula.Role.FixedEffect),
        (dependent, formula.Role.Independent),
        (model.select(formula.Role.Independent), formula.Role.Dependent),
    ]:
        with pytest.raises(KeyError, match="does not contain role"):
            target.children(missing)
    # Transforming either entry point yields children backed by the new data.
    prediction = data.slice(1, 1)
    for transformed in (model.transform(prediction), models.transform(prediction)[0]):
        output = transformed.children(formula.Role.Independent)
        assert_frame_equal(
            output[0].select(formula.Role.Dependent).frame(), prediction.select("x")
        )
        assert_frame_equal(
            output[0].children(formula.Role.Independent)[0].frame(),
            prediction.select("z", "x"),
        )

    selected = children.select(formula.Role.Dependent)
    applied_children = children.transform(prediction)
    for index, column in enumerate(["x", "z"]):
        assert_frame_equal(selected[index].frame(), data.select(column))
        assert_frame_equal(
            applied_children[index].select(formula.Role.Dependent).frame(),
            prediction.select(column),
        )
    assert len(model.children(formula.Role.Independent)) == 2


def test_models_transform_preserves_model_order_and_fitted_state(data):
    models = formula.parse(["y ~ C(firm) - 1", "y ~ x - 1"]).fit(data)
    transformed = models.transform(data.slice(1, 1))

    assert len(transformed) == len(models) == 2
    for index in range(len(models)):
        training = models[index].select(formula.Role.Independent).frame()
        independent = transformed[index].select(formula.Role.Independent).frame()
        assert training.height == data.height
        assert_frame_equal(independent, training.slice(1, 1))


@pytest.mark.parametrize(
    "roles",
    [formula.Role.Independent, [formula.Role.Independent], (formula.Role.Independent,)],
)
def test_select_accepts_single_roles_and_sequences(data, roles):
    models = formula.parse("y ~ x - 1").fit(data)
    selected_model = models[0].select(roles)
    selected_models = models.select(roles)

    new_data = data.select("x")
    for model in (selected_model, selected_models[0]):
        frame = model.transform(new_data).frame()
        assert_frame_equal(frame, new_data)


def test_select_rejects_missing_roles_and_empty_selections(data):
    models = formula.parse("y ~ x").fit(data)
    for target in (models, models[0]):
        for roles in (formula.Role.FixedEffect, [formula.Role.FixedEffect]):
            with pytest.raises(KeyError, match="FixedEffect"):
                target.select(roles)
        with pytest.raises(ValueError, match="at least one role"):
            target.select([])


@pytest.mark.parametrize("formulas", [42, b"y ~ x", ["y ~ x", 42]])
def test_parse_rejects_invalid_python_arguments(formulas):
    with pytest.raises(TypeError):
        formula.parse(formulas)


def test_rust_errors_become_python_exceptions(data):
    with pytest.raises(SyntaxError):
        formula.parse("y ~")

    with pytest.raises(ValueError, match="missing"):
        formula.parse("y ~ missing").fit(data)

    models = formula.parse("y ~ x").fit(data)
    prediction = data.drop("x")
    with pytest.raises(ValueError, match="x"):
        models.transform(prediction)
    with pytest.raises(ValueError, match="x"):
        models[0].transform(prediction)


@pytest.mark.parametrize("keyword_role", [False, True])
def test_select_terms_returns_an_independent_fitted_formula(data, keyword_role):
    model = formula.parse("y ~ x + z").fit(data)[0]
    if keyword_role:
        selected = model.select_terms("z", role=formula.Role.Independent)
    else:
        selected = model.select_terms("z", formula.Role.Independent)
    del model
    assert_frame_equal(selected.frame(), data.select("z"))
    prediction = polars.DataFrame({"z": [10.0, 20.0]})
    assert_frame_equal(selected.transform(prediction).frame(), prediction)


@pytest.mark.parametrize(
    ("query", "role", "exception", "message"),
    [
        ("firm", formula.Role.FixedEffect, KeyError, "FixedEffect"),
        ("missing", formula.Role.Independent, KeyError, "missing"),
        ("x +", formula.Role.Independent, SyntaxError, None),
        (
            "sw(x, z)",
            formula.Role.Independent,
            SyntaxError,
            "operation is not supported",
        ),
    ],
)
def test_select_terms_translates_rust_errors(data, query, role, exception, message):
    model = formula.parse("y ~ x + z").fit(data)[0]
    with pytest.raises(exception, match=message):
        model.select_terms(query, role)


@pytest.mark.parametrize("roles", [(), ("Independent",)])
def test_select_terms_requires_a_valid_role(data, roles):
    model = formula.parse("y ~ x").fit(data)[0]
    with pytest.raises(TypeError):
        model.select_terms("x", *roles)


def test_varying_slope_selection_exports_typed_prediction_columns(data):
    training = data.with_columns(
        polars.Series("time", [1, 2, 3], dtype=polars.Int32),
        polars.Series("exposure", [0.5, 1.0, 1.5]),
    )
    model = formula.parse("y ~ x | firm[[time, exposure]]").fit(training)[0]
    selected = model.select_terms(
        "firm[[time, exposure]]", role=formula.Role.FixedEffect
    )
    del model

    predicted = selected.transform(
        polars.DataFrame(
            {
                "firm": ["b", "a", "unseen"],
                "time": polars.Series([4, 5, None], dtype=polars.Int32),
                "exposure": [2.0, 2.5, 3.0],
            }
        )
    ).frame()
    expected = polars.DataFrame(
        {
            "G(firm, time, exposure)[group]": polars.Series(
                [1, 0, None], dtype=polars.UInt32
            ),
            "G(firm, time, exposure)[loading:1]": polars.Series(
                [4.0, 5.0, None], dtype=polars.Float64
            ),
            "G(firm, time, exposure)[loading:2]": polars.Series(
                [2.0, 2.5, 3.0], dtype=polars.Float64
            ),
        }
    )
    assert_frame_equal(predicted, expected)


def test_varying_slope_fit_errors_are_value_errors(data):
    with pytest.raises(ValueError, match="expected a primitive numeric dtype"):
        formula.parse("y ~ x | firm[[firm]]").fit(data)
    with pytest.raises(ValueError, match="expanding transform"):
        formula.parse("y ~ x | firm[[C(firm)]]").fit(data)


def test_varying_slope_syntax_errors_are_syntax_errors():
    with pytest.raises(SyntaxError):
        formula.parse("y ~ x | firm[]")


@pytest.fixture(params=["group", "group[log(x), x]", "group[[log(x), x]]"])
def fixed_effect(request):
    data = polars.DataFrame(
        {"y": [1.0, 2.0, 3.0], "group": ["a", "b", "a"], "x": [1.0, 2.0, 3.0]}
    )
    model = formula.parse(f"y ~ 1 | {request.param}").fit(data)[0]
    (effect,) = model.fixed_effects
    assert effect.codes.to_list() == [0, 1, 0]
    assert effect.codes.dtype == polars.UInt32
    assert effect.intercept is ("[[" not in request.param)
    if request.param == "group":
        assert effect.loadings.width == 0
    else:
        time, exposure = effect.loadings
        assert time.dtype == exposure.dtype == polars.Float64
        assert time.to_list() == pytest.approx([0.0, math.log(2), math.log(3)])
        assert exposure.to_list() == [1.0, 2.0, 3.0]
    return effect


def test_fixed_effects_preserve_order_and_support_name_lookup(data):
    data = data.with_columns(polars.Series("year", [2020, 2021, 2020]))
    model = formula.parse("y ~ x | firm + year[x]").fit(data)[0]
    effects = model.fixed_effects
    assert len(effects) == 2
    assert [effect.encoding.levels.columns for effect in effects] == [
        ["firm"],
        ["year"],
    ]
    firm = model.fixed_effect("firm")
    assert firm.encoding.levels.columns == ["firm"]
    assert firm.loadings.width == 0
    year = model.fixed_effect("year[x]")
    assert year.encoding.levels.columns == ["year"]
    (loading,) = year.loadings
    assert loading.to_list() == data["x"].to_list()
    for key in ("year", "missing", "year[ x ]"):
        with pytest.raises(KeyError) as error:
            model.fixed_effect(key)
        assert error.value.args == (key,)
    with pytest.raises(TypeError):
        model.fixed_effect(1.5)  # ty: ignore[invalid-argument-type]


def test_model_without_fixed_effects_returns_an_empty_collection(data):
    model = formula.parse("y ~ x").fit(data)[0]
    assert model.fixed_effects == []
    with pytest.raises(KeyError):
        model.fixed_effect("firm")


def test_fixed_effect_encoding_exposes_codes_and_levels(fixed_effect):
    state = fixed_effect.encoding
    assert_frame_equal(
        state.codes,
        polars.DataFrame({"code": [0, 1]}, schema={"code": polars.UInt32}),
    )
    assert_frame_equal(state.levels, polars.DataFrame({"group": ["a", "b"]}))


@pytest.mark.parametrize("specification", ["group", "group[x]", "group[[x]]"])
def test_fixed_effect_data_tracks_prediction_rows(specification):
    training = polars.DataFrame({"y": [1.0, 2.0], "group": ["a", "b"], "x": [1.0, 2.0]})
    model = formula.parse(f"y ~ 1 | {specification}").fit(training)[0]
    prediction = polars.DataFrame(
        {"y": [3.0, 4.0, 5.0], "group": ["b", "new", "a"], "x": [10.0, 20.0, None]}
    )
    for data in (prediction, prediction.head(0)):
        (effect,) = model.transform(data).fixed_effects
        assert effect.codes.to_list() == [1, None, 0][: data.height]
        assert effect.codes.dtype == polars.UInt32
        assert effect.intercept is ("[[" not in specification)
        if specification == "group":
            assert effect.loadings.width == 0
        else:
            (loading,) = effect.loadings
            assert loading.dtype == polars.Float64
            assert loading.to_list() == data["x"].to_list()
        assert_frame_equal(
            effect.encoding.levels, polars.DataFrame({"group": ["a", "b"]})
        )


def test_transform_state_mapping_interface():
    data = polars.DataFrame({"y": [1.0, 2.0, 3.0], "group": ["a", "b", "a"]})
    model = formula.parse("y ~ 1 | group").fit(data)[0]
    (term,) = model.terms(formula.Role.FixedEffect)
    (factor,) = term.factors
    assert factor.call is not None
    (encoder,) = factor.call.inputs
    assert encoder.call is not None
    state = encoder.call.state
    assert state is not None
    assert len(state) == 2
    assert set(state) == {"codes", "levels"}
    assert isinstance(state["codes"], polars.DataFrame)
    assert isinstance(state["levels"], polars.DataFrame)
    with pytest.raises(KeyError) as error:
        _ = state["missing"]
    assert error.value.args == ("missing",)
    with pytest.raises(TypeError):
        operator.setitem(state, "codes", polars.DataFrame())  # ty: ignore[no-matching-overload]


def test_fixed_effect_encoding_values_are_independent_snapshots(fixed_effect):
    state = fixed_effect.encoding
    codes = state.codes
    levels = state.levels
    codes.replace_column(0, polars.Series("code", [99, 99], dtype=polars.UInt32))
    levels.replace_column(0, polars.Series("group", ["changed", "changed"]))

    expected_codes = polars.DataFrame({"code": [0, 1]}, schema={"code": polars.UInt32})
    expected_levels = polars.DataFrame({"group": ["a", "b"]})
    for snapshot in [state, fixed_effect.encoding]:
        assert_frame_equal(snapshot.codes, expected_codes)
        assert_frame_equal(snapshot.levels, expected_levels)


def test_stateless_transform_has_an_empty_state():
    data = polars.DataFrame({"y": [1.0, 2.0], "x": [2.0, 3.0]})
    model = formula.parse("y ~ log(x) - 1").fit(data)[0]
    (term,) = model.terms(formula.Role.Independent)
    assert term.fixed_effect is None
    (factor,) = term.factors
    call = factor.call
    assert call is not None
    state = call.state
    assert state is not None
    assert len(state) == 0
    assert list(state) == []
    with pytest.raises(KeyError):
        _ = state["codes"]
