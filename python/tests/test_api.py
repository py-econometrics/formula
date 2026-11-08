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
    assert isinstance(specification, formula.Formulas)
    models = specification.fit(data)

    assert len(models) == len(expected_columns)
    for index, column in enumerate(expected_columns):
        frame = models[index].select(formula.Role.Independent).frame()
        assert_frame_equal(frame, data.select(column))


def test_role_frames_preserve_rows_and_nulls(data):
    model = formula.parse("y ~ x | firm").fit(data)[0]

    dependent = model.select(formula.Role.Dependent).frame()
    independent = model.select(formula.Role.Independent).frame()
    assert_frame_equal(dependent, data.select("y"))
    assert_frame_equal(independent, data.select("x"))
    fixed_effects = model.select(formula.Role.FixedEffect).frame()
    assert fixed_effects.shape == (data.height, 1)
    assert fixed_effects.schema == {"firm": polars.UInt32}

    frame = model.frame()
    assert_frame_equal(frame, data.select("y", "x").hstack(fixed_effects))
    assert_frame_equal(
        model.select([formula.Role.Dependent, formula.Role.FixedEffect]).frame(),
        frame.select("y", "firm"),
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


def test_model_indexing(data):
    models = formula.parse(["y ~ x - 1", "y ~ z - 1"]).fit(data)

    by_formula = models["y ~ x - 1"]
    assert isinstance(by_formula, formula.FittedFormula)
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
        "y ~ x + [x ~ sw(x, z)]",
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
    assert isinstance(model, formula.FittedFormula)
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
    assert isinstance(children, formula.FittedFormulas)
    assert len(children) == 2
    assert all(isinstance(child, formula.FittedFormula) for child in children)
    assert_frame_equal(
        children[0].select(formula.Role.Dependent).frame(), data.select("x")
    )
    assert_frame_equal(
        children[1].select(formula.Role.Dependent).frame(), data.select("z")
    )
    grandchild = children[0].children(formula.Role.Independent)[0]
    assert_frame_equal(grandchild.frame(), data.select("z", "x"))
    empty = children[1].children(formula.Role.Independent)
    assert isinstance(empty, formula.FittedFormulas)
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
    assert isinstance(selected_model, formula.FittedFormula)
    assert isinstance(selected_models, formula.FittedFormulas)

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
