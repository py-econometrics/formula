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
def test_parse_and_materialize(formulas, expected_columns, data):
    specification = formula.parse(formulas)
    assert isinstance(specification, formula.Specification)
    models = specification.materialize(data)

    assert len(models) == len(expected_columns)
    for index, column in enumerate(expected_columns):
        frame = models[index].frame(formula.Role.Independent)
        assert frame is not None
        assert_frame_equal(frame, data.select(column))


def test_role_frames_preserve_rows_and_nulls(data):
    model = formula.parse("y ~ x | firm").materialize(data)[0]

    dependent = model.frame(formula.Role.Dependent)
    independent = model.frame(formula.Role.Independent)
    assert dependent is not None
    assert independent is not None
    assert_frame_equal(dependent, data.select("y"))
    assert_frame_equal(independent, data.select("x"))
    fixed_effects = model.frame(formula.Role.FixedEffect)
    assert fixed_effects is not None
    assert fixed_effects.shape == (data.height, 1)
    assert fixed_effects.schema == {"firm": polars.UInt32}

    ordinary_model = formula.parse("y ~ x").materialize(data)[0]
    assert ordinary_model.frame(formula.Role.FixedEffect) is None


def test_model_indexing(data):
    models = formula.parse(["y ~ x - 1", "y ~ z - 1"]).materialize(data)

    by_formula = models["y ~ x - 1"]
    assert isinstance(by_formula, formula.Model)
    assert by_formula.formula == models[0].formula == "y ~ x - 1"
    assert models[" y~x -1 "].formula == by_formula.formula
    independent = by_formula.frame(formula.Role.Independent)
    assert independent is not None
    assert_frame_equal(independent, data.select("x"))

    first = models[-2].frame(formula.Role.Independent)
    last = models[-1].frame(formula.Role.Independent)
    assert first is not None
    assert last is not None
    assert_frame_equal(first, data.select("x"))
    assert_frame_equal(last, data.select("z"))
    for index in (2, -3):
        with pytest.raises(IndexError, match="model index out of range"):
            models[index]

    with pytest.raises(KeyError, match="y ~ missing"):
        models["y ~ missing"]
    with pytest.raises(SyntaxError, match="exactly one formula"):
        models["y ~ sw(x, z)"]


def test_expanded_models_can_be_looked_up_by_their_formula(data):
    models = formula.parse("y ~ sw(x, z)").materialize(data)

    for index in range(len(models)):
        model = models[index]
        found = models[model.formula]
        expected = model.frame(formula.Role.Independent)
        actual = found.frame(formula.Role.Independent)
        assert expected is not None
        assert actual is not None
        assert_frame_equal(actual, expected)


def test_model_owns_its_data_and_returns_independent_frames(data):
    model = formula.parse("y ~ x - 1").materialize(data)[0]
    assert isinstance(model, formula.Model)
    frame = model.frame(formula.Role.Independent)
    assert frame is not None
    frame.replace_column(0, polars.Series("x", [99.0, 99.0, 99.0]))

    original = model.frame(formula.Role.Independent)
    assert original is not None
    assert_frame_equal(original, data.select("x"))


def test_model_evaluate_reuses_fitted_categories(data):
    model = formula.parse("y ~ C(firm) - 1").materialize(data)[0]
    training = model.frame(formula.Role.Independent)
    assert training is not None

    # Only one training category occurs in the new data; retain both columns.
    evaluated = model.evaluate(data.slice(1, 1))
    independent = evaluated.frame(formula.Role.Independent)
    dependent = evaluated.frame(formula.Role.Dependent)
    assert independent is not None
    assert dependent is not None
    assert_frame_equal(independent, training.slice(1, 1))
    assert_frame_equal(dependent, data.select("y").slice(1, 1))

    original = model.frame(formula.Role.Independent)
    assert original is not None
    assert_frame_equal(original, training)


def test_models_evaluate_preserves_model_order_and_fitted_state(data):
    models = formula.parse(["y ~ C(firm) - 1", "y ~ x - 1"]).materialize(data)
    evaluated = models.evaluate(data.slice(1, 1))

    assert len(evaluated) == len(models) == 2
    for index in range(len(models)):
        training = models[index].frame(formula.Role.Independent)
        independent = evaluated[index].frame(formula.Role.Independent)
        assert training is not None
        assert independent is not None
        assert training.height == data.height
        assert_frame_equal(independent, training.slice(1, 1))


def test_select_accepts_role_sequences(data):
    models = formula.parse("y ~ x - 1").materialize(data)
    selected_model = models[0].select((formula.Role.Independent,))
    selected_models = models.select([formula.Role.Independent])
    assert isinstance(selected_model, formula.Model)
    assert isinstance(selected_models, formula.Models)

    new_data = data.select("x")
    for model in (selected_model, selected_models[0]):
        frame = model.evaluate(new_data).frame(formula.Role.Independent)
        assert frame is not None
        assert_frame_equal(frame, new_data)


def test_select_rejects_missing_roles_and_empty_selections(data):
    models = formula.parse("y ~ x").materialize(data)
    for target in (models, models[0]):
        with pytest.raises(KeyError, match="FixedEffect"):
            target.select([formula.Role.FixedEffect])
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
        formula.parse("y ~ missing").materialize(data)

    models = formula.parse("y ~ x").materialize(data)
    prediction = data.drop("x")
    with pytest.raises(ValueError, match="x"):
        models.evaluate(prediction)
    with pytest.raises(ValueError, match="x"):
        models[0].evaluate(prediction)
