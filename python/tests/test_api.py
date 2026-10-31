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
    models = formula.parse(formulas).materialize(data)

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

    first = models[-2].frame(formula.Role.Independent)
    last = models[-1].frame(formula.Role.Independent)
    assert first is not None
    assert last is not None
    assert_frame_equal(first, data.select("x"))
    assert_frame_equal(last, data.select("z"))
    for index in (2, -3):
        with pytest.raises(IndexError, match="model index out of range"):
            models[index]


def test_model_owns_its_data_and_returns_independent_frames(data):
    model = formula.parse("y ~ x - 1").materialize(data)[0]
    assert isinstance(model, formula.Model)
    frame = model.frame(formula.Role.Independent)
    assert frame is not None
    frame.replace_column(0, polars.Series("x", [99.0, 99.0, 99.0]))

    original = model.frame(formula.Role.Independent)
    assert original is not None
    assert_frame_equal(original, data.select("x"))


@pytest.mark.parametrize("formulas", [42, b"y ~ x", ["y ~ x", 42]])
def test_parse_rejects_invalid_python_arguments(formulas):
    with pytest.raises(TypeError):
        formula.parse(formulas)


def test_rust_errors_become_python_exceptions(data):
    with pytest.raises(SyntaxError):
        formula.parse("y ~")

    with pytest.raises(ValueError, match="missing"):
        formula.parse("y ~ missing").materialize(data)
