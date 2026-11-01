"""Compare model-matrix construction for a stepwise specification.

Requires formula-py and pyfixest in the same Python environment, plus numpy,
pandas, and polars. Build formula-py in release mode before running this script.

Example:
    python benchmarks/compare_pyfixest_model_matrix.py --rows 100000 --repeats 7

The input Pandas and Polars frames are built before timing. Formula fits
all expanded models in one call, so shared term transformation is included in its
result. PyFixest constructs each expanded matrix separately, as its current
pipeline does. Both paths expose their dependent, independent, and fixed-effect
frames inside the timed region. No estimator fitting is timed.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas
import polars
from pyfixest.estimation.formula.model_matrix import create_model_matrix
from pyfixest.estimation.formula.parse import Formula as PyFixestFormula

import formula

SOURCE = "y ~ csw(x1, C(g)*x2, x3, x4) | csw(f1, f2:f3)"
ROLES = (formula.Role.Dependent, formula.Role.Independent, formula.Role.FixedEffect)


def make_data(rows: int) -> tuple[polars.DataFrame, pandas.DataFrame]:
    index = np.arange(rows)
    values = {
        "y": np.sin(index * 0.01),
        "x1": np.cos(index * 0.03),
        "x2": 1.0 + np.sin(index * 0.07),
        "x3": np.cos(index * 0.11),
        "x4": 1.0 + np.sin(index * 0.13),
        "g": np.take(["a", "b", "c", "d"], index % 4),
        "f1": np.char.add("firm_", (index % 1000).astype(str)),
        "f2": (2000 + (index // 1000) % 20).astype(np.int32),
        "f3": np.take(["north", "south", "east", "west"], (index // 7) % 4),
    }
    return polars.DataFrame(values), pandas.DataFrame(values)


def construct_formula(
    specification: formula.Specification, data: polars.DataFrame
) -> list[tuple[polars.DataFrame | None, ...]]:
    models = specification.fit(data)
    return [
        tuple(models[index].select(role).frame() for role in ROLES)
        for index in range(len(models))
    ]


def construct_pyfixest(
    formulas: list[PyFixestFormula], data: pandas.DataFrame, *, full_rank: bool
) -> list[tuple[pandas.DataFrame | None, ...]]:
    matrices = [
        create_model_matrix(
            parsed,
            data,
            ensure_full_rank=full_rank,
            drop_singletons=False,
        )
        for parsed in formulas
    ]
    for matrix in matrices:
        assert matrix.n_rows == len(data) and not matrix.dropped_row_index
    return [
        (matrix.dependent, matrix.independent, matrix.fixed_effects)
        for matrix in matrices
    ]


def check_same_output(
    formula_frames: list[tuple[polars.DataFrame | None, ...]],
    pyfixest_frames: list[tuple[pandas.DataFrame | None, ...]],
    rows: int,
) -> None:
    assert len(formula_frames) == len(pyfixest_frames)
    assert len(formula_frames) > 0
    for model_index, (left, right) in enumerate(
        zip(formula_frames, pyfixest_frames, strict=True)
    ):
        for role_index, (formula_frame, pyfixest_frame) in enumerate(
            zip(left, right, strict=True)
        ):
            assert formula_frame is not None and pyfixest_frame is not None
            assert formula_frame.shape == pyfixest_frame.shape, (
                model_index,
                role_index,
            )
            assert formula_frame.height == rows
            assert set(formula_frame.columns) == set(pyfixest_frame.columns), (
                model_index,
                role_index,
            )
            actual = formula_frame.to_numpy()
            expected = pyfixest_frame.loc[:, formula_frame.columns].to_numpy()
            if role_index < 2:
                np.testing.assert_allclose(actual, expected)
            else:
                # Fixed-effect codes can use different integers for the same groups.
                for column in range(actual.shape[1]):
                    np.testing.assert_array_equal(
                        pandas.factorize(actual[:, column])[0],
                        pandas.factorize(expected[:, column])[0],
                    )


def time_calls(calls: dict[str, Callable[[], Any]], repeats: int) -> dict[str, float]:
    for call in calls.values():
        call()  # warm up each path
    samples: dict[str, list[float]] = {name: [] for name in calls}
    names = list(calls)
    for repeat in range(repeats):
        for name in names[repeat % len(names) :] + names[: repeat % len(names)]:
            gc.collect()
            start = time.perf_counter_ns()
            result = calls[name]()
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
            del result
    return {name: statistics.median(times) for name, times in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--source", default=SOURCE)
    args = parser.parse_args()
    if args.rows < 4 or args.repeats < 1:
        parser.error("--rows must be at least 4 and --repeats must be positive")

    polars_data, pandas_data = make_data(args.rows)
    specification = formula.parse(args.source)
    parsed_pyfixest = PyFixestFormula.parse(args.source)
    formula_call = lambda: construct_formula(specification, polars_data)
    matched_call = lambda: construct_pyfixest(
        parsed_pyfixest, pandas_data, full_rank=False
    )
    default_call = lambda: construct_pyfixest(
        parsed_pyfixest, pandas_data, full_rank=True
    )

    check_same_output(formula_call(), matched_call(), args.rows)
    print(f"Formula: {args.source}")
    print(f"Rows: {args.rows:,}; expanded models: {len(parsed_pyfixest)}")
    print(f"Polars: {polars.__version__}; Pandas: {pandas.__version__}")
    print("Input frame creation and formula parsing excluded; all role frames exposed.")
    timings = time_calls(
        {
            "formula": formula_call,
            "pyfixest_matched": matched_call,
            "pyfixest_default": default_call,
        },
        args.repeats,
    )
    print(f"formula, shared fitting: {timings['formula']:.1f} ms")
    print(f"PyFixest, matched columns:       {timings['pyfixest_matched']:.1f} ms")
    print(f"PyFixest, default rank policy:   {timings['pyfixest_default']:.1f} ms")
    print("Matched columns use ensure_full_rank=False in PyFixest.")


if __name__ == "__main__":
    main()
