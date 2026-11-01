# formula

`formula` is  [`Polars`](https://pola.rs)-powerd implementation of Wilkinson formulas in Rust with a Python API. It parses formula strings and constructs data frames for regression
analysis. It is primarily designed for [`PyFixest`](https://github.com/py-econometrics/pyfixest).

> [!NOTE]
> `formula` is in early development and its scope, syntax, API is subject to change. It is
not ready for production use.

## Python API

```python
import polars

import formula

data = polars.DataFrame(
    {
        "y": [1.0, 2.0, 3.0],
        "x": [4.0, 5.0, 6.0],
        "z": [7.0, 8.0, 9.0],
        "firm": ["a", "b", "a"],
    }
)

specification = formula.parse("y ~ sw(x, z) | firm")  # Parse into a specification
models = specification.fit(data)  # Fit specification given input data
model = models[0]  # Integer indexing follows expansion order
same_model = models[model.formula]  # Look up an expanded model by its formula

# Access the complete frame or select specific roles
frame = model.frame()
dependent = model.select(formula.Role.Dependent).frame()
independent = model.select(formula.Role.Independent).frame()
fixed_effects = model.select(formula.Role.FixedEffect).frame()

# Select roles before transforming data that has no dependent column.
predictors = models.select([formula.Role.Independent, formula.Role.FixedEffect])
new_data = polars.DataFrame({"x": [8.0], "z": [9.0], "firm": ["a"]})
transformed = predictors.transform(new_data)
```

`parse` accepts one formula string or a sequence of strings and returns a
`Specification`. Its `fit` method returns one model per expanded formula, containing
the computed columns and learned transformation state. `transform` computes columns
for new data using that learned state. Fitting does not estimate regression coefficients.
`frame()` returns a Polars `DataFrame` containing all selected roles. `select` and
`transform` return new models and retain fitted transform state. Selecting an absent
role raises `KeyError`.

## Rust API

```rust
use formula::{Role, parse};
use polars::df;
use polars::prelude::{DataFrame, Float64Chunked, NamedFrom, PolarsResult, Series};

fn sample_data(n: usize) -> PolarsResult<DataFrame> {
    df!(
        "y" => Float64Chunked::rand_standard_normal("y".into(), n),
        "x1" => Float64Chunked::rand_uniform("x1".into(), n, 1.0, 10.0),
        "x2" => Float64Chunked::rand_uniform("x2".into(), n, 1.0, 10.0),
        "category" => Series::new("category".into(), ["a", "b"])
            .sample_n(n, true, None, Some(1))?,
        "firm" => Series::new("firm".into(), ["firm_a", "firm_b", "firm_c"])
            .sample_n(n, true, None, Some(2))?,
        "year" => Series::new("year".into(), [2020, 2021, 2022])
            .sample_n(n, true, None, Some(3))?,
    )
}

fn main() {
    let formula =
        "y ~ add(x1, x2) + sw(log(x1)*sqrt(x2), C(category, reference='a')) | firm + year";

    // Stores all specifications associated with a formula
    let specification = parse(formula).expect("valid formula");

    // Fitting a specification on data creates a collection of `Model`s
    // Transformations and data are shared across models, so no work is duplicated
    let data = sample_data(1_000_000).expect("valid sample data");
    let models = specification
        .fit(&data)
        .expect("successful fitting");

    for model in &models {
        println!("Model {}", model.formula());
        println!("Fitted frame:\n{}", model.frame().expect("valid frame"));
        let independent = model
            .select(&[Role::Independent])
            .expect("independent role exists");
        println!("Independent design:\n{}", independent.frame().expect("valid frame"));
    }
}
```

## Formula syntax

A formula has a response on the left of `~` and predictors on the right. The intercept
is included by default; use `-1` to omit it.

```text
y ~ x + z       # response y, predictors x and z, with an intercept
y ~ x:z         # interaction between x and z
y ~ x*z         # equivalent to x + z + x:z
y ~ x/z         # equivalent to x + x:z
y ~ log(x) - 1  # transform x and omit the intercept
```

Use backticks `` ` `` to escape column names containing spaces, punctuation or special
characters, for example ``y ~ `household income` ``.

### Fixed effects

Use `|` to specify fixed effects separately from ordinary predictors:

```text
y ~ x | firm + year      # Separate firm and year fixed effects
y ~ x | firm:year        # One fixed effect for observed firm-year combinations
y ~ 1 + x | firm         # Fixed effects suppress even an explicitly requested intercept
```

Each grouping term produces one integer-coded column, available through
`model.select(formula.Role.FixedEffect).frame()` in Python or
`model.select(&[Role::FixedEffect])?.frame()?` in Rust. Column names preserve the input column
names and factor order: `firm:year` and `year:firm` retain their respective order.

The pipe accepts column references, parentheses, formula operators, and stepwise
expansion. Transform calls such as `log(x)`, `C(firm)`, and `poly(x)` are rejected
after the pipe.

### Transformations

`formula` supports the following transformations, applied to data passed to `fit()`
or `transform()`:

| Function                                                                     | Description                                                                                                   |
|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `log(x)`<br>`sqrt(x)`<br>`exp(x)`                                            | Apply the natural logarithm, square root, or exponential function to a numeric column or scalar argument.     |
| `add(x, y, ...)`<br>`sub(x, y, ...)`<br>`mul(x, y, ...)`<br>`div(x, y, ...)` | Add, subtract, multiply, or divide two or more numeric columns or scalar arguments.                           |
| `C(group, reference=...)`                                                    | Create categorical indicator columns for `group`; the optional `reference` argument sets the reference level. |
| `poly(x, degree=1, raw=false)`                                               | Expand a numeric input into orthonormal polynomial columns, or raw powers with `raw=true`.                    |

Numeric arguments are supported inside transformation calls:

```text
y ~ mul(x, 2)             # multiply x by 2
y ~ div(x, 2.0)           # divide x by 2.0
y ~ mul(add(x, -2), 3)    # combine nested calls and numeric arguments
```

Integers, decimals, and scientific notation are supported, including negative
values. Numeric arguments are broadcast over the input rows. To reference a
column with a numeric name, quote it: ``mul(x, `2`)``.

### Multiple estimation

`formula` supports the [`fixest`](https://lrberge.github.io/fixest/index.html)-style
[multiple estimation syntax](https://lrberge.github.io/fixest/reference/index.html#multiple-estimations)
which enables generating
multiple model specifications from a single formula. For example, the formula
`y ~ sw(x1, x2)` expands to the formulas `y ~ x1` and `y ~ x2`.

| Operator             | Models generated                                                                        |
|----------------------|-----------------------------------------------------------------------------------------|
| `sw(x, z, w, ...)`   | One model per argument: `x`, `z`, `w`, and so on.                                       |
| `sw0(x, z, w, ...)`  | An empty expression, then one model per argument: `1`, `x`, `z`, `w`, and so on.        |
| `csw(x, z, w, ...)`  | Cumulative models: `x`, then `x + z`, then `x + z + w`, and so on.                      |
| `csw0(x, z, w, ...)` | An empty expression, then cumulative models: `1`, `x`, `x + z`, `x + z + w`, and so on. |

## Development

The Rust workspace contains the core crate in `crates/formula` and the Python
binding in `crates/formula-py`. The Python package lives in `python/formula`.

Use [Pixi](https://pixi.sh) to build the Python package and run the Rust tests:

```sh
pixi run --locked develop  # Build and install the extension in the Pixi environment
pixi run --locked build    # Build a release wheel in dist/
pixi run --locked test     # Build the extension and run the Python tests
pixi run --locked lint     # Check Python lint rules
pixi run --locked format   # Format the Python sources
pixi run --locked typecheck
pixi run --locked cargo test --locked
pixi run --locked pre-commit run --all-files
```

Run `pixi run --locked pre-commit install` to enable the local commit hooks.

The development environment uses Python 3.12. The package requires Python 3.11
or newer, matching its Polars dependency. Python Polars is initially pinned to
the verified version, 2.1.0.
