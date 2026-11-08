# formula

`formula` is  [`Polars`](https://pola.rs)-powered implementation of Wilkinson formulas in Rust with a Python API. It parses formula strings and constructs data frames for regression
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

formulas = formula.parse("y ~ sw(x, z) | firm")  # Parse and expand formulas
fits = formulas.fit(data)  # Compute columns and learn transformation state
fit = fits[0]  # Integer indexing follows expansion order
same_fit = fits[fit.formula]  # Look up a fitted formula by its formula string

# Access the complete frame or select specific roles
frame = fit.frame()
dependent = fit.select(formula.Role.Dependent).frame()
independent = fit.select(formula.Role.Independent).frame()
fixed_effects = fit.select(formula.Role.FixedEffect).frame()

# Select roles before transforming data that has no dependent column.
predictors = fits.select([formula.Role.Independent, formula.Role.FixedEffect])
new_data = polars.DataFrame({"x": [8.0], "z": [9.0], "firm": ["a"]})
transformed = predictors.transform(new_data)
```

`parse` accepts one formula string or a sequence of strings and returns a
`Formulas` collection. Its `fit` method returns `FittedFormulas`, containing one
`FittedFormula` per expanded formula, with the computed columns and learned
transformation state. `transform` computes columns
for new data using that learned state. Fitting does not estimate regression coefficients.
`frame()` returns a Polars `DataFrame` containing all selected roles. `select` and
`transform` return new fitted formulas and retain fitted transform state. Selecting an absent
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

    // Stores the expanded formulas
    let formulas = parse(formula).expect("valid formula");

    // Fitting formulas creates a collection of `FittedFormula`s
    // Transformations and data are shared across fitted formulas
    let data = sample_data(1_000_000).expect("valid sample data");
    let fits = formulas
        .fit(&data)
        .expect("successful fitting");

    for fit in &fits {
        println!("Formula {}", fit.formula());
        println!("Fitted frame:\n{}", fit.frame().expect("valid frame"));
        let independent = fit
            .select(&[Role::Independent])
            .expect("independent role exists");
        println!("Independent design:\n{}", independent.frame().expect("valid frame"));
    }
}
```

## Formula syntax

A formula has a response on the left of `~` and predictors on the right. The intercept
is included by default; use `-1` or `0` to omit it. Adding `1` or subtracting `0`
restores a removed intercept.

```text
y ~ x + z       # response y, predictors x and z, with an intercept
y ~ x:z         # interaction between x and z
y ~ x*z         # equivalent to x + z + x:z
y ~ x/z         # equivalent to x + x:z
y ~ log(x) - 1  # transform x and omit the intercept
y ~ 0 + x       # omit the intercept
y ~ x - 1 - 0   # restore the intercept
y ~ (0 + x)     # retain the outer default intercept
```

Parentheses and stepwise arguments are evaluated locally. For example,
`y ~ (-1 + x)` retains the outer default intercept, and `y ~ sw(-1 + x, z)`
expands to `y ~ x` and `y ~ z`, both with an intercept.

Use backticks `` ` `` to escape column names containing spaces, punctuation or special
characters, for example ``y ~ `household income` ``.

### Multistage formulas

Square brackets attach child formulas to the containing expression. Using the
data from the Python example above:

```python
model = formula.parse("y ~ z + [x ~ log(z) - 1]").fit(data)[0]
child = model.children(formula.Role.Independent)[0]
same_child = model.children(formula.Role.Independent)["x ~ log(z) - 1"]
endogenous = child.select(formula.Role.Dependent).frame()
instruments = child.select(formula.Role.Independent).frame()
```

`children(role)` returns a `FittedFormulas` collection in attachment order, supporting
integer and formula-string indexing, selection, and transformation. A present
role without children returns an empty collection; an absent or excluded role
raises `KeyError`.
Children support the same selection and transformation methods as their parent
and remain usable after the parent is dropped. In Rust,
`model.children(Role::Independent)` returns a borrowed slice or `SelectionError`.

Sibling blocks and deeper nesting are supported. Every formula includes its own
intercept by default; explicit intercept controls and fixed effects apply locally.
For example, `y ~ z - 1 + [x ~ log(z)]` removes only the parent's intercept.

### Fixed effects

Use `|` to specify fixed effects separately from ordinary predictors:

```text
y ~ x | firm + year      # Separate firm and year fixed effects
y ~ x | firm:year        # One fixed effect for observed firm-year combinations
y ~ 1 + x | firm         # Fixed effects suppress even an explicitly requested intercept
```

Each grouping term produces one integer-coded column. Column names preserve the input column
names and factor order: `firm:year` and `year:firm` retain their respective order.

The fixed-effect pipe accepts column references, addition (`+`), interactions (`:`),
parentheses, and stepwise calls (`sw`, `sw0`, `csw`, `csw0`).

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
multiple formulas from a single formula string. For example, the formula
`y ~ sw(x1, x2)` expands to the formulas `y ~ x1` and `y ~ x2`.

| Operator             | Formulas generated                                                                 |
|----------------------|------------------------------------------------------------------------------------|
| `sw(x, z, w, ...)`   | One formula per argument: `x`, `z`, `w`, and so on.                                |
| `sw0(x, z, w, ...)`  | An empty expression, then one formula per argument: `x`, `z`, `w`, and so on.      |
| `csw(x, z, w, ...)`  | Cumulative formulas: `x`, then `x + z`, then `x + z + w`, and so on.               |
| `csw0(x, z, w, ...)` | An empty expression, then cumulative formulas: `x`, `x + z`, `x + z + w`, and so on. |

The empty alternative adds no terms and does not restore a removed intercept.
For example, `y ~ x - 1 + sw0(z)` expands to `y ~ x - 1` and `y ~ x + z - 1`.

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
