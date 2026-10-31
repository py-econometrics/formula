# formula

`formula` is a Rust implementation of Wilkinson formulas. `formula` provides an interface
to parse formula strings and materialize data as a design
matrix for regression analysis.

> [!NOTE]
> `formula` is in early development and its scope, syntax, API is subject to change. It is
not ready for production use.

## Quickstart

```rust
use formula::parse;
use polars::df;
use polars::prelude::{DataFrame, Float64Chunked, NamedFrom, PolarsResult, Series};

fn sample_data(n: usize) -> PolarsResult<DataFrame> {
    df!(
        "y" => Float64Chunked::rand_standard_normal("y".into(), n),
        "x1" => Float64Chunked::rand_uniform("x1".into(), n, 1.0, 10.0),
        "x2" => Float64Chunked::rand_uniform("x2".into(), n, 1.0, 10.0),
        "category" => Series::new("category".into(), ["a", "b"])
            .sample_n(n, true, None, Some(1))?,
    )
}

fn main() {
    let formula = "y ~ add(x1, x2) + sw(log(x1)*sqrt(x2), C(category, reference='a'))";

    // Stores all specifications associated with a formula
    let specification = parse(formula).expect("valid formula");

    // Materializing a specification on data creates a collection of `Model`s
    // Transformations and data are shared across models, so no work is duplicated
    let data = sample_data(1_000_000).expect("valid sample data");
    let models = specification
        .materialize(&data)
        .expect("successful materialization");

    for (formula, model) in &models {
        println!("Model {formula:?}");
        println!("Dependent design:\n{}", model.left().frame());
        println!("Independent design:\n{}", model.right().frame());
    }
}
```

## Main API

| Type or method                     | Description                                                                                                      |
|------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `parse(source)`                    | Parse one formula string into a `Specification`.                                                                 |
| `parse_many(sources)`              | Parse several formula strings into a `Specification`.                                                            |
| `Specification::materialize(data)` | Evaluate the formulas against a Polars `DataFrame` and return `Models`.                                          |
| `Models`                           | A collection `Vec<(Formula, Model)>` storing all `Formula`-`Model` pairs materialized by a given `Specification` |

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

### Transformations

`formula` supports symbolic transformations that will be applied to the data provided
during formula materialization:

| Function                                                                     | Description                                                                                                            |
|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `log(x)`<br>`sqrt(x)`<br>`exp(x)`                                            | Apply the natural logarithm, square root, or exponential function to a numeric column or scalar argument.              |
| `add(x, y, ...)`<br>`sub(x, y, ...)`<br>`mul(x, y, ...)`<br>`div(x, y, ...)` | Add, subtract, multiply, or divide two or more numeric columns or scalar arguments.                                    |
| `C(group, reference=...)`                                                    | Create categorical indicator columns for `group`; the optional `reference` argument sets the reference level.          |
| `poly(x, degree=1, raw=false)`                                               | Expand a numeric input into orthonormal polynomial columns, or raw powers with `raw=true`.                             |

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
