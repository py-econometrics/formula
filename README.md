# formula

`formula` is a Rust implementation of Wilkinson formulas.

> [!NOTE]
> `formula` is in early development and its scope, syntax, API is subject to change.

## Scope

`formula` provides an interface to parse formula strings and materialize data as a design
matrix for regression analysis.

## Quickstart

```rust
use formula::parse;
use polars::df;
use polars::prelude::{DataFrame, PolarsResult};

fn sample_data() -> PolarsResult<DataFrame> {
    df!(
        "y" => [1.0, 2.0, 3.0],
        "x1" => [2.0, 4.0, 6.0],
        "x2" => [12.0, 2.0, 1.0],
        "category" => ["a", "b", "a"],
        "f1" => [13, 1, 1],
        "f2" => ["x", "y", "z"],
    )
}

fn main() {
    let formula = "y ~ add(x1, x2) + sw(log(x1)*sqrt(x2), C(category, reference='a')) + F(f1, f2)";

    // Stores all specifications associated with a formula
    let specification = parse(formula).expect("valid formula");

    // Materializing a specification on data creates a collection of `Model`s
    // Transformations and data are shared across models, so no work is duplicated
    let data = sample_data().expect("valid sample data");
    let models = specification
        .materialize(&data)
        .expect("successful materialization");

    for (formula, model) in &models {
        println!("Model {formula:?}");
        println!("Dependent design:\n{}", model.dependent().frame());
        println!("Independent design:\n{}", model.independent().frame());
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
y ~ x*z         # x, z, and their interaction
y ~ log(x) - 1  # transform x and omit the intercept
```

### Transformations

Function calls apply transformations:

| Function                                                                     | Description                                                                                                            |
|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `log(x)`<br>`sqrt(x)`<br>`exp(x)`                                            | Apply the natural logarithm, square root, or exponential function to a numeric input.                                  |
| `add(x, y, ...)`<br>`sub(x, y, ...)`<br>`mul(x, y, ...)`<br>`div(x, y, ...)` | Add, subtract, multiply, or divide two or more numeric inputs.                                                         |
| `C(group, reference=...)`                                                    | Create categorical indicator columns for `group`; the optional `reference` argument sets the reference level.          |
| `F(group)`                                                                   | Encode observed levels as integer codes. Multiple inputs, such as `F(firm, year)`, encode their observed combinations. |

### Multiple estimation

Expansion operators generate multiple formulas in a `Specification`:

| Operator             | Models generated                                                                        |
|----------------------|-----------------------------------------------------------------------------------------|
| `sw(x, z, w, ...)`   | One model per argument: `x`, `z`, `w`, and so on.                                       |
| `sw0(x, z, w, ...)`  | An empty expression, then one model per argument: `1`, `x`, `z`, `w`, and so on.        |
| `csw(x, z, w, ...)`  | Cumulative models: `x`, then `x + z`, then `x + z + w`, and so on.                      |
| `csw0(x, z, w, ...)` | An empty expression, then cumulative models: `1`, `x`, `x + z`, `x + z + w`, and so on. |