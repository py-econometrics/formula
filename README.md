# formula

`formula` is  [`Polars`](https://pola.rs)-powered implementation of Wilkinson formulas in Rust with a Python API. It parses formula strings and constructs data frames for regression
analysis. Its design is inspired by [`PyFixest`](https://github.com/py-econometrics/pyfixest) but intended to be generic.

## Python API

### Parsing formula strings

```python
import polars

import formula

data = polars.DataFrame(
    {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "x2": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "category": ["a", "b", "a", "b", "a", "b"],
        "firm": ["firm_a", "firm_b", "firm_c", "firm_a", "firm_b", "firm_c"],
        "year": [2020, 2021, 2022, 2020, 2021, 2022],
    }
)

# Parsing formula strings returns a collection of `Formula` objects
source = "y ~ sw(log(x1)*sqrt(x2), C(category, omit='a')) | firm + year"
formulas = formula.parse(source)  # Parse and expand formulas
first_formula = formulas[0]  # Parsed formulas are an indexable collection
same_formula = formulas[str(first_formula)]  # Look-up by formula-string also work
```

### Materializing formula data

A `Formula` object can "materialize" data according to the instructions encoded in a formula string. This retains state for future materializations, for example categorical encoding.

```python
fits = formulas.fit(data)  # Construct columns and learn transformation state
# fits = formulas.fit(data, ensure_full_rank=False)  # Keep every categorical level
fit = fits[0]  # Integer indexing follows expansion order
same_fit = fits[fit.formula]  # Look up a fitted formula by its formula string
```

A `FittedFormula` exposes its data as a `DataFrame` that can be queried for specific `Role`s, for example dependent or independent variables.

```python
frame = fit.frame()  # Get the materialized data as a single frame

# Select specific `Role`s, e.g., dependent variables, independent variables, and fixed effects
dependent = fit.select(formula.Role.Dependent).frame()
independent = fit.select(formula.Role.Independent).frame()
fixed_effects = fit.select(formula.Role.FixedEffect).frame()
```

It is also possible to query specific formula terms using `FittedFormula.select_terms`.

```python
# Select terms by their formula strings
interaction_terms = fit.select_terms("log(x1)*sqrt(x2)", role=formula.Role.Independent)
# Select the categorical term in the second formula
categorical_fit = fits["y ~ C(category, omit='a') | firm + year"]
categorical_terms = categorical_fit.select_terms(
    "C(category, omit='a')", role=formula.Role.Independent
)
```

It is possible to retain only part of a `FittedFormula` to apply it to new data. This is useful for prediction:

```python
predictors = fits.select(roles=[formula.Role.Independent, formula.Role.FixedEffect])
new_data = polars.DataFrame(
    {
        "x1": [8.0],
        "x2": [9.0],
        "category": ["b"],
        "firm": ["firm_a"],
        "year": [2020],
    }
)
transformed = predictors.transform(new_data)
```

### Inspecting materialized data

Materialized data expose metadata from their underlying formula. This makes it possible to inspect the semantics of a column and identify its provenance in the underlying formula:

```python
for term in interaction_terms.terms(role=formula.Role.Independent):
    print("Factors:", term.factors)
    print("Columns:", [column.data.name for column in term.columns])

for term in categorical_terms.terms(role=formula.Role.Independent):
    # Inspect the factors constituting a term
    for factor in term.factors:
        print(factor)
    # Inspect the columns a term produces
    for column in term.columns:
        for contribution in column.contributions:
            print(contribution.key, column.data)
```

`column.term` lists every factor of the column's term, and `column.contributions`
the factors the column includes, each with its key (such as a level). The two differ
when full-rank coding leaves a factor out of a column: in `y ~ a:b`, the column
`b[T.v]` belongs to the term `a:b` but has a contribution for `b` only. The intercept
column has no contributions.

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
        "y ~ add(x1, x2) + sw(log(x1)*sqrt(x2), C(category, omit='a')) | firm + year";

    // Stores the expanded formulas
    let formulas = parse(formula).expect("valid formula");

    println!("Expanded formulas: {}", formulas.len());
    for formula in &formulas {
        println!("Formula: {formula}");
    }

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

Like `FittedFormulas`, the Rust `Formulas` collection supports `len()`, `is_empty()`,
`iter()`, and iteration over `&formulas`. Iterators of `Formula` can be collected
into `Formulas`, preserving their order and duplicates; parsing continues to
deduplicate expanded formulas.

## Formula syntax

A formula has a response on the left of `~` and predictors on the right. The intercept
is included by default; use `-1` or `0` to omit it. Adding `1` or subtracting `0`
restores a removed intercept.

```text
y ~ x + z          # response y, predictors x and z, with an intercept
y ~ x:z            # interaction between x and z
y ~ x*z            # equivalent to x + z + x:z
y ~ x/z            # equivalent to x + x:z
y ~ (x + z + w)^2  # main effects and every two-way interaction
y ~ log(x) - 1     # transform x and omit the intercept
y ~ 0 + x          # omit the intercept
y ~ x - 1 - 0      # restore the intercept
y ~ (0 + x)        # retain the outer default intercept
~ 0 + log(x)       # no response, like 0 ~ 0 + log(x)
```

Parentheses and stepwise arguments are evaluated locally. For example,
`y ~ (-1 + x)` retains the outer default intercept, and `y ~ sw(-1 + x, z)`
expands to `y ~ x` and `y ~ z`, both with an intercept.

Use backticks `` ` `` to escape column names containing spaces, punctuation or special
characters, for example ``y ~ `household income` ``.

### Categorical variables

String, categorical, and enum columns used directly as predictors are encoded as if
wrapped in `C()`, with indicator columns named after the column.

Structural coding for ordinary predictors follows the column space of Formulaic.
An unrestricted categorical factor is
either *reduced* (its reference level is dropped, and columns are named
`[T.level]`) or *full* (one column per level, named `[level]`). Terms are considered
from lowest to highest order, and each term adds only the columns that the
intercept, fixed effects, and earlier terms do not already span. A term can
therefore emit several groups of columns, some of which leave out one of its
factors: with an intercept, `y ~ a:b` gives `b[T.v]` and `a[T.q]:b[u]`, `a[T.q]:b[v]`,
`a[T.r]:b[u]`, `a[T.r]:b[v]`, all of which belong to the term `a:b`. Columns keep the
order in which terms are written.

```text
y ~ state                       # Intercept, state[T.NY], state[T.TX]
y ~ state - 1                   # state[CA], state[NY], state[TX]
y ~ C(state)                    # Intercept, C(state)[T.NY], C(state)[T.TX]
y ~ C(state, omit='NY')         # Intercept, C(state, omit='NY')[CA], C(state, omit='NY')[TX]
y ~ C(state, omit='NY') - 1     # C(state, omit='NY')[CA], C(state, omit='NY')[TX]
y ~ state:x                     # Intercept, state[CA]:x, state[NY]:x, state[TX]:x
y ~ x + state:x                 # Intercept, x, state[T.NY]:x, state[T.TX]:x
y ~ state*group                 # main effects and their interaction, all reduced
y ~ state:group                 # Intercept, group[T.b], state[T.NY]:group[a], ...: two groups of columns
y ~ state | firm                # state[T.NY], state[T.TX]: the fixed effect spans the constant
y ~ state:C(firm) | firm        # state[T.NY]:C(firm)[firm_a], ...: state reduced within each firm
```

Observed levels are sorted by numeric value, string or categorical label, or enum
declaration order. Automatic reduction drops the first fitted level. Use
`omit=` with a scalar or list to exclude specific levels in any coding mode,
including for an explicit categorical response. Structural coding accounts for the
space spanned by the retained levels. Known omitted levels give zeros in retained
indicators. The coding is decided once per term in each formula when fitting, so
`y ~ C(a)` and `y ~ C(a) - 1` fitted together can code the same term differently.
`transform()` and `select_terms()` keep the fitted coding.

A bare categorical predictor and `C()` on the same input share a level space. The
structural coding of `a:C(a)` uses the first written factor to name its indicators
and records both factors in the term. Repeated factors with explicit omissions keep
only the levels retained by every occurrence. A numeric slope `x` and `C(x)` have
distinct structural identities. Numerical redundancy between their columns is
left to the estimator.

A fixed-effect group intercept spans the constant and functions of its grouping
inputs. A loading spans its numeric slope within and across groups, so
`| firm[[x]]` can absorb `x`, and `| x` can absorb both `x` and `C(x)`.

A term that is already spanned entirely has no columns, for example `C(state)` in
`y ~ state + C(state)`, `C(firm)` in `y ~ C(firm) | firm`, or `x` in
`y ~ x | firm[[x]]`. The term is kept, so `select_terms()` still finds it, and fitting
emits a `formula.RankDeficiencyWarning` naming the formula and terms (in Rust,
`FittedFormula::rank_deficiencies()`). Redundancy that depends on the data, such as
`x + mul(2, x)` or empty interaction cells, is not detected.

To disable structural coding, fit with `ensure_full_rank=False`
(`FitOptions { ensure_full_rank: false, ..Default::default() }` with
`Formulas::fit_with_options` in Rust).
Each term emits the Cartesian product of its factors' columns, subject to explicit
omissions. This mode emits no rank deficiency warnings.

Whether a column is categorical is decided from its dtype when fitting. `transform()`
reuses the fitted levels and requires the same dtype; unseen levels and nulls give
null indicators. Only predictors written as a bare column are converted. A bare
boolean column is numeric, coded 0/1 (use `C(flag)` for indicators), a string
response remains an error, and call arguments such as `C(state)` or the fixed
effect in `| state` receive the original column.

A numeric column read as strings, for example from a CSV file, therefore becomes a
categorical predictor instead of raising an error. Cast such columns before fitting.

### Fixest-style interactions with `i()`

`i(f)` encodes `f` as a categorical, like `C(f)`, but names its columns `f::level`.
`i(f, x)` interacts the levels of `f` with a second variable. A numeric or boolean
`x` gives one slope per level, named `f::level:x`. A second variable is categorical
when it is a string, categorical or enum column, when it is written as `C(g)`, or
when `ref2` or `bin2` is given, and gives one column per pair of levels,
`f::level:g::level`.

```text
y ~ i(period, ref=-1) | unit + year     # event-study indicators
y ~ i(period, treated, ref=-1) | unit   # event-study slopes on a treatment indicator
y ~ i(group, x)                         # group-specific slopes on x
y ~ i(group, region, ref='a', ref2='n') # group-by-region cells
y ~ i(group, C(year))                   # a numeric second variable as a factor
y ~ i(group, bin={'ab': ['a', 'b']})    # merge levels a and b
```

`ref` and `ref2` omit levels of the first and second variable, like `omit=` for
`C()`, and accept a value or a list. Integer and float references name the same
level when the value is exact, so `ref=-1` also names `-1.0`. `bin={'new': ['old', ...]}`
relabels the listed values before levels are fitted and `ref` is applied. Every
binned value must occur in the fitting data, and each value may belong to one bin.
A label that is not of the input's type, such as `'pre'` for an integer event time,
bins the variable as strings.

Structural coding treats `i(f, g)` as the interaction `f:g` and `i(f, x)` as `f:x`.
So `y ~ i(f)` drops the first level next to an intercept or a fixed effect, `y ~ i(f) | f`
has no columns, and `y ~ x + i(f, x)` drops the first slope. An unbinned `i(f)` shares
its level space with `f` and `C(f)`. A binned variable has its own level space, so
`f + i(f, bin=...)` is not detected as redundant. Column names of `i()` omit its
options, so two spellings that keep the same level, such as `i(f)` and
`i(f, bin=...)`, raise an error that names the repeated column.

Each column of `i()` has one contribution, the `i()` factor, keyed by the level of
each axis: `(level, None)` for a slope column, and `None` for an axis a column group
leaves out, as in the column `g::v` of `y ~ i(f, g)`.

### Missing values

By default, `fit()` leaves out the rows where any column a formula reads is null,
NaN or infinite. That includes the response, the predictors, child formulas,
fixed-effect groups and loadings, and the inputs of transformations. Rows that a
transformation makes nonfinite, such as `log(0)`, are left out too. Fitted state,
such as categorical levels and the mean of `center(x)`, is learned on the rows that
remain, so a level observed only in a dropped row is not a level.

```python
model = formula.parse("y ~ x + log(z)").fit(data)[0]
model.dropped_rows  # positions of the left-out rows in `data`, ascending
```

Each formula keeps its own rows: in `y ~ sw(x, z)`, a missing `x` drops a row only
from the first model. Formulas that keep the same rows share their fitted
transformations. Pass `missing="keep"` to fit on every row, so that missing inputs
give null or nonfinite outputs, or `missing="raise"` to fail with the positions of
the incomplete rows (`FitOptions { missing: Missing::Keep, ..Default::default() }`
in Rust). `transform()` never drops rows.

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
y ~ x | firm + year  # Separate firm and year fixed effects
y ~ x | firm:year  # One fixed effect for observed firm-year combinations
y ~ x | firm[time]  # Firm intercepts and firm-specific coefficients on time
y ~ x | firm[[time]]  # Firm-specific coefficients on time, without firm intercepts
y ~ x | firm[time, exposure]  # Firm intercepts and separate coefficients on both loadings
y ~ x | firm:year[time]  # Intercepts and slopes for observed firm-year combinations
```

Prediction reuses the fitted group encoding and evaluates loadings on the new
data. Nulls are preserved, and unseen groups receive null codes.

### Transformations

`formula` supports the following transformations, applied to data passed to `fit()`
or `transform()`:

| Function                                                                     | Description                                                                                                   |
|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `log(x)`<br>`sqrt(x)`<br>`exp(x)`                                            | Apply the natural logarithm, square root, or exponential function to a numeric column or scalar argument.     |
| `add(x, y, ...)`<br>`sub(x, y, ...)`<br>`mul(x, y, ...)`<br>`div(x, y, ...)` | Add, subtract, multiply, or divide two or more numeric columns or scalar arguments.                           |
| `pow(x, k)`                                                                   | Raise a numeric column to the power `k`, a scalar or another column.                                          |
| `center(x)`<br>`scale(x)`                                                    | Subtract the fitted mean of `x`, and for `scale` divide by its fitted sample standard deviation. `transform()` reuses both. |
| `C(group, omit=...)`                                                    | Create categorical indicator columns for `group`, optionally excluding the levels specified by `omit`. |
| `i(group, x, ref=..., ref2=..., bin=..., bin2=...)`                          | Create fixest-style indicators `group::level`, optionally interacted with a second variable (see above). |
| `poly(x, degree=1, raw=false)`                                               | Expand a numeric input into orthonormal polynomial columns, or raw powers with `raw=true`. `poly(x, 2)` gives the degree by position. |

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

| Operator             | Formulas generated                                                                   |
|----------------------|--------------------------------------------------------------------------------------|
| `sw(x, z, w, ...)`   | One formula per argument: `x`, `z`, `w`, and so on.                                  |
| `sw0(x, z, w, ...)`  | An empty expression, then one formula per argument: `x`, `z`, `w`, and so on.        |
| `csw(x, z, w, ...)`  | Cumulative formulas: `x`, then `x + z`, then `x + z + w`, and so on.                 |
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
