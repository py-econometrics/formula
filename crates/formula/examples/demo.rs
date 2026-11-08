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
    let fits = formulas.fit(&data).expect("successful fitting");

    for fit in &fits {
        println!("Formula {}", fit.formula());
        for (role, label) in [
            (Role::Dependent, "Dependent design"),
            (Role::Independent, "Independent design"),
            (Role::FixedEffect, "Fixed effects"),
        ] {
            let selected = fit.select(&[role]).expect("role exists in the formula");
            println!("{label}:\n{}", selected.frame().expect("valid frame"));
        }
    }
}
