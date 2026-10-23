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
