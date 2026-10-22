use polars::df;

fn main() {
    let data = df!(
        "y" => [1.0, 2.0, 3.0],
        "x1" => [2.0, 4.0, 6.0],
        "x2" => [12.0, 2.0, 1.0],
        "category" => ["a", "b", "a"],
        "f1" => [13, 1, 1],
        "f2" => ["x", "y", "z"],
    )
    .expect("valid sample data");

    let fml = "y ~ add(x1, x2) + sw(log(x1)*sqrt(x2), C(category, reference='a')) + F(f1, f2)";
    let specification = formula::parse(fml).expect("valid formulas");
    let models = specification
        .materialize(&data)
        .expect("successful materialisation");
    println!("Formula: {fml}\n");
    for (formula, model) in &models {
        println!("Model {formula:?}\n");
        println!("Dependent:\n{}\n", model.dependent().frame());
        println!("Independent:\n{}\n", model.independent().frame());
    }
}
