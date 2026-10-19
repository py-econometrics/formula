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

    let fml = "y ~ add(x1, x2) + log(x1)*sqrt(x2) + C(category, reference='a') + F(f1, f2)";
    let specification = formula::parse(fml).expect("valid formula");
    let model = specification
        .materialize(&data)
        .expect("successful materialisation");
    let dependent = model.dependent().frame();
    let independent = model.independent().frame();

    println!("Formula: {fml}\n");
    println!("Dependent:\n{dependent}\n");
    println!("Independent:\n{independent}");
}
