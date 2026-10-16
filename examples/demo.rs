use polars::df;

fn main() {
    let data = df!(
        "y" => [1.0, 2.0, 3.0],
        "x" => [2.0, 4.0, 6.0],
        "group" => ["a", "b", "a"],
    )
    .expect("valid sample data");

    let fml = "y ~ log(x) + C(group, reference='a')";
    let specification = formula::parse(fml).expect("valid formula");
    let model = specification
        .materialize(&data)
        .expect("successful materialisation");
    let dependent = model.dependent().to_frame().expect("valid frame");
    let independent = model.independent().to_frame().expect("valid frame");

    println!("Formula: {fml}\n");
    println!("Dependent:\n{dependent}\n");
    println!("Independent:\n{independent}");
}
