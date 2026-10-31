//! Run with `cargo bench --bench benchmark -- [rows] [samples]`.

mod results;

use formula::{Models, Role, parse};
use polars::prelude::*;
use results::{ModelShape, Operation, Output, Results, Sample};
use std::env;
use std::hint::black_box;
use std::time::Instant;

// Deterministic mixed inputs; data generation is outside timing.
fn sample_data(rows: usize) -> PolarsResult<DataFrame> {
    // Every tenth row is exceptional; each numeric exception occurs once per 40 rows.
    let numeric = |i: usize, value: f64| match i % 40 {
        9 => None,
        19 => Some(f64::NAN),
        29 => Some(f64::INFINITY),
        39 => Some(f64::NEG_INFINITY),
        _ => Some(value),
    };

    let mut data = df!(
        "y" => (0..rows).map(|i| numeric(i, i as f64)).collect::<Vec<_>>(),
        "a" => (0..rows).map(|i| numeric(i, 1.0 + (i % 100) as f64)).collect::<Vec<_>>(),
        "b" => (0..rows).map(|i| numeric(i, 1.0 + (i % 73) as f64)).collect::<Vec<_>>(),
        "g1" => (0..rows).map(|i| {
            if i % 10 == 9 {
                None
            } else if i % 2 == 0 {
                Some("first")
            } else {
                Some("second")
            }
        }).collect::<Vec<_>>(),
        "firm" => (0..rows).map(|i| format!("firm_{}", i % 10_000)).collect::<Vec<_>>(),
        "year" => (0..rows).map(|i| 2000 + ((i / 10_000) % 20) as i32).collect::<Vec<_>>(),
    )?;
    let categorical = data
        .column("g1")?
        .cast(&DataType::from_categories(Categories::random(
            "formula-benchmarks".into(),
            CategoricalPhysical::U32,
        )))?
        .with_name("g_cat".into());
    data.with_column(categorical)?;
    Ok(data)
}

fn measure_samples(
    benchmark: &str,
    formula: &str,
    operation: Operation,
    samples: usize,
    mut run: impl FnMut() -> Models,
) -> Vec<Sample> {
    drop(black_box(run())); // One unreported warm-up.
    (0..samples)
        .map(|sample| {
            let start = Instant::now();
            let output = black_box(run());
            let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
            // Inspect and drop the output after timing.
            Sample {
                benchmark: benchmark.to_owned(),
                formula: formula.to_owned(),
                operation,
                sample,
                iterations: 1,
                elapsed_ms,
                output: model_shapes(&output),
            }
        })
        .collect()
}

fn model_shapes(models: &Models) -> Output {
    Output::Materialized {
        models: models
            .iter()
            .map(|(_, model)| ModelShape {
                left: model
                    .expression(Role::Dependent)
                    .expect("benchmark has a dependent expression")
                    .frame()
                    .shape(),
                right: model
                    .expression(Role::Independent)
                    .expect("benchmark has an independent expression")
                    .frame()
                    .shape(),
                fixed_effects: model
                    .expression(Role::FixedEffect)
                    .map(|expression| expression.frame().shape()),
            })
            .collect(),
    }
}

fn main() {
    let mut args = env::args().skip(1).filter(|arg| arg != "--bench");
    let rows = args.next().map_or(1_000_000, |arg| arg.parse().unwrap());
    let samples = args.next().map_or(10, |arg| arg.parse().unwrap());

    let data = sample_data(rows).unwrap();
    let mut results = Results {
        rows,
        threads: env::var("POLARS_MAX_THREADS").ok(),
        samples: Vec::new(),
    };
    for (name, source) in [
        ("numeric", "y ~ a*b"),
        ("transform", "y ~ log(a) + sqrt(b)"),
        ("polynomial_raw_degree3", "y ~ poly(a, degree=3, raw=true)"),
        ("polynomial_orthogonal_degree3", "y ~ poly(a, degree=3)"),
        ("categorical", "y ~ C(g1)"),
        ("categorical_native", "y ~ C(g_cat)"),
        ("integer", "y ~ a | g1"),
        ("integer_native", "y ~ a | g_cat"),
        ("interaction", "y ~ a:C(g1)"),
        ("fixed_effects", "y ~ a*b + log(a) | firm + year:g1"),
    ] {
        results.samples.extend(measure_samples(
            name,
            source,
            Operation::ParseAndFit,
            samples,
            || {
                parse(black_box(source))
                    .unwrap()
                    .materialize(black_box(&data))
                    .unwrap()
            },
        ));
        let fitted = parse(source).unwrap().materialize(&data).unwrap();
        results.samples.extend(measure_samples(
            name,
            source,
            Operation::Evaluate,
            samples,
            || fitted.evaluate(black_box(&data)).unwrap(),
        ));
    }
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}
