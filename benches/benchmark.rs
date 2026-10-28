//! Run with `cargo bench --bench benchmark -- [rows] [samples] [parse_batch]`.

mod results;

use formula::{Models, parse};
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

fn measure<T>(iterations: usize, run: &mut impl FnMut() -> T) -> (f64, Vec<T>) {
    // Retain results until after timing to exclude their destruction.
    let mut outputs = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        outputs.push(black_box(run()));
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    (elapsed_ms, outputs)
}

fn measure_samples<T>(
    benchmark: &str,
    formula: &str,
    operation: Operation,
    samples: usize,
    iterations: usize,
    mut run: impl FnMut() -> T,
    describe: impl Fn(&T) -> Output,
) -> Vec<Sample> {
    // One complete, unreported warm-up batch.
    drop(measure(iterations, &mut run));
    (0..samples)
        .map(|sample| {
            let (elapsed_ms, outputs) = measure(iterations, &mut run);
            Sample {
                benchmark: benchmark.to_owned(),
                formula: formula.to_owned(),
                operation,
                sample,
                iterations,
                elapsed_ms,
                output: describe(&outputs[0]),
            }
        })
        .collect()
}

fn model_shapes(models: &Models) -> Output {
    Output::Materialized {
        models: models
            .iter()
            .map(|(_, model)| ModelShape {
                left: model.left().frame().shape(),
                right: model.right().frame().shape(),
            })
            .collect(),
    }
}

fn main() {
    let mut args = env::args().skip(1).filter(|arg| arg != "--bench");
    let rows = args.next().map_or(1_000_000, |arg| arg.parse().unwrap());
    let samples = args.next().map_or(10, |arg| arg.parse().unwrap());
    let parse_batch = args.next().map_or(1_000, |arg| arg.parse().unwrap());

    let data = sample_data(rows).unwrap();
    let mut results = Results {
        rows,
        threads: env::var("POLARS_MAX_THREADS").ok(),
        samples: Vec::new(),
    };
    for (name, source) in [
        ("numeric", "y ~ a*b"),
        ("transform", "y ~ log(a) + sqrt(b)"),
        ("categorical", "y ~ C(g1)"),
        ("categorical_native", "y ~ C(g_cat)"),
        ("integer", "y ~ F(g1)"),
        ("integer_native", "y ~ F(g_cat)"),
        ("interaction", "y ~ a:C(g1)"),
    ] {
        results.samples.extend(measure_samples(
            name,
            source,
            Operation::Parse,
            samples,
            parse_batch,
            || parse(black_box(source)).unwrap(),
            |specification| Output::Parsed {
                formulas: specification.formulas().len(),
            },
        ));
        results.samples.extend(measure_samples(
            name,
            source,
            Operation::ParseAndFit,
            samples,
            1,
            || {
                parse(black_box(source))
                    .unwrap()
                    .materialize(black_box(&data))
                    .unwrap()
            },
            model_shapes,
        ));
        let fitted = parse(source).unwrap().materialize(&data).unwrap();
        results.samples.extend(measure_samples(
            name,
            source,
            Operation::Evaluate,
            samples,
            1,
            || fitted.evaluate(black_box(&data)).unwrap(),
            model_shapes,
        ));
    }
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}
