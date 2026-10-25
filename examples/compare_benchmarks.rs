//! Run with `cargo run --example compare_benchmarks -- base.json candidate.json`.
//! Use `-` instead of the base path to report candidate timings without a baseline.

#[path = "../benches/results/mod.rs"]
mod results;

use results::{Operation, Results, Sample};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::ExitCode;
use std::{env, fs};

fn read_results(path: PathBuf) -> Results {
    let mut paths = if path.is_dir() {
        fs::read_dir(path)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>()
    } else {
        vec![path]
    };
    paths.sort();
    let mut rounds = paths
        .into_iter()
        .map(|path| serde_json::from_slice::<Results>(&fs::read(path).unwrap()).unwrap());
    let mut results = rounds.next().unwrap();
    for round in rounds {
        assert_eq!(results.rows, round.rows, "row count changed between rounds");
        assert_eq!(
            results.threads, round.threads,
            "threads changed between rounds"
        );
        results.samples.extend(round.samples);
    }
    results
}

fn group(results: &Results) -> BTreeMap<(&str, Operation), Vec<&Sample>> {
    let mut groups = BTreeMap::new();
    for sample in &results.samples {
        groups
            .entry((sample.benchmark.as_str(), sample.operation))
            .or_insert_with(Vec::new)
            .push(sample);
    }
    groups
}

fn summary(samples: &[&Sample]) -> (f64, f64) {
    let times = samples
        .iter()
        .map(|sample| sample.elapsed_ms / sample.iterations as f64)
        .collect::<Vec<_>>();
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance =
        times.iter().map(|time| (time - mean).powi(2)).sum::<f64>() / (times.len() - 1) as f64;
    (mean, variance.sqrt())
}

fn format_number(value: f64) -> String {
    if !value.is_finite() {
        return "—".to_owned();
    }
    if value == 0.0 {
        return "0".to_owned();
    }
    let scale = 10_f64.powf(2.0 - value.abs().log10().floor());
    let rounded = (value * scale).round() / scale;
    let decimals = (2.0 - rounded.abs().log10().floor()).max(0.0) as usize;
    format!("{rounded:.decimals$}")
}

fn format_summary(samples: &[&Sample]) -> String {
    let (mean, sd) = summary(samples);
    format!("{} ± {}", format_number(mean), format_number(sd))
}

fn main() -> ExitCode {
    let regression_percent = env::var("BENCH_REGRESSION_PERCENT")
        .ok()
        .map(|value| value.parse::<f64>().unwrap());
    let mut regressed = false;
    let mut args = env::args().skip(1);
    let base_path = args.next().unwrap();
    let base = (base_path != "-").then(|| read_results(base_path.into()));
    let candidate = read_results(args.next().unwrap().into());
    let base_groups = base.as_ref().map(group).unwrap_or_default();
    let candidate_groups = group(&candidate);
    let keys = base_groups
        .keys()
        .chain(candidate_groups.keys())
        .copied()
        .collect::<BTreeSet<_>>();

    println!("Times are ms/operation: mean ± sample standard deviation.");
    println!();
    if base.is_some() {
        println!("Change compares candidate and base means; positive means slower.");
        if let Some(limit) = regression_percent {
            println!("Mean runtime regressions greater than {limit}% fail the job.");
        }
        println!();
        println!("| Benchmark | Operation | Base | Candidate | Change | Status |");
        println!("|---|---|---:|---:|---:|---|");
    } else {
        println!("| Benchmark | Operation | Runtime (ms, mean ± SD) |");
        println!("|---|---|---:|");
    }

    for (benchmark, operation) in keys {
        let operation_name = match operation {
            Operation::Parse => "Parse",
            Operation::ParseAndFit => "Parse + fit",
            Operation::Evaluate => "Evaluate",
        };
        if base.is_none() {
            let timing = format_summary(&candidate_groups[&(benchmark, operation)]);
            println!("| {benchmark} | {operation_name} | {timing} |");
            continue;
        }
        match (
            base_groups.get(&(benchmark, operation)),
            candidate_groups.get(&(benchmark, operation)),
        ) {
            (Some(before), Some(after)) => {
                let expected = before[0];
                let comparable = base.as_ref().is_some_and(|base| {
                    base.rows == candidate.rows && base.threads == candidate.threads
                }) && before.iter().chain(after.iter()).all(|sample| {
                    sample.formula == expected.formula
                        && sample.iterations == expected.iterations
                        && sample.output == expected.output
                });
                if !comparable {
                    println!(
                        "| {benchmark} | {operation_name} | — | — | — | Incomparable configuration or output |"
                    );
                    continue;
                }
                let (base_mean, _) = summary(before);
                let (candidate_mean, _) = summary(after);
                let change = (candidate_mean / base_mean - 1.0) * 100.0;
                let regression = regression_percent
                    .is_some_and(|limit| candidate_mean > base_mean * (1.0 + limit / 100.0));
                regressed |= regression;
                let status = if regression {
                    "Regression"
                } else {
                    "Comparable"
                };
                let before = format_summary(before);
                let after = format_summary(after);
                println!(
                    "| {benchmark} | {operation_name} | {before} | {after} | {change:+.1}% | {status} |"
                );
            }
            (None, Some(after)) => {
                let timing = format_summary(after);
                println!("| {benchmark} | {operation_name} | — | {timing} | — | Added |");
            }
            (Some(before), None) => {
                let timing = format_summary(before);
                println!("| {benchmark} | {operation_name} | {timing} | — | — | Removed |");
            }
            (None, None) => unreachable!(),
        }
    }
    if regressed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use results::Output;

    fn sample(elapsed_ms: f64) -> Sample {
        Sample {
            benchmark: "numeric".to_owned(),
            formula: "y ~ a".to_owned(),
            operation: Operation::Parse,
            sample: 0,
            iterations: 10,
            elapsed_ms,
            output: Output::Parsed { formulas: 1 },
        }
    }

    #[test]
    fn summarizes_normalized_times_with_sample_standard_deviation() {
        let samples = [sample(10.0), sample(20.0), sample(30.0)];
        let samples = samples.iter().collect::<Vec<_>>();
        assert_eq!(summary(&samples), (2.0, 1.0));
        assert_eq!(format_summary(&samples), "2.00 ± 1.00");
    }

    #[test]
    fn single_sample_has_no_standard_deviation() {
        assert_eq!(format_summary(&[&sample(20.0)]), "2.00 ± —");
    }

    #[test]
    fn formats_three_significant_digits_across_scales() {
        for (value, expected) in [
            (0.0, "0"),
            (0.001417, "0.00142"),
            (16.871078, "16.9"),
            (9.999, "10.0"),
            (12345.0, "12300"),
        ] {
            assert_eq!(format_number(value), expected);
        }
    }
}
