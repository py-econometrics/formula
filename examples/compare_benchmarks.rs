//! Run with `cargo run --example compare_benchmarks -- base.json candidate.json`.
//! Use `-` instead of the base path to report candidate timings without a baseline.

#[path = "../benches/results/mod.rs"]
mod results;

use results::{Operation, Results, Sample};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
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

fn summary(samples: &[&Sample]) -> (f64, f64, f64) {
    let mut times = samples
        .iter()
        .map(|sample| sample.elapsed_ms / sample.iterations as f64)
        .collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    let median = (times[(times.len() - 1) / 2] + times[times.len() / 2]) / 2.0;
    (median, times[0], times[times.len() - 1])
}

fn format_summary(samples: &[&Sample]) -> String {
    let (median, min, max) = summary(samples);
    format!("{median:.6} ({min:.6}–{max:.6})")
}

fn main() {
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

    println!("Times are ms/operation: median (min–max). Change is candidate/base − 1.");
    println!();
    println!("| Benchmark | Operation | Base | Candidate | Change | Status |");
    println!("|---|---|---:|---:|---:|---|");

    for (benchmark, operation) in keys {
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
                        "| {benchmark} | {operation:?} | — | — | — | Incomparable configuration or output |"
                    );
                    continue;
                }
                let (base_median, base_min, base_max) = summary(before);
                let (candidate_median, candidate_min, candidate_max) = summary(after);
                let change = (candidate_median / base_median - 1.0) * 100.0;
                println!(
                    "| {benchmark} | {operation:?} | {base_median:.6} ({base_min:.6}–{base_max:.6}) | {candidate_median:.6} ({candidate_min:.6}–{candidate_max:.6}) | {change:+.1}% | Comparable |"
                );
            }
            (None, Some(after)) => {
                let timing = format_summary(after);
                let status = if base.is_some() {
                    "Added"
                } else {
                    "No baseline"
                };
                println!("| {benchmark} | {operation:?} | — | {timing} | — | {status} |");
            }
            (Some(before), None) => {
                let timing = format_summary(before);
                println!("| {benchmark} | {operation:?} | {timing} | — | — | Removed |");
            }
            (None, None) => unreachable!(),
        }
    }
}
