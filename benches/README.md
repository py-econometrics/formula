# Benchmarks

Run in Cargo's optimized benchmark profile. Positional arguments are rows,
measured samples, and parses per parsing sample (defaults: 1000000, 10, 1000).

```sh
POLARS_MAX_THREADS=1 cargo bench --locked --bench benchmark -- 1000000 10 1000 > /tmp/base.json
POLARS_MAX_THREADS=1 cargo bench --locked --bench benchmark -- 1000000 10 1000 > /tmp/candidate.json
cargo run --locked --example compare_benchmarks -- /tmp/base.json /tmp/candidate.json
```

These commands compare two runs of the current revision. The comparator also
accepts directories of JSON results, pooling samples across rounds.

The Benchmarks workflow compares each PR's merge commit with its exact base
(the merge commit's first parent). Manual runs accept explicit base and candidate
revisions; choose the same revision for a control run. It uses Ubuntu 24.04,
Rust 1.99.1, default crate features, release builds, and one Polars thread.
Both revisions are built with their own lockfiles before any measurement.
Each revision uses a separate target directory. Cargo launches the benchmarks;
its build checks are outside the timed sections.
Four rounds alternate base/candidate and candidate/base order. Each process
performs its own warm-up. Results appear in the job summary and artifact.

Orchestration lives entirely in `.github/workflows/benchmarks.yml`, with separate
build, benchmark, and comparison steps. If the base predates the harness, only
the candidate runs, and its timings are reported without percentage changes.
Compilation, execution, and malformed JSON failures fail
the job. A comparable workload whose mean runtime increases by more than 10%
also fails the job; the full report and artifacts remain available.

The runner writes JSON using the shared types in `results/mod.rs`. Results
contain raw samples, formulas, operation types, batch sizes, output shapes,
row count, and the configured `POLARS_MAX_THREADS` value. The comparator reports
mean milliseconds per operation ± sample standard deviation, rounded to three
significant digits. Percentage changes compare means. Candidate-only runs use a
three-column table without empty baseline or percentage columns. Standard
deviation is unavailable (`—`) when there is only one sample.
Operations are parsing, parsing plus fitting (`ParseAndFit`), and evaluation.
Durations are stored as fractional milliseconds to retain submillisecond precision.
Different formulas, batch sizes, output shapes, row counts, or thread settings
are reported as incomparable; added and removed workloads retain their available
timings, with percentage changes omitted.
The workflow sets `BENCH_REGRESSION_PERCENT=10` to control the regression
threshold. Without this environment variable, the comparator only reports
timings. Candidate-only, added, removed, and incomparable workloads do not
trigger the regression check. Runner noise can cause false alarms.

Data construction, warm-up, output inspection, serialization, and result
destruction are outside timing. Parsing samples retain a batch of parsed
results until timing stops; fit and evaluation samples contain one operation.
Evaluation reuses a model fitted outside timing. The deterministic dataset has
90% fully finite rows; the remaining rows contain nulls, NaN, or infinities.
