# Benchmarks

Run in Cargo's optimized benchmark profile. Positional arguments are rows and
measured samples (defaults: 1000000 and 10).

```sh
POLARS_MAX_THREADS=1 cargo bench --locked --bench benchmark -- 1000000 10 > /tmp/base.json
POLARS_MAX_THREADS=1 cargo bench --locked --bench benchmark -- 1000000 10 > /tmp/candidate.json
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
contain raw samples, formulas, operation types, output shapes,
row count, and the configured `POLARS_MAX_THREADS` value. The comparator reports
mean milliseconds per operation ± sample standard deviation, rounded to three
significant digits. Percentage changes compare means. Candidate-only runs use a
three-column table without empty baseline or percentage columns. Standard
deviation is unavailable (`—`) when there is only one sample.
The two operations are parsing plus materialization (`ParseAndFit`) and fitted
evaluation (`Evaluate`). There is no standalone parsing benchmark. Each sample
times one operation. Older parsing results remain readable for comparisons
against earlier revisions.
Durations are stored as fractional milliseconds to retain submillisecond precision.
Different formulas, output shapes, row counts, or thread settings
are reported as incomparable; added and removed workloads retain their available
timings, with percentage changes omitted.
The workflow sets `BENCH_REGRESSION_PERCENT=10` to control the regression
threshold. Without this environment variable, the comparator only reports
timings. Candidate-only, added, removed, and incomparable workloads do not
trigger the regression check. Runner noise can cause false alarms.

Data construction, warm-up, output inspection, serialization, and result
destruction are outside timing.
Evaluation reuses a model fitted outside timing. The deterministic dataset has
90% fully finite rows; the remaining rows contain nulls, NaN, or infinities.
The `categorical` workload applies `C` to a string column; `integer` encodes it
as a fixed effect using pipe syntax. Their `_native` counterparts use a
categorical copy of the same column.
Categorical conversion happens during data construction, outside timing.

The ten workloads cover numeric interactions, expression transforms, raw and
orthogonal degree-3 polynomials, categorical encoding, integer encoding, and
numeric-by-categorical interactions. Categorical and integer encoding each retain
string and native-categorical inputs because they exercise different code paths.
The `fixed_effects` workload combines numeric controls, their interaction, and a
log transform with firm and year-by-group fixed effects:
`y ~ a*b + log(a) | firm + year:g1`. The panel-like inputs contain up to 10,000
string-valued firms and 20 numeric calendar years. Fixed-effect output shapes are
recorded alongside the dependent and independent shapes.

The two polynomial workloads produce four right-hand columns including the
intercept. Nulls, NaN, and infinities produce null polynomial rows and are excluded
from orthogonal fitting. Provide at least four distinct finite training values;
the quick run below uses 1,000 rows. Higher degrees and polynomial interaction
combinations can be benchmarked separately when investigating those operations.

```sh
POLARS_MAX_THREADS=1 cargo bench --locked --bench benchmark -- 1000 2 > /tmp/bench.json
cargo run --locked --example compare_benchmarks -- - /tmp/bench.json
```

The workflow discovers workloads through the runner's JSON output. New workloads
appear as added and old workloads as removed, without percentage changes.
