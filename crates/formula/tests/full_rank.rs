//! Structural full-rank coding of categorical factors.

#[path = "full_rank/acceptance.rs"]
mod acceptance;

use formula::{ComponentKey, FitOptions, FittedFormula, Role};
use polars::prelude::{DataFrame, DataType};

/// Every combination of `a` (p, q, r), `b` (u, v) and `c` (m, n), twice, with a numeric `x`.
fn grid_data() -> DataFrame {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    for _ in 0..2 {
        for level_a in ["r", "p", "q"] {
            for level_b in ["v", "u"] {
                for level_c in ["n", "m"] {
                    a.push(level_a);
                    b.push(level_b);
                    c.push(level_c);
                }
            }
        }
    }
    let x = (0..a.len())
        .map(|index| ((index * 7919) % 23) as f64 / 3.0 - 2.0)
        .collect::<Vec<_>>();
    let y = (0..a.len()).map(|index| index as f64).collect::<Vec<_>>();
    let f = (0..a.len())
        .map(|index| if index % 3 == 0 { "k" } else { "l" })
        .collect::<Vec<_>>();
    let g = (0..a.len())
        .map(|index| if index % 4 < 2 { "s" } else { "t" })
        .collect::<Vec<_>>();
    let x = x.into_iter().map(|value| value + 3.0).collect::<Vec<_>>();
    polars::df!("y" => y, "x" => x, "a" => a, "b" => b, "c" => c, "f" => f, "g" => g).unwrap()
}

fn fit_all(source: &str, data: &DataFrame, options: FitOptions) -> Vec<FittedFormula> {
    formula::parse(source)
        .unwrap()
        .fit_with_options(data, options)
        .unwrap()
        .iter()
        .cloned()
        .collect()
}

fn fit(source: &str, data: &DataFrame) -> FittedFormula {
    let [model] = fit_all(source, data, FitOptions::default())
        .try_into()
        .unwrap_or_else(|models: Vec<_>| {
            panic!("{source}: expected one model, got {}", models.len())
        });
    model
}

fn names(model: &FittedFormula, role: Role) -> Vec<String> {
    model
        .select(&[role])
        .unwrap()
        .frame()
        .unwrap()
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect()
}

fn warned_terms(model: &FittedFormula) -> Vec<String> {
    model
        .rank_deficiencies()
        .into_iter()
        .flat_map(|deficiency| deficiency.terms().to_vec())
        .collect()
}

fn values(column: &polars::prelude::Column) -> Vec<f64> {
    column
        .cast(&DataType::Float64)
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .map(Option::unwrap)
        .collect()
}

/// Rank of a set of columns by Gaussian elimination with partial pivoting.
fn rank(columns: Vec<Vec<f64>>) -> usize {
    // Rows of the transposed matrix are the model columns.
    let mut rows = columns;
    let width = rows.first().map_or(0, Vec::len);
    let mut rank = 0;
    for column in 0..width {
        let Some(pivot) = (rank..rows.len()).max_by(|&left, &right| {
            rows[left][column]
                .abs()
                .total_cmp(&rows[right][column].abs())
        }) else {
            break;
        };
        if rows[pivot][column].abs() < 1e-9 {
            continue;
        }
        rows.swap(rank, pivot);
        let pivot = rows[rank].clone();
        for (index, row) in rows.iter_mut().enumerate() {
            if index != rank {
                let factor = row[column] / pivot[column];
                for (value, pivot) in row.iter_mut().zip(&pivot) {
                    *value -= factor * pivot;
                }
            }
        }
        rank += 1;
    }
    rank
}

/// The dummy columns a fixed effect stands for: group indicators and group slopes.
fn fixed_effect_columns(model: &FittedFormula) -> Vec<Vec<f64>> {
    let mut columns = Vec::new();
    let Ok(terms) = model.terms(Role::FixedEffect) else {
        return columns;
    };
    for term in terms {
        let effect = term.as_fixed_effect().unwrap();
        let codes = values(effect.codes());
        let groups = codes.iter().fold(0.0_f64, |max, &code| max.max(code)) as usize + 1;
        for group in 0..groups {
            let indicator = codes
                .iter()
                .map(|&code| f64::from(code as usize == group))
                .collect::<Vec<_>>();
            if effect.intercept() {
                columns.push(indicator.clone());
            }
            for loading in effect.loadings().columns() {
                columns.push(
                    indicator
                        .iter()
                        .zip(values(loading))
                        .map(|(indicator, loading)| indicator * loading)
                        .collect(),
                );
            }
        }
    }
    columns
}

/// The predictor columns, and the dummy columns the fixed effects stand for.
fn design(model: &FittedFormula) -> Vec<Vec<f64>> {
    let frame = model.select(&[Role::Independent]).unwrap().frame().unwrap();
    let mut columns = frame.columns().iter().map(values).collect::<Vec<_>>();
    columns.extend(fixed_effect_columns(model));
    columns
}

fn assert_full_rank(model: &FittedFormula, source: &str) {
    let design = design(model);
    assert_eq!(rank(design.clone()), design.len(), "{source}");
}

/// The 14 right-hand sides with and without an intercept, in this crate's naming.
///
/// Columns, column groups and their order are Formulaic 1.2.2's, with `[T.level]`
/// for reduced and `[level]` for full factors. Within a column group, the first
/// factor varies slowest, as everywhere in this crate (Formulaic varies it fastest).
const GRID: &[(&str, &[&str])] = &[
    ("1 + a", &["Intercept", "a[T.q]", "a[T.r]"]),
    ("0 + a", &["a[p]", "a[q]", "a[r]"]),
    ("1 + a + b", &["Intercept", "a[T.q]", "a[T.r]", "b[T.v]"]),
    ("0 + a + b", &["a[p]", "a[q]", "a[r]", "b[T.v]"]),
    (
        "1 + a:b",
        &[
            "Intercept",
            "b[T.v]",
            "a[T.q]:b[u]",
            "a[T.q]:b[v]",
            "a[T.r]:b[u]",
            "a[T.r]:b[v]",
        ],
    ),
    (
        "0 + a:b",
        &[
            "a[p]:b[u]",
            "a[p]:b[v]",
            "a[q]:b[u]",
            "a[q]:b[v]",
            "a[r]:b[u]",
            "a[r]:b[v]",
        ],
    ),
    (
        "1 + a*b",
        &[
            "Intercept",
            "a[T.q]",
            "a[T.r]",
            "b[T.v]",
            "a[T.q]:b[T.v]",
            "a[T.r]:b[T.v]",
        ],
    ),
    (
        "0 + a*b",
        &[
            "a[p]",
            "a[q]",
            "a[r]",
            "b[T.v]",
            "a[T.q]:b[T.v]",
            "a[T.r]:b[T.v]",
        ],
    ),
    (
        "1 + a + a:b",
        &[
            "Intercept",
            "a[T.q]",
            "a[T.r]",
            "a[p]:b[T.v]",
            "a[q]:b[T.v]",
            "a[r]:b[T.v]",
        ],
    ),
    (
        "0 + a + a:b",
        &[
            "a[p]",
            "a[q]",
            "a[r]",
            "a[p]:b[T.v]",
            "a[q]:b[T.v]",
            "a[r]:b[T.v]",
        ],
    ),
    (
        "1 + b + a:b",
        &[
            "Intercept",
            "b[T.v]",
            "a[T.q]:b[u]",
            "a[T.q]:b[v]",
            "a[T.r]:b[u]",
            "a[T.r]:b[v]",
        ],
    ),
    (
        "0 + b + a:b",
        &[
            "b[u]",
            "b[v]",
            "a[T.q]:b[u]",
            "a[T.q]:b[v]",
            "a[T.r]:b[u]",
            "a[T.r]:b[v]",
        ],
    ),
    ("1 + a:x", &["Intercept", "a[p]:x", "a[q]:x", "a[r]:x"]),
    ("0 + a:x", &["a[p]:x", "a[q]:x", "a[r]:x"]),
    ("1 + x + a:x", &["Intercept", "x", "a[T.q]:x", "a[T.r]:x"]),
    ("0 + x + a:x", &["x", "a[T.q]:x", "a[T.r]:x"]),
    (
        "1 + a + a:x",
        &[
            "Intercept",
            "a[T.q]",
            "a[T.r]",
            "a[p]:x",
            "a[q]:x",
            "a[r]:x",
        ],
    ),
    (
        "0 + a + a:x",
        &["a[p]", "a[q]", "a[r]", "a[p]:x", "a[q]:x", "a[r]:x"],
    ),
    (
        "1 + a*x",
        &["Intercept", "a[T.q]", "a[T.r]", "x", "a[T.q]:x", "a[T.r]:x"],
    ),
    (
        "0 + a*x",
        &["a[p]", "a[q]", "a[r]", "x", "a[T.q]:x", "a[T.r]:x"],
    ),
    (
        "1 + a + b:c",
        &[
            "Intercept",
            "a[T.q]",
            "a[T.r]",
            "c[T.n]",
            "b[T.v]:c[m]",
            "b[T.v]:c[n]",
        ],
    ),
    (
        "0 + a + b:c",
        &[
            "a[p]",
            "a[q]",
            "a[r]",
            "c[T.n]",
            "b[T.v]:c[m]",
            "b[T.v]:c[n]",
        ],
    ),
    (
        "1 + a:b:c",
        &[
            "Intercept",
            "a[T.q]:b[u]",
            "a[T.q]:b[v]",
            "a[T.r]:b[u]",
            "a[T.r]:b[v]",
            "a[p]:c[T.n]",
            "a[q]:c[T.n]",
            "a[r]:c[T.n]",
            "b[T.v]:c[m]",
            "b[T.v]:c[n]",
            "a[T.q]:b[T.v]:c[T.n]",
            "a[T.r]:b[T.v]:c[T.n]",
        ],
    ),
    (
        "0 + a:b:c",
        &[
            "a[p]:b[u]:c[m]",
            "a[p]:b[u]:c[n]",
            "a[p]:b[v]:c[m]",
            "a[p]:b[v]:c[n]",
            "a[q]:b[u]:c[m]",
            "a[q]:b[u]:c[n]",
            "a[q]:b[v]:c[m]",
            "a[q]:b[v]:c[n]",
            "a[r]:b[u]:c[m]",
            "a[r]:b[u]:c[n]",
            "a[r]:b[v]:c[m]",
            "a[r]:b[v]:c[n]",
        ],
    ),
    (
        "1 + a:b + a:b:c",
        &[
            "Intercept",
            "b[T.v]",
            "a[T.q]:b[u]",
            "a[T.q]:b[v]",
            "a[T.r]:b[u]",
            "a[T.r]:b[v]",
            "a[p]:b[u]:c[T.n]",
            "a[p]:b[v]:c[T.n]",
            "a[q]:b[u]:c[T.n]",
            "a[q]:b[v]:c[T.n]",
            "a[r]:b[u]:c[T.n]",
            "a[r]:b[v]:c[T.n]",
        ],
    ),
    (
        "0 + a:b + a:b:c",
        &[
            "a[p]:b[u]",
            "a[p]:b[v]",
            "a[q]:b[u]",
            "a[q]:b[v]",
            "a[r]:b[u]",
            "a[r]:b[v]",
            "a[p]:b[u]:c[T.n]",
            "a[p]:b[v]:c[T.n]",
            "a[q]:b[u]:c[T.n]",
            "a[q]:b[v]:c[T.n]",
            "a[r]:b[u]:c[T.n]",
            "a[r]:b[v]:c[T.n]",
        ],
    ),
    (
        "1 + a*b*c",
        // Formulaic sorts terms by degree; this crate keeps the expansion order of
        // `(a*b)*c`, so `c` follows `a:b`. The columns of each term are the same.
        &[
            "Intercept",
            "a[T.q]",
            "a[T.r]",
            "b[T.v]",
            "a[T.q]:b[T.v]",
            "a[T.r]:b[T.v]",
            "c[T.n]",
            "a[T.q]:c[T.n]",
            "a[T.r]:c[T.n]",
            "b[T.v]:c[T.n]",
            "a[T.q]:b[T.v]:c[T.n]",
            "a[T.r]:b[T.v]:c[T.n]",
        ],
    ),
    (
        "0 + a*b*c",
        // Formulaic sorts terms by degree; this crate keeps the expansion order of
        // `(a*b)*c`, so `c` follows `a:b`. The columns of each term are the same.
        &[
            "a[p]",
            "a[q]",
            "a[r]",
            "b[T.v]",
            "a[T.q]:b[T.v]",
            "a[T.r]:b[T.v]",
            "c[T.n]",
            "a[T.q]:c[T.n]",
            "a[T.r]:c[T.n]",
            "b[T.v]:c[T.n]",
            "a[T.q]:b[T.v]:c[T.n]",
            "a[T.r]:b[T.v]:c[T.n]",
        ],
    ),
];

/// Spells the grid's categorical factors as `C(...)` instead of bare columns.
fn explicit(source: &str) -> String {
    let mut output = String::new();
    for character in source.chars() {
        match character {
            'a' | 'b' | 'c' => output.push_str(&format!("C({character})")),
            other => output.push(other),
        }
    }
    output
}

#[test]
fn grid_matches_formulaic_and_is_full_rank() {
    let data = grid_data();
    assert_eq!(GRID.len(), 28);
    for &(rhs, expected) in GRID {
        for (source, expected) in [
            (
                format!("y ~ {rhs}"),
                expected
                    .iter()
                    .map(|name| name.to_string())
                    .collect::<Vec<_>>(),
            ),
            (
                format!("y ~ {}", explicit(rhs)),
                expected
                    .iter()
                    .map(|name| {
                        if *name == "Intercept" {
                            name.to_string()
                        } else {
                            explicit(name)
                        }
                    })
                    .collect(),
            ),
        ] {
            let model = fit(&source, &data);
            assert_eq!(names(&model, Role::Independent), expected, "{source}");
            assert!(model.rank_deficiencies().is_empty(), "{source}");
            assert_full_rank(&model, &source);
        }
    }
}

#[test]
fn one_term_emits_several_column_groups() {
    let data = grid_data();
    let model = fit("y ~ a:b:c", &data);
    let terms = model.terms(Role::Independent).unwrap();
    assert_eq!(terms.len(), 2);
    // All column groups belong to the one term `a:b:c`.
    let columns = terms[1].columns().collect::<Vec<_>>();
    assert_eq!(columns.len(), 11);
    for column in &columns {
        let term = column
            .term()
            .factors()
            .map(|factor| factor.name().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(term, ["a", "b", "c"]);
    }
}

#[test]
fn absorbed_terms_name_the_formula_and_the_terms() {
    let data = grid_data();
    let model = fit("y ~ C(a) + a + x + C(f):C(g) + f:g", &data);
    let [deficiency] = model.rank_deficiencies().try_into().unwrap();
    assert_eq!(deficiency.formula(), model.formula().to_string());
    assert_eq!(deficiency.terms(), ["a", "f:g"]);
    let message = deficiency.to_string();
    assert!(message.contains("y ~ 1 + C(a) + a + x"), "{message}");
    assert!(
        message.contains("terms a, f:g are fully spanned"),
        "{message}"
    );
    let model = fit("y ~ a + C(a)", &data);
    let message = model.rank_deficiencies()[0].to_string();
    assert!(message.contains("term C(a) is fully spanned"), "{message}");
    assert!(message.contains("has no columns"), "{message}");
}

#[test]
fn absorbed_terms_have_no_columns() {
    let data = grid_data();
    for (source, expected, absorbed) in [
        // `a` and `C(a)` are one factor.
        (
            "y ~ a + C(a)",
            &["Intercept", "a[T.q]", "a[T.r]"][..],
            "C(a)",
        ),
        (
            "y ~ C(a) + C(a, omit='q')",
            &["Intercept", "C(a)[T.q]", "C(a)[T.r]"],
            "C(a, omit='q')",
        ),
        // The grouping variable is the same factor as its categorical spellings.
        ("y ~ C(f) | f", &[], "C(f)"),
        ("y ~ f | f", &[], "f"),
        // Loadings span their variable within and across groups.
        ("y ~ x | f[[x]]", &["Intercept"], "x"),
        ("y ~ C(f):x | f[x]", &[], "C(f):x"),
        ("y ~ log(x) | f[[log(x)]]", &["Intercept"], "log(x)"),
    ] {
        let model = fit(source, &data);
        assert_eq!(names(&model, Role::Independent), expected, "{source}");
        assert_eq!(warned_terms(&model), [absorbed], "{source}");
        assert_full_rank(&model, source);

        // The term is kept, without columns, through selection and transformation.
        let terms = model.terms(Role::Independent).unwrap();
        let term = terms.last().unwrap();
        assert_eq!(term.columns().count(), 0, "{source}");
        let selected = model.select_terms(absorbed, Role::Independent).unwrap();
        assert_eq!(selected.terms(Role::Independent).unwrap().len(), 1);
        let frame = selected.frame().unwrap();
        assert_eq!(frame.shape(), (data.height(), 0), "{source}");
        let applied = selected.transform(&data.slice(2, 3)).unwrap();
        assert_eq!(applied.frame().unwrap().shape(), (3, 0), "{source}");
        let applied = model.transform(&data.slice(2, 3)).unwrap();
        assert_eq!(names(&applied, Role::Independent), expected, "{source}");
        assert_eq!(applied.rank_deficiencies(), model.rank_deficiencies());
    }
}

#[test]
fn explicit_omissions_restrict_the_fitted_level_space() {
    let data = grid_data();
    for (source, expected) in [
        ("y ~ C(a)", &["Intercept", "C(a)[T.q]", "C(a)[T.r]"][..]),
        (
            "y ~ C(a, omit='q')",
            &["Intercept", "C(a, omit='q')[p]", "C(a, omit='q')[r]"],
        ),
        (
            "y ~ C(a, omit='q') - 1",
            &["C(a, omit='q')[p]", "C(a, omit='q')[r]"],
        ),
        (
            "y ~ C(a, omit='q') + C(b, omit='v') - 1",
            &[
                "C(a, omit='q')[p]",
                "C(a, omit='q')[r]",
                "C(b, omit='v')[u]",
            ],
        ),
        (
            "y ~ C(a, omit='q'):C(b, omit='v')",
            &[
                "Intercept",
                "C(a, omit='q')[p]:C(b, omit='v')[u]",
                "C(a, omit='q')[r]:C(b, omit='v')[u]",
            ],
        ),
        ("y ~ a - 1", &["a[p]", "a[q]", "a[r]"]),
    ] {
        let model = fit(source, &data);
        assert_eq!(names(&model, Role::Independent), expected, "{source}");
        assert_full_rank(&model, source);
    }
}

#[test]
fn fixed_effects_seed_the_span_with_their_groups_and_loadings() {
    let data = grid_data();
    for (source, expected) in [
        // A group with intercepts spans the constant, also where the intercept is stripped.
        ("y ~ C(a) | f", &["C(a)[T.q]", "C(a)[T.r]"][..]),
        ("y ~ 0 + C(a) | f", &["C(a)[T.q]", "C(a)[T.r]"]),
        ("y ~ C(a) | f:g", &["C(a)[T.q]", "C(a)[T.r]"]),
        // Slopes alone do not span the constant, but the global intercept remains.
        (
            "y ~ C(a) | f[[x]]",
            &["Intercept", "C(a)[T.q]", "C(a)[T.r]"],
        ),
        ("y ~ 0 + C(a) | f[[x]]", &["C(a)[p]", "C(a)[q]", "C(a)[r]"]),
        // Loadings span their variable within and across groups.
        (
            "y ~ C(a):x | f[[x]]",
            &["Intercept", "C(a)[T.q]:x", "C(a)[T.r]:x"],
        ),
        ("y ~ C(a):x | f[x]", &["C(a)[T.q]:x", "C(a)[T.r]:x"]),
        // `a` is reduced within the levels of the fixed effect.
        (
            "y ~ C(a):C(f) | f",
            &[
                "C(a)[T.q]:C(f)[k]",
                "C(a)[T.q]:C(f)[l]",
                "C(a)[T.r]:C(f)[k]",
                "C(a)[T.r]:C(f)[l]",
            ],
        ),
        // An interaction alongside a fixed effect emits only what it adds.
        (
            "y ~ a:b | f",
            &[
                "b[T.v]",
                "a[T.q]:b[u]",
                "a[T.q]:b[v]",
                "a[T.r]:b[u]",
                "a[T.r]:b[v]",
            ],
        ),
    ] {
        let model = fit(source, &data);
        assert_eq!(names(&model, Role::Independent), expected, "{source}");
        assert!(model.rank_deficiencies().is_empty(), "{source}");
        assert_full_rank(&model, source);
    }
    // Redundancy among fixed effects is left to the solver.
    let model = fit("y ~ C(a) | f + g + f:g", &data);
    assert_eq!(names(&model, Role::Independent), ["C(a)[T.q]", "C(a)[T.r]"]);
    assert!(model.rank_deficiencies().is_empty());
    let model = fit("y ~ x | f + f:g + f[[x]]", &data);
    assert!(names(&model, Role::Independent).is_empty());
    assert_eq!(warned_terms(&model), ["x"]);
}

#[test]
fn multiverse_codes_one_term_differently_per_expression() {
    let data = grid_data();
    let batches: [(Vec<FittedFormula>, [&[&str]; 2]); 3] = [
        (
            fit_all(
                "y ~ sw(C(a), C(b)) + C(a):C(b)",
                &data,
                FitOptions::default(),
            ),
            [
                &[
                    "Intercept",
                    "C(a)[T.q]",
                    "C(a)[T.r]",
                    "C(a)[p]:C(b)[T.v]",
                    "C(a)[q]:C(b)[T.v]",
                    "C(a)[r]:C(b)[T.v]",
                ],
                &[
                    "Intercept",
                    "C(b)[T.v]",
                    "C(a)[T.q]:C(b)[u]",
                    "C(a)[T.q]:C(b)[v]",
                    "C(a)[T.r]:C(b)[u]",
                    "C(a)[T.r]:C(b)[v]",
                ],
            ],
        ),
        // One coding has a single column group, the other two.
        (
            fit_all("y ~ sw0(C(a)) + C(a):C(b)", &data, FitOptions::default()),
            [
                &[
                    "Intercept",
                    "C(b)[T.v]",
                    "C(a)[T.q]:C(b)[u]",
                    "C(a)[T.q]:C(b)[v]",
                    "C(a)[T.r]:C(b)[u]",
                    "C(a)[T.r]:C(b)[v]",
                ],
                &[
                    "Intercept",
                    "C(a)[T.q]",
                    "C(a)[T.r]",
                    "C(a)[p]:C(b)[T.v]",
                    "C(a)[q]:C(b)[T.v]",
                    "C(a)[r]:C(b)[T.v]",
                ],
            ],
        ),
        (
            formula::parse_many(["y ~ C(a)", "y ~ C(a) - 1"])
                .unwrap()
                .fit(&data)
                .unwrap()
                .iter()
                .cloned()
                .collect(),
            [
                &["Intercept", "C(a)[T.q]", "C(a)[T.r]"],
                &["C(a)[p]", "C(a)[q]", "C(a)[r]"],
            ],
        ),
    ];
    let input = data.slice(3, 5);
    for (models, expected) in batches {
        assert_eq!(models.len(), 2);
        let fitted = models.iter().cloned().collect::<formula::FittedFormulas>();
        let transformed = fitted.transform(&input).unwrap();
        for ((original, applied), expected) in models.iter().zip(transformed.iter()).zip(expected) {
            assert_eq!(names(original, Role::Independent), expected);
            assert!(original.rank_deficiencies().is_empty());
            assert_full_rank(original, &original.formula().to_string());
            assert_eq!(
                names(applied, Role::Independent),
                names(original, Role::Independent)
            );
            let applied = applied
                .select(&[Role::Independent])
                .unwrap()
                .frame()
                .unwrap();
            let original = original
                .select(&[Role::Independent])
                .unwrap()
                .frame()
                .unwrap();
            assert!(applied.equals_missing(&original.slice(3, 5)));
        }
    }
}

#[test]
fn dependent_categoricals_keep_every_level() {
    let data = grid_data();
    let model = fit("C(a) ~ C(a)", &data);
    assert_eq!(
        names(&model, Role::Dependent),
        ["C(a)[p]", "C(a)[q]", "C(a)[r]"]
    );
    assert_eq!(
        names(&model, Role::Independent),
        ["Intercept", "C(a)[T.q]", "C(a)[T.r]"]
    );
    let model = fit("C(a):C(b) ~ x", &data);
    assert_eq!(names(&model, Role::Dependent).len(), 6);
    let model = fit("C(a, omit='q') ~ x", &data);
    assert_eq!(
        names(&model, Role::Dependent),
        ["C(a, omit='q')[p]", "C(a, omit='q')[r]",]
    );
}

#[test]
fn bare_string_predictors_are_full_rank_by_default() {
    let data = polars::df!(
        "y" => [1.0, 2.0, 3.0, 4.0], "x" => [5.0, 6.0, 8.0, 7.0],
        "state" => ["NY", "CA", "TX", "CA"],
    )
    .unwrap();
    let model = fit("y ~ x + state", &data);
    assert_eq!(
        names(&model, Role::Independent),
        ["Intercept", "x", "state[T.NY]", "state[T.TX]"]
    );
    assert_full_rank(&model, "y ~ x + state");
}

#[test]
fn opting_out_still_honors_explicit_omissions() {
    let data = grid_data();
    let options = FitOptions {
        ensure_full_rank: false,
        ..FitOptions::default()
    };
    for (source, expected) in [
        (
            "y ~ C(a)",
            &["Intercept", "C(a)[p]", "C(a)[q]", "C(a)[r]"][..],
        ),
        ("y ~ a", &["Intercept", "a[p]", "a[q]", "a[r]"]),
        (
            "y ~ C(a, omit='q')",
            &["Intercept", "C(a, omit='q')[p]", "C(a, omit='q')[r]"],
        ),
        (
            "y ~ C(a, omit='q') - 1",
            &["C(a, omit='q')[p]", "C(a, omit='q')[r]"],
        ),
        ("y ~ C(a) | c", &["C(a)[p]", "C(a)[q]", "C(a)[r]"]),
        (
            "y ~ a + C(a)",
            &[
                "Intercept",
                "a[p]",
                "a[q]",
                "a[r]",
                "C(a)[p]",
                "C(a)[q]",
                "C(a)[r]",
            ],
        ),
        ("y ~ x | f[[x]]", &["Intercept", "x"]),
        (
            "y ~ a:b",
            &[
                "Intercept",
                "a[p]:b[u]",
                "a[p]:b[v]",
                "a[q]:b[u]",
                "a[q]:b[v]",
                "a[r]:b[u]",
                "a[r]:b[v]",
            ],
        ),
    ] {
        let [model] = fit_all(source, &data, options).try_into().unwrap();
        assert_eq!(names(&model, Role::Independent), expected, "{source}");
        assert!(model.rank_deficiencies().is_empty(), "{source}");
    }
    assert!(FitOptions::default().ensure_full_rank);
}

#[test]
fn transformation_reuses_the_fitted_coding() {
    let data = grid_data();
    let model = fit("y ~ x + C(a)", &data);
    // The selection has no intercept, but the term keeps the coding it was fitted with.
    let selected = model.select_terms("C(a)", Role::Independent).unwrap();
    assert_eq!(
        names(&selected, Role::Independent),
        ["C(a)[T.q]", "C(a)[T.r]"]
    );
    let input = polars::df!("y" => [0.0, 0.0], "x" => [1.0, 2.0], "a" => ["p", "r"]).unwrap();
    for applied in [
        selected.transform(&input).unwrap(),
        model.transform(&input).unwrap(),
    ] {
        let frame = applied
            .select_terms("C(a)", Role::Independent)
            .unwrap()
            .frame()
            .unwrap();
        assert_eq!(frame.get_column_names(), ["C(a)[T.q]", "C(a)[T.r]"]);
        assert_eq!(
            frame[1].bool().unwrap().iter().collect::<Vec<_>>(),
            [Some(false), Some(true)]
        );
    }
    let applied = model.transform(&input).unwrap();
    assert_eq!(
        names(&applied, Role::Independent),
        ["Intercept", "x", "C(a)[T.q]", "C(a)[T.r]"]
    );
}

#[test]
fn column_groups_round_trip_through_selection_and_transformation() {
    let data = grid_data();
    let model = fit("y ~ x + a:b", &data);
    let expected = [
        "b[T.v]",
        "a[T.q]:b[u]",
        "a[T.q]:b[v]",
        "a[T.r]:b[u]",
        "a[T.r]:b[v]",
    ];
    // Selection returns every column group of the term, in either factor order.
    for query in ["a:b", "b:a"] {
        let selected = model.select_terms(query, Role::Independent).unwrap();
        assert_eq!(names(&selected, Role::Independent), expected);
    }
    let selected = model.select_terms("a:b", Role::Independent).unwrap();
    let input = data.slice(5, 7);
    let applied = selected.transform(&input).unwrap().frame().unwrap();
    assert!(applied.equals_missing(&selected.frame().unwrap().slice(5, 7)));
    let applied = model.transform(&input).unwrap();
    let applied = applied
        .select(&[Role::Independent])
        .unwrap()
        .frame()
        .unwrap();
    let original = model.select(&[Role::Independent]).unwrap().frame().unwrap();
    assert!(applied.equals_missing(&original.slice(5, 7)));
    assert_eq!(
        values(&applied["b[T.v]"]),
        values(&original["b[T.v]"].slice(5, 7))
    );
}

/// Each column's contributions as `factor=level`, with an empty level for numeric factors.
fn contributions(model: &FittedFormula) -> Vec<String> {
    model
        .columns(Role::Independent)
        .unwrap()
        .map(|column| {
            column
                .contributions()
                .map(|(factor, key)| {
                    let name = factor
                        .name()
                        .map(str::to_owned)
                        .unwrap_or_else(|| factor.transform().unwrap().name().to_owned());
                    let level = match key {
                        ComponentKey::Scalar(level) => {
                            level.as_any_value().str_value().into_owned()
                        }
                        _ => String::new(),
                    };
                    format!("{name}={level}")
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

#[test]
fn columns_attribute_only_the_factors_they_include() {
    let data = grid_data();
    let model = fit("y ~ a:x + C(b)", &data);
    assert_eq!(
        contributions(&model),
        ["", "a=p,x=", "a=q,x=", "a=r,x=", "C=v"]
    );

    // `b[T.v]` omits `a` from its term, so it lists only `b`.
    let model = fit("y ~ a:b", &data);
    assert_eq!(
        contributions(&model),
        ["", "b=v", "a=q,b=u", "a=q,b=v", "a=r,b=u", "a=r,b=v"]
    );
    let columns = model
        .columns(Role::Independent)
        .unwrap()
        .collect::<Vec<_>>();
    let term = columns[1]
        .term()
        .factors()
        .map(|factor| factor.name().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(term, ["a", "b"]);

    // The term `b:c` gives `c[T.n]` and `b[T.v]:c[...]`, as in Formulaic.
    let model = fit("y ~ 0 + C(a) + b:c", &data);
    assert_eq!(
        contributions(&model),
        ["C=p", "C=q", "C=r", "c=n", "b=v,c=m", "b=v,c=n"]
    );
}

#[test]
fn categorical_by_numeric_interactions_multiply_reduced_indicators() {
    let data = polars::df!(
        "y" => [1.0, 2.0, 3.0], "x" => [2.0, 3.0, 4.0], "a" => ["p", "q", "r"],
    )
    .unwrap();
    let model = fit("y ~ x + a:x", &data);
    let frame = model.select(&[Role::Independent]).unwrap().frame().unwrap();
    assert_eq!(
        frame.get_column_names(),
        ["Intercept", "x", "a[T.q]:x", "a[T.r]:x"]
    );
    assert_eq!(
        frame[2].f64().unwrap().iter().collect::<Vec<_>>(),
        [Some(0.0), Some(3.0), Some(0.0)]
    );
}

#[test]
fn child_formulas_are_coded_and_flagged_on_their_own() {
    let data = grid_data();
    let model = fit("y ~ C(a) + [x ~ a + C(a)]", &data);
    assert_eq!(
        names(&model, Role::Independent),
        ["Intercept", "C(a)[T.q]", "C(a)[T.r]"]
    );
    let [child] = model.children(Role::Independent).unwrap() else {
        panic!("expected one child");
    };
    assert_eq!(
        names(child, Role::Independent),
        ["Intercept", "a[T.q]", "a[T.r]"]
    );
    assert_eq!(child.rank_deficiencies().len(), 1);
    let deficiencies = model.rank_deficiencies();
    assert_eq!(deficiencies.len(), 1);
    assert_eq!(deficiencies[0].formula(), child.formula().to_string());
}
