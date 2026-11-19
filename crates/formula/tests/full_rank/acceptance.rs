use super::*;
use formula::ParameterValue;

const A: [&str; 4] = ["p", "q", "r", "s"];
const B: [&str; 2] = ["u", "v"];

fn acceptance_grid() -> DataFrame {
    polars::df!(
        "y" => [0.0; 8],
        "a" => ["s", "r", "q", "p", "s", "r", "q", "p"],
        "b" => ["v", "v", "v", "v", "u", "u", "u", "u"],
        "x" => [4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
    )
    .unwrap()
}

fn level_indicator(data: &DataFrame, column: &str, level: &str) -> Vec<f64> {
    data[column]
        .str()
        .unwrap()
        .iter()
        .map(|value| f64::from(value.unwrap() == level))
        .collect()
}

fn numeric_level_indicators(x: &[f64]) -> Vec<Vec<f64>> {
    [1.0, 2.0, 3.0, 4.0]
        .map(|level| x.iter().map(|&value| f64::from(value == level)).collect())
        .to_vec()
}

fn product(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .collect()
}

fn cell_indicators(data: &DataFrame, levels: &[&str]) -> Vec<Vec<f64>> {
    levels
        .iter()
        .flat_map(|level_a| {
            let a = level_indicator(data, "a", level_a);
            B.map(|level_b| product(&a, &level_indicator(data, "b", level_b)))
        })
        .collect()
}

fn predictors(model: &FittedFormula) -> DataFrame {
    model.select(&[Role::Independent]).unwrap().frame().unwrap()
}

fn assert_span(model: &FittedFormula, expected: Vec<Vec<f64>>, source: &str) {
    let actual = design(model);
    let expected_rank = rank(expected.clone());
    assert_eq!(rank(actual.clone()), expected_rank, "{source}");
    assert_eq!(
        rank(actual.into_iter().chain(expected).collect()),
        expected_rank,
        "{source}: joint rank"
    );
}

#[test]
fn numeric_slopes_and_level_indicators_have_distinct_identities() {
    let data = acceptance_grid();
    let x = values(&data["x"]);
    let expected = numeric_level_indicators(&x);
    for intercept in [0, 1] {
        for terms in ["x + C(x)", "C(x) + x"] {
            let source = format!("y ~ {intercept} + {terms}");
            let model = fit(&source, &data);
            assert_span(&model, expected.clone(), &source);
            assert!(warned_terms(&model).is_empty(), "{source}");
            let frame = predictors(&model);
            assert_eq!(values(&frame["x"]), x, "{source}");
            if intercept == 0 {
                for (level, indicator) in (1..=4).zip(&expected) {
                    assert_eq!(
                        values(&frame[format!("C(x)[{}.0]", level).as_str()]),
                        *indicator
                    );
                }
            }
        }
    }
}

#[test]
fn repeated_categorical_dimension_is_materialized_once_and_survives_selection() {
    let data = acceptance_grid();
    for (term, spelling, other, contribution, bare_is_second) in [
        ("a:C(a)", "a", "C(a):a", "a", false),
        ("C(a):a", "C(a)", "a:C(a)", "C", true),
    ] {
        for intercept in [false, true] {
            let source = format!("y ~ {} + {term}", u8::from(intercept));
            let model = fit(&source, &data);
            let expected = A.map(|level| level_indicator(&data, "a", level));
            assert_span(&model, expected.to_vec(), &source);
            let frame = predictors(&model);
            assert_eq!(frame.width(), 4, "{source}");
            let selected = model.select_terms(other, Role::Independent).unwrap();
            let selected_terms = selected.terms(Role::Independent).unwrap();
            let factors = selected_terms[0].factors().collect::<Vec<_>>();
            assert_eq!(factors.len(), 2);
            let (bare, explicit) = if bare_is_second {
                (factors[1], factors[0])
            } else {
                (factors[0], factors[1])
            };
            assert_eq!(bare.name(), Some("a"));
            let call = explicit.transform().unwrap();
            assert_eq!(call.name(), "C");
            assert_eq!(call.inputs().next().unwrap().name(), Some("a"));

            let prediction_data = polars::df!("a" => ["r", "p", "p", "s"]).unwrap();
            let prediction = selected.transform(&prediction_data).unwrap();
            let fitted_columns = predictors(&selected);
            let prediction_columns = predictors(&prediction);
            let levels = &A[usize::from(intercept)..];
            let expected_names = levels
                .iter()
                .map(|level| {
                    let treatment = if intercept { "T." } else { "" };
                    format!("{spelling}[{treatment}{level}]")
                })
                .collect::<Vec<_>>();
            assert_eq!(
                names(&selected, Role::Independent),
                expected_names,
                "{source}"
            );
            assert_eq!(
                names(&prediction, Role::Independent),
                expected_names,
                "{source}"
            );
            let expected_contributions = levels
                .iter()
                .map(|level| format!("{contribution}={level}"))
                .collect::<Vec<_>>();
            assert_eq!(contributions(&selected), expected_contributions);
            assert_eq!(contributions(&prediction), expected_contributions);
            for (index, level) in levels.iter().enumerate() {
                assert_eq!(
                    values(&fitted_columns[index]),
                    level_indicator(&data, "a", level)
                );
                assert_eq!(
                    values(&prediction_columns[index]),
                    level_indicator(&prediction_data, "a", level)
                );
            }
        }
    }
}

#[test]
fn repeated_categorical_dimension_intersects_hard_omissions() {
    let data = acceptance_grid();
    let source = "y ~ 0 + C(a, omit='p'):C(a, omit='q')";
    let model = fit(source, &data);
    let expected = ["r", "s"].map(|level| level_indicator(&data, "a", level));
    assert_span(&model, expected.to_vec(), source);
    assert_eq!(
        names(&model, Role::Independent),
        ["C(a, omit='p')[r]", "C(a, omit='p')[s]"]
    );
    assert_eq!(contributions(&model), ["C=r", "C=s"]);
    let frame = predictors(&model);
    assert_eq!(values(&frame[0]), expected[0]);
    assert_eq!(values(&frame[1]), expected[1]);
    let terms = model.terms(Role::Independent).unwrap();
    let factors = terms[0].factors().collect::<Vec<_>>();
    assert_eq!(factors.len(), 2);
    for (factor, omitted) in factors.into_iter().zip(["p", "q"]) {
        let parameters = factor.transform().unwrap().parameters();
        let ParameterValue::Scalar(value) = &parameters["omit"] else {
            panic!("expected the original scalar omission");
        };
        assert_eq!(value.as_any_value().str_value(), omitted);
    }

    let selected = model
        .select_terms("C(a, omit='q'):C(a, omit='p')", Role::Independent)
        .unwrap();
    let prediction_data = polars::df!("a" => ["p", "r", "q", "s"]).unwrap();
    let prediction = selected.transform(&prediction_data).unwrap();
    assert_eq!(
        names(&prediction, Role::Independent),
        names(&model, Role::Independent)
    );
    assert_eq!(contributions(&prediction), contributions(&model));
    let frame = predictors(&prediction);
    assert_eq!(values(&frame[0]), [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(values(&frame[1]), [0.0, 0.0, 0.0, 1.0]);

    let [opt_out] = fit_all(
        source,
        &data,
        FitOptions {
            ensure_full_rank: false,
            ..FitOptions::default()
        },
    )
    .try_into()
    .unwrap();
    let frame = predictors(&opt_out);
    assert_eq!(frame.width(), 9);
    for (index, (left, right)) in ["q", "r", "s"]
        .into_iter()
        .flat_map(|left| ["p", "r", "s"].map(|right| (left, right)))
        .enumerate()
    {
        assert_eq!(
            values(&frame[index]),
            product(
                &level_indicator(&data, "a", left),
                &level_indicator(&data, "a", right)
            )
        );
    }
}

#[test]
fn additive_omission_masks_supply_only_their_missing_directions() {
    let data = acceptance_grid();
    let left = "C(a, omit=['p', 'q'])";
    let right = "C(a, omit=['q', 'r'])";
    for intercept in [0, 1] {
        for (first, second, levels) in [
            (left, right, &["p", "r", "s"][..]),
            (right, left, &["p", "r", "s"][..]),
            ("a", left, A.as_slice()),
            (left, "a", A.as_slice()),
        ] {
            let source = format!("y ~ {intercept} + {first} + {second}");
            let model = fit(&source, &data);
            let expected = if intercept == 1 { A.as_slice() } else { levels };
            assert_span(
                &model,
                expected
                    .iter()
                    .map(|level| level_indicator(&data, "a", level))
                    .collect(),
                &source,
            );
            assert_eq!(predictors(&model).width(), expected.len(), "{source}");
            if first != "a" {
                let (omissions, retained) = if first == left {
                    (["p", "q"], ["r", "s"])
                } else {
                    (["q", "r"], ["p", "s"])
                };
                let selected = model.select_terms(first, Role::Independent).unwrap();
                let frame = predictors(&selected);
                assert_eq!(frame.width(), retained.len(), "{source}");
                for (column, level) in frame.columns().iter().zip(retained) {
                    assert_eq!(column.name().as_str(), format!("{first}[{level}]"));
                    assert_eq!(values(column), level_indicator(&data, "a", level));
                }

                let terms = selected.terms(Role::Independent).unwrap();
                let parameters = terms[0]
                    .factors()
                    .next()
                    .unwrap()
                    .transform()
                    .unwrap()
                    .parameters();
                let ParameterValue::List(omitted) = &parameters["omit"] else {
                    panic!("expected the original list omission");
                };
                assert_eq!(
                    omitted
                        .iter()
                        .map(|value| value.as_any_value().str_value().into_owned())
                        .collect::<Vec<_>>(),
                    omissions
                );
            }
            let warnings = warned_terms(&model);
            if first == "a" {
                assert_eq!(warnings, [left], "{source}");
            } else {
                assert!(warnings.is_empty(), "{source}");
            }
        }
    }
}

#[test]
fn overlapping_omission_masks_compose_with_other_interaction_factors() {
    let data = acceptance_grid();
    let left = "C(a, omit=['p', 'q'])";
    let right = "C(a, omit=['q', 'r'])";
    for (first, second) in [(left, right), (right, left)] {
        for (prefix, levels) in [
            ("0", &["p", "r", "s"][..]),
            ("1", &["p", "r", "s"][..]),
            ("0 + b", A.as_slice()),
        ] {
            let source = format!("y ~ {prefix} + {first}:b + b:{second}");
            let model = fit(&source, &data);
            let mut expected = cell_indicators(&data, levels);
            if prefix == "1" {
                expected.push(vec![1.0; data.height()]);
            }
            assert_eq!(predictors(&model).width(), expected.len(), "{source}");
            assert_span(&model, expected, &source);
            assert!(warned_terms(&model).is_empty(), "{source}");
        }
        let source = format!("y ~ 0 + {first}:b:x + x:b:{second}");
        let model = fit(&source, &data);
        let slope = values(&data["x"]);
        let expected = cell_indicators(&data, &["p", "r", "s"])
            .iter()
            .map(|cell| product(cell, &slope))
            .collect::<Vec<_>>();
        assert_eq!(predictors(&model).width(), expected.len(), "{source}");
        assert_span(&model, expected, &source);
    }
}

#[test]
fn fixed_effects_have_distinct_coverage_for_group_inputs_and_slopes() {
    let data = acceptance_grid();
    let x = values(&data["x"]);
    let cells = numeric_level_indicators(&x);
    let slopes = B.map(|level| product(&level_indicator(&data, "b", level), &x));
    let mut intercept_and_slopes = vec![vec![1.0; data.height()]];
    intercept_and_slopes.extend(slopes.clone());

    let source = "y ~ C(x) | b[[x]]";
    let model = fit(source, &data);
    assert_span(
        &model,
        cells.iter().cloned().chain(slopes).collect(),
        source,
    );
    assert!(warned_terms(&model).is_empty());
    assert!(predictors(&model).width() > 1);

    let source = "y ~ x | b[[x]]";
    let model = fit(source, &data);
    assert_span(&model, intercept_and_slopes, source);
    assert_eq!(names(&model, Role::Independent), ["Intercept"]);
    assert_eq!(warned_terms(&model), ["x"]);

    for term in ["C(x)", "x"] {
        let source = format!("y ~ {term} | x");
        let model = fit(&source, &data);
        assert_span(&model, cells.clone(), &source);
        assert!(names(&model, Role::Independent).is_empty(), "{source}");
        assert_eq!(warned_terms(&model), [term], "{source}");
    }
    let source = "y ~ a:b | a + b";
    let model = fit(source, &data);
    assert_span(&model, cell_indicators(&data, &A), source);
    assert_eq!(predictors(&model).width(), 3);
    assert!(warned_terms(&model).is_empty());
}

#[test]
fn ordinary_interaction_values_follow_named_levels_and_numeric_inputs() {
    let data = grid_data();
    let a = |level| level_indicator(&data, "a", level);
    let b = |level| level_indicator(&data, "b", level);
    let constant = vec![1.0; data.height()];
    let x = values(&data["x"]);
    for (rhs, columns) in [
        (
            "a:b",
            vec![
                ("Intercept", constant.clone()),
                ("b[T.v]", b("v")),
                ("a[T.q]:b[u]", product(&a("q"), &b("u"))),
                ("a[T.q]:b[v]", product(&a("q"), &b("v"))),
                ("a[T.r]:b[u]", product(&a("r"), &b("u"))),
                ("a[T.r]:b[v]", product(&a("r"), &b("v"))),
            ],
        ),
        (
            "a*b",
            vec![
                ("Intercept", constant.clone()),
                ("a[T.q]", a("q")),
                ("a[T.r]", a("r")),
                ("b[T.v]", b("v")),
                ("a[T.q]:b[T.v]", product(&a("q"), &b("v"))),
                ("a[T.r]:b[T.v]", product(&a("r"), &b("v"))),
            ],
        ),
        (
            "x + a:x",
            vec![
                ("Intercept", constant.clone()),
                ("x", x.clone()),
                ("a[T.q]:x", product(&a("q"), &x)),
                ("a[T.r]:x", product(&a("r"), &x)),
            ],
        ),
    ] {
        for use_explicit in [false, true] {
            let spell = |value: &str| {
                if use_explicit && value != "Intercept" {
                    explicit(value)
                } else {
                    value.to_owned()
                }
            };
            let source = format!("y ~ {}", spell(rhs));
            let model = fit(&source, &data);
            let frame = predictors(&model);
            let expected_names = columns
                .iter()
                .map(|(name, _)| spell(name))
                .collect::<Vec<_>>();
            assert_eq!(names(&model, Role::Independent), expected_names, "{source}");
            assert_span(
                &model,
                columns.iter().map(|(_, values)| values.clone()).collect(),
                &source,
            );
            for (name, expected) in &columns {
                assert_eq!(
                    &values(&frame[spell(name).as_str()]),
                    expected,
                    "{source}: {name}"
                );
            }
        }
    }
}

/// `i()` spans what its `C()` spelling spans, with every column independent.
#[test]
fn interactions_span_their_categorical_spellings() {
    // `x` varies within every level of `a`, so slopes are not collinear by chance.
    let data = polars::df!(
        "y" => [0.0; 16],
        "a" => A.repeat(4),
        "b" => [B[0]; 8].into_iter().chain([B[1]; 8]).collect::<Vec<_>>(),
        "x" => (1..=16).map(|x| f64::from(x * x % 7) + 0.5).collect::<Vec<_>>(),
    )
    .unwrap();
    let opt_out = FitOptions {
        ensure_full_rank: false,
        ..FitOptions::default()
    };
    for (source, spelling) in [
        ("i(a)", "C(a)"),
        ("i(a, ref='p')", "C(a, omit='p')"),
        ("i(a, ref=['p', 'q'])", "C(a, omit=['p', 'q'])"),
        ("i(a, x)", "C(a):x"),
        ("x + i(a, x)", "x + C(a):x"),
        ("i(a, x, ref='q')", "C(a, omit='q'):x"),
        ("i(a, b)", "C(a):C(b)"),
        (
            "i(a, b, ref='p', ref2='u')",
            "C(a, omit='p'):C(b, omit='u')",
        ),
        ("i(a, C(b, omit='v'))", "C(a):C(b, omit='v')"),
        ("a + i(a, b)", "a + C(a):C(b)"),
        ("i(a) + i(a, b)", "C(a) + C(a):C(b)"),
        ("i(b, a)", "C(b):C(a)"),
    ] {
        for intercept in [0, 1] {
            let source = format!("y ~ {intercept} + {source}");
            let spelling = format!("y ~ {intercept} + {spelling}");
            let model = fit(&source, &data);
            assert_full_rank(&model, &source);
            let [cartesian] = fit_all(&spelling, &data, opt_out).try_into().unwrap();
            assert_span(&model, design(&cartesian), &source);
        }
    }
}

#[test]
fn interaction_bins_merge_levels_before_references() {
    let data = acceptance_grid();
    let model = fit("y ~ 0 + i(a, bin={'pq': ['p', 'q']}, ref='r')", &data);
    assert_eq!(names(&model, Role::Independent), ["a::pq", "a::s"]);
    let frame = predictors(&model);
    let pq = product(&level_indicator(&data, "a", "p"), &[1.0; 8])
        .iter()
        .zip(level_indicator(&data, "a", "q"))
        .map(|(p, q)| p + q)
        .collect::<Vec<_>>();
    assert_eq!(values(&frame["a::pq"]), pq);
    assert_eq!(values(&frame["a::s"]), level_indicator(&data, "a", "s"));
    // A binned variable has its own level space, reduced next to an intercept.
    let model = fit("y ~ 1 + i(a, bin={'pq': ['p', 'q']})", &data);
    assert_eq!(
        names(&model, Role::Independent),
        ["Intercept", "a::r", "a::s"]
    );
}

#[test]
fn interaction_contributions_key_each_axis() {
    let data = acceptance_grid();
    // An axis left out of a column group has no key.
    let model = fit("y ~ i(a, b)", &data);
    let column = model.columns(Role::Independent).unwrap().nth(1).unwrap();
    assert_eq!(column.data().name().as_str(), "b::v");
    let (_, key) = column.contributions().next().unwrap();
    assert_eq!(
        *key,
        ComponentKey::Tuple(vec![
            ComponentKey::None,
            ComponentKey::Scalar(polars::prelude::Scalar::from(
                polars::prelude::PlSmallStr::from("v")
            )),
        ])
    );
    // With `ref`, the explicit omission leaves no reduced group.
    let model = fit("y ~ i(a, b, ref='p') + i(a, x, ref='p')", &data);
    let keys = model
        .columns(Role::Independent)
        .unwrap()
        .map(|column| {
            let (factor, key) = column
                .contributions()
                .next()
                .map_or((None, ComponentKey::None), |(factor, key)| {
                    (factor.transform().map(|call| call.name()), key.clone())
                });
            (column.data().name().to_string(), factor, key)
        })
        .collect::<Vec<_>>();
    let level = |level: &str| {
        ComponentKey::Scalar(polars::prelude::Scalar::from(
            polars::prelude::PlSmallStr::from(level),
        ))
    };
    let pair = |a: Option<&str>, b: Option<&str>| {
        ComponentKey::Tuple(vec![
            a.map_or(ComponentKey::None, level),
            b.map_or(ComponentKey::None, level),
        ])
    };
    assert_eq!(
        keys,
        [
            ("Intercept".into(), None, ComponentKey::None),
            ("a::q:b::u".into(), Some("i"), pair(Some("q"), Some("u"))),
            ("a::q:b::v".into(), Some("i"), pair(Some("q"), Some("v"))),
            ("a::r:b::u".into(), Some("i"), pair(Some("r"), Some("u"))),
            ("a::r:b::v".into(), Some("i"), pair(Some("r"), Some("v"))),
            ("a::s:b::u".into(), Some("i"), pair(Some("s"), Some("u"))),
            ("a::s:b::v".into(), Some("i"), pair(Some("s"), Some("v"))),
            ("a::q:x".into(), Some("i"), pair(Some("q"), None)),
            ("a::r:x".into(), Some("i"), pair(Some("r"), None)),
            ("a::s:x".into(), Some("i"), pair(Some("s"), None)),
        ]
    );
}
