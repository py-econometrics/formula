use crate::fit::FitError;
use crate::formula::{
    Expression, FittedExpression, FittedTerm, Role, ValidationError, parse_one, transform_terms,
};
use crate::parse::{ParseResult, parse_term};
use crate::transform::TransformError;
use crate::{Formula, FormulaColumn};
use polars::error::PolarsResult;
use polars::prelude::DataFrame;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct FittedFormula {
    formula: Formula,
    expressions: BTreeMap<Role, FittedExpression>,
    height: usize,
}

impl FittedFormula {
    pub(crate) fn new(
        formula: Formula,
        expressions: BTreeMap<Role, FittedExpression>,
        height: usize,
    ) -> Self {
        debug_assert!(
            expressions
                .values()
                .all(|expression| expression.frame().height() == height)
        );
        Self {
            formula,
            expressions,
            height,
        }
    }

    pub fn formula(&self) -> &Formula {
        &self.formula
    }

    pub fn frame(&self) -> PolarsResult<DataFrame> {
        let columns = self
            .expressions
            .values()
            .flat_map(|expression| expression.frame().columns())
            .cloned()
            .collect();

        DataFrame::new(self.height, columns)
    }

    pub fn columns(
        &self,
        role: Role,
    ) -> Result<impl Iterator<Item = FormulaColumn<'_>>, SelectionError> {
        let expression = self
            .expression(role)
            .ok_or(SelectionError::MissingRole(role))?;

        Ok(expression.columns())
    }

    fn assemble(&self, height: usize, terms: &[FittedTerm]) -> Result<Self, ModelError> {
        let expressions = self
            .expressions
            .iter()
            .map(|(&role, transformed)| {
                transformed
                    .expression()
                    .assemble(height, terms)
                    .map(|expression| (role, expression))
            })
            .collect::<Result<BTreeMap<_, _>, ModelError>>()?;

        Ok(Self::new(self.formula.clone(), expressions, height))
    }

    pub fn transform(&self, data: &DataFrame) -> Result<Self, ModelError> {
        let expressions = self.all_expressions();
        let terms = transform_terms(&expressions, data)?;

        self.assemble(data.height(), &terms)
    }

    pub(crate) fn expression(&self, role: Role) -> Option<&FittedExpression> {
        self.expressions
            .get(&role)
            .filter(|transformed| !transformed.expression().is_empty())
    }

    pub fn select(&self, roles: &[Role]) -> Result<Self, SelectionError> {
        if roles.is_empty() {
            return Err(SelectionError::EmptySelection);
        }

        let expressions = roles
            .iter()
            .map(|&role| {
                self.expression(role)
                    .cloned()
                    .map(|expression| (role, expression))
                    .ok_or(SelectionError::MissingRole(role))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;

        Ok(Self::new(self.formula.clone(), expressions, self.height))
    }

    pub fn select_terms(&self, source: &str, role: Role) -> Result<Self, SelectionError> {
        let expression = self
            .expression(role)
            .ok_or(SelectionError::MissingRole(role))?;

        let query = parse_term(source, role)
            .map_err(|error| SelectionError::InvalidQuery(error.to_string()))?;

        let selected = expression.select_terms(&query)?;

        Ok(Self::new(
            self.formula.clone(),
            BTreeMap::from([(role, selected)]),
            self.height,
        ))
    }

    pub fn children(&self, role: Role) -> Result<&[Self], SelectionError> {
        self.expression(role)
            .map(FittedExpression::children)
            .ok_or(SelectionError::MissingRole(role))
    }

    fn collect_expressions<'a>(&'a self, output: &mut Vec<&'a Expression>) {
        for expression in self.expressions.values() {
            output.push(expression.expression());

            for child in expression.children() {
                child.collect_expressions(output);
            }
        }
    }

    fn all_expressions(&self) -> Vec<&Expression> {
        let mut expressions = Vec::new();
        self.collect_expressions(&mut expressions);
        expressions
    }
}

pub struct FittedFormulas {
    entries: Vec<FittedFormula>,
}

impl FromIterator<FittedFormula> for FittedFormulas {
    fn from_iter<T: IntoIterator<Item = FittedFormula>>(entries: T) -> Self {
        Self::new(entries.into_iter().collect())
    }
}

impl FittedFormulas {
    pub(crate) fn new(entries: Vec<FittedFormula>) -> Self {
        Self { entries }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, FittedFormula> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn get(&self, formula: &Formula) -> Option<&FittedFormula> {
        self.entries.iter().find(|model| model.formula() == formula)
    }

    pub fn get_by_str<'a>(&self, source: &'a str) -> ParseResult<'a, Option<&FittedFormula>> {
        let formula = parse_one(source)?;
        Ok(self.get(&formula))
    }

    fn expressions(&self) -> impl Iterator<Item = &Expression> {
        self.iter().flat_map(FittedFormula::all_expressions)
    }

    pub fn transform(&self, data: &DataFrame) -> Result<Self, ModelError> {
        let expressions: Vec<&Expression> = self.expressions().collect();
        let terms = transform_terms(&expressions, data)?;

        let entries = self
            .iter()
            .map(|model| model.assemble(data.height(), &terms))
            .collect::<Result<Vec<_>, ModelError>>()?;

        Ok(Self { entries })
    }

    pub fn select(&self, roles: &[Role]) -> Result<Self, SelectionError> {
        if roles.is_empty() {
            return Err(SelectionError::EmptySelection);
        }

        let entries = self
            .iter()
            .map(|model| model.select(roles))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self::new(entries))
    }
}

impl<'a> IntoIterator for &'a FittedFormulas {
    type Item = &'a FittedFormula;
    type IntoIter = std::slice::Iter<'a, FittedFormula>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

#[derive(Debug)]
pub enum ModelError {
    Validation(ValidationError),
    Transform(TransformError),
    Fit(FitError<TransformError>),
    TransformCall(String),
}

impl From<ValidationError> for ModelError {
    fn from(error: ValidationError) -> Self {
        Self::Validation(error)
    }
}

impl From<TransformError> for ModelError {
    fn from(error: TransformError) -> Self {
        Self::Transform(error)
    }
}

impl From<FitError<TransformError>> for ModelError {
    fn from(error: FitError<TransformError>) -> Self {
        Self::Fit(error)
    }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(error) => error.fmt(f),
            Self::Transform(error) => error.fmt(f),
            Self::Fit(error) => error.fmt(f),
            Self::TransformCall(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ModelError {}

#[derive(Debug)]
pub enum SelectionError {
    EmptySelection,
    MissingRole(Role),
    InvalidQuery(String),
    MissingTerm(String),
    Frame(polars::error::PolarsError),
}

impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySelection => f.write_str("select at least one role"),
            Self::MissingRole(role) => {
                write!(f, "model does not contain role {role:?}")
            }
            Self::InvalidQuery(message) => f.write_str(message),
            Self::MissingTerm(term) => {
                write!(f, "model does not contain term {term}")
            }
            Self::Frame(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for SelectionError {}

#[cfg(test)]
mod tests {
    use super::{Role, SelectionError};

    mod varying_slopes {
        use super::*;
        use crate::{ComponentKey, FittedFormula, ModelError, ValidationError};
        use polars::prelude::{Column, DataFrame, DataType, PlSmallStr, Scalar};

        fn fit(source: &str, data: &DataFrame) -> FittedFormula {
            crate::parse(source)
                .unwrap()
                .fit(data)
                .unwrap()
                .iter()
                .next()
                .unwrap()
                .clone()
        }

        #[test]
        fn prediction_reuses_group_encoding_and_updates_loadings() {
            let training = polars::df!(
                "y" => [1.0, 2.0, 3.0], "x" => [4.0, 5.0, 6.0],
                "firm" => ["a", "b", "a"], "time" => [10_i32, 20, 30],
            )
            .unwrap();
            let model = fit("y ~ x | firm[time]", &training);
            let fitted = model.expression(Role::FixedEffect).unwrap().frame();
            assert_eq!(fitted.shape(), (3, 2));
            assert_eq!(
                fitted[0].u32().unwrap().iter().collect::<Vec<_>>(),
                [Some(0), Some(1), Some(0)],
            );
            assert_eq!(
                model
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .get_column_names(),
                ["x"],
            );

            let input = polars::df!(
                "y" => [7.0, 8.0, 9.0], "x" => [10.0, 11.0, 12.0],
                "firm" => ["b", "a", "unseen"], "time" => [40_i32, 50, 60],
            )
            .unwrap();
            let predicted = model.transform(&input).unwrap();
            let effects = predicted.expression(Role::FixedEffect).unwrap().frame();
            assert_eq!(effects.shape(), (3, 2));
            assert_eq!(effects.get_column_names(), fitted.get_column_names());
            assert_eq!(
                effects[0].u32().unwrap().iter().collect::<Vec<_>>(),
                [Some(1), Some(0), None],
            );
            assert_eq!(
                effects[1].f64().unwrap().iter().collect::<Vec<_>>(),
                [Some(40.0), Some(50.0), Some(60.0)],
            );
        }

        #[test]
        fn slope_only_joint_groups_preserve_nulls_and_empty_prediction_schema() {
            let data = polars::df!(
                "y" => [1.0, 2.0, 3.0], "x" => [4.0, 5.0, 6.0],
                "firm" => ["a", "a", "b"], "year" => [2020_i32, 2021, 2020],
                "time" => [Some(2_i32), None, Some(4)],
                "exposure" => [Some(0.5), Some(1.0), None],
            )
            .unwrap();
            let model = fit("y ~ x | firm:year[[time, exposure]]", &data);
            assert_eq!(
                model
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .get_column_names(),
                ["Intercept", "x"],
            );
            let effects = model.expression(Role::FixedEffect).unwrap().frame();
            assert_eq!(effects.shape(), (3, 3));
            assert_eq!(
                effects[0].u32().unwrap().iter().collect::<Vec<_>>(),
                [Some(0), Some(1), Some(2)],
            );
            assert_eq!(
                effects[1].f64().unwrap().iter().collect::<Vec<_>>(),
                [Some(2.0), None, Some(4.0)],
            );
            assert_eq!(
                effects[2].f64().unwrap().iter().collect::<Vec<_>>(),
                [Some(0.5), Some(1.0), None],
            );

            let predicted = model.transform(&data.head(Some(0))).unwrap();
            let empty = predicted.expression(Role::FixedEffect).unwrap().frame();
            assert_eq!(empty.shape(), (0, 3));
            assert_eq!(empty.schema(), effects.schema());
        }

        #[test]
        fn term_selection_preserves_complete_terms_and_prediction_dependencies() {
            let data = polars::df!(
                "y" => [1.0, 2.0, 3.0], "x" => [4.0, 5.0, 6.0],
                "firm" => ["a", "b", "a"], "year" => [2020_i32, 2021, 2020],
                "time" => [10_i32, 20, 30], "exposure" => [0.5, 1.0, 1.5],
            )
            .unwrap();
            let model = fit("y ~ x | firm[[time, exposure]] + year", &data);
            let selected = model
                .select_terms("firm[[time, exposure, time]]", Role::FixedEffect)
                .unwrap();
            assert_eq!(
                selected.frame().unwrap().get_column_names(),
                [
                    "G(firm, time, exposure)[group]",
                    "G(firm, time, exposure)[loading:1]",
                    "G(firm, time, exposure)[loading:2]",
                ],
            );
            let repeated = selected
                .select_terms("firm[[time, exposure]]", Role::FixedEffect)
                .unwrap();
            assert_eq!(repeated.frame().unwrap().shape(), (3, 3));
            assert_eq!(
                repeated.frame().unwrap()[2]
                    .f64()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>(),
                [Some(0.5), Some(1.0), Some(1.5)],
            );
            for query in [
                "firm[[time]]",
                "firm[[exposure]]",
                "firm[[exposure, time]]",
                "firm[time, exposure]",
            ] {
                assert!(
                    matches!(
                        selected.select_terms(query, Role::FixedEffect),
                        Err(SelectionError::MissingTerm(_)),
                    ),
                    "{query}"
                );
            }
            drop(model);

            let input = polars::df!(
                "firm" => ["b", "unseen"], "time" => [40_i32, 50],
                "exposure" => [2.0, 2.5],
            )
            .unwrap();
            let predicted = selected.transform(&input).unwrap();
            let frame = predicted.frame().unwrap();
            assert_eq!(frame.shape(), (2, 3));
            assert_eq!(
                frame[0].u32().unwrap().iter().collect::<Vec<_>>(),
                [Some(1), None],
            );
            assert_eq!(
                frame[1].f64().unwrap().iter().collect::<Vec<_>>(),
                [Some(40.0), Some(50.0)],
            );
            assert_eq!(
                frame[2].f64().unwrap().iter().collect::<Vec<_>>(),
                [Some(2.0), Some(2.5)],
            );
        }

        #[test]
        fn term_selection_requires_each_requested_loading_and_group_intercept() {
            let data = polars::df!(
                "y" => [1.0, 2.0], "firm" => ["a", "b"], "time" => [1_i32, 2],
            )
            .unwrap();
            let model = fit("y ~ 1 | firm[[time]]", &data);
            for query in ["firm", "firm[time]", "firm[[exposure]]"] {
                assert!(
                    matches!(
                        model.select_terms(query, Role::FixedEffect),
                        Err(SelectionError::MissingTerm(_)),
                    ),
                    "{query}"
                );
            }
        }

        #[test]
        fn provenance_identifies_output_roles_group_inputs_and_nested_loadings() {
            let training = polars::df!(
                "y" => [1.0, 2.0, 3.0], "x" => [4.0, 5.0, 6.0],
                "firm" => ["a", "a", "b"], "year" => [2020_i32, 2021, 2020],
                "time" => [1_i32, 4, 9], "exposure" => [0.5, 1.0, 1.5],
            )
            .unwrap();
            let model = fit("y ~ x | firm:year[log(time), exposure]", &training);
            let input = polars::df!(
                "y" => [7.0, 8.0], "x" => [9.0, 10.0],
                "firm" => ["b", "unseen"], "year" => [2020_i32, 2021],
                "time" => [16_i32, 25], "exposure" => [2.0, 2.5],
            )
            .unwrap();
            let predicted = model.transform(&input).unwrap();

            for model in [&model, &predicted] {
                let columns = model
                    .columns(Role::FixedEffect)
                    .unwrap()
                    .collect::<Vec<_>>();
                assert_eq!(columns.len(), 3);
                for (index, column) in columns.iter().enumerate() {
                    let contributions = column.contributions().collect::<Vec<_>>();
                    let [(factor, key)] = contributions.as_slice() else {
                        panic!("expected one grouped-factor contribution");
                    };
                    let expected_key = if index == 0 {
                        ComponentKey::Scalar(Scalar::from(PlSmallStr::from_static("group")))
                    } else {
                        ComponentKey::Tuple(vec![
                            ComponentKey::Scalar(Scalar::from(PlSmallStr::from_static("loading"))),
                            ComponentKey::Scalar(Scalar::from(index as u64)),
                        ])
                    };
                    assert_eq!(**key, expected_key);
                    assert_eq!(
                        column.data().dtype(),
                        if index == 0 {
                            &DataType::UInt32
                        } else {
                            &DataType::Float64
                        }
                    );
                    assert_eq!(column.term().factors().collect::<Vec<_>>(), vec![*factor]);
                    let call = factor.transform().unwrap();
                    assert_eq!(call.name(), "G");
                    assert_eq!(call.parameters()["intercept"], Scalar::from(true));
                    let inputs = call.inputs().collect::<Vec<_>>();
                    let [group, time, exposure] = inputs.as_slice() else {
                        panic!("expected a group and two loadings");
                    };
                    let encoding = group.transform().unwrap();
                    assert_eq!(encoding.name(), "F");
                    assert_eq!(
                        encoding
                            .inputs()
                            .map(|factor| factor.name())
                            .collect::<Vec<_>>(),
                        [Some("firm"), Some("year")]
                    );
                    let transform = time.transform().unwrap();
                    assert_eq!(transform.name(), "log");
                    assert_eq!(
                        transform
                            .inputs()
                            .map(|factor| factor.name())
                            .collect::<Vec<_>>(),
                        [Some("time")]
                    );
                    assert_eq!(exposure.name(), Some("exposure"));
                }
            }
            assert_eq!(
                predicted.expression(Role::FixedEffect).unwrap().frame()[1]
                    .f64()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>(),
                [Some(16_f64.ln()), Some(25_f64.ln())],
            );
        }

        #[test]
        fn numeric_literal_loadings_use_prediction_height() {
            let data = polars::df!("y" => [1.0, 2.0], "firm" => ["a", "b"]).unwrap();
            let model = fit("y ~ 1 | firm[[2]]", &data);
            let selected = model.select_terms("firm[[2]]", Role::FixedEffect).unwrap();
            let input = polars::df!("firm" => ["b", "a", "unseen"]).unwrap();
            let predicted = selected.transform(&input).unwrap();
            assert_eq!(
                predicted.frame().unwrap()[1]
                    .f64()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>(),
                [Some(2.0); 3],
            );
        }

        #[test]
        fn rejects_nonnumeric_loading_columns() {
            for loading in [
                Column::new("loading".into(), ["a", "b"]),
                Column::new("loading".into(), [true, false]),
                Column::full_null("loading".into(), 2, &DataType::Null),
            ] {
                let data = DataFrame::new(
                    2,
                    vec![
                        Column::new("y".into(), [1.0, 2.0]),
                        Column::new("firm".into(), ["a", "b"]),
                        loading,
                    ],
                )
                .unwrap();
                let specification = crate::parse("y ~ 1 | firm[[loading]]").unwrap();
                let Err(ModelError::TransformCall(message)) = specification.fit(&data) else {
                    panic!("expected a nonnumeric loading error");
                };
                assert!(message.contains("G(F(firm), loading)"), "{message}");
                assert!(
                    message.contains("expected a primitive numeric dtype"),
                    "{message}"
                );
            }
        }

        #[test]
        fn rejects_expanding_loadings_even_when_their_output_has_one_column() {
            let data = polars::df!(
                "y" => [1.0, 2.0], "firm" => ["a", "b"],
                "category" => ["a", "b"], "time" => [1_i32, 2],
            )
            .unwrap();
            for loading in [
                "C(category)",
                "C(category, reference='a')",
                "poly(time, degree=2, raw=true)",
                "poly(time, degree=1, raw=true)",
                "log(C(category))",
            ] {
                let source = format!("y ~ 1 | firm[[{loading}]]");
                let specification = crate::parse(&source).unwrap();
                assert!(
                    matches!(
                        specification.fit(&data),
                        Err(ModelError::Validation(
                            ValidationError::InvalidTransformComposition
                        )),
                    ),
                    "{source}"
                );
            }
        }
    }

    mod column_provenance {
        use super::*;
        use crate::{ComponentKey, FittedFormula};
        use polars::prelude::{DataFrame, PlSmallStr, Scalar};

        fn fit(source: &str, data: &DataFrame) -> FittedFormula {
            crate::parse(source)
                .unwrap()
                .fit(data)
                .unwrap()
                .iter()
                .next()
                .unwrap()
                .clone()
        }

        #[test]
        fn categorical_interactions_identify_factors_and_levels() {
            let data =
                polars::df!("y" => [1.0, 2.0, 3.0], "g" => ["a", "b", "c"], "x" => [2.0, 3.0, 4.0])
                    .unwrap();
            let fitted = fit("y ~ C(g, reference='a'):x - 1", &data);
            let columns = fitted
                .columns(Role::Independent)
                .unwrap()
                .collect::<Vec<_>>();
            assert_eq!(columns.len(), 2);

            for (column, level, expected) in [
                (&columns[0], "b", [Some(0.0), Some(3.0), Some(0.0)]),
                (&columns[1], "c", [Some(0.0), Some(0.0), Some(4.0)]),
            ] {
                let contributions = column.contributions().collect::<Vec<_>>();
                assert_eq!(contributions.len(), 2);
                let (categorical, key) = contributions[0];
                assert!(categorical.name().is_none());
                assert_eq!(categorical.transform().unwrap().name(), "C");
                assert_eq!(
                    key,
                    &ComponentKey::Scalar(Scalar::from(PlSmallStr::from(level)))
                );
                assert_eq!(
                    categorical.transform().unwrap().parameters()["reference"],
                    Scalar::from(PlSmallStr::from_static("a")),
                );
                let inputs = categorical
                    .transform()
                    .unwrap()
                    .inputs()
                    .collect::<Vec<_>>();
                assert_eq!(inputs.len(), 1);
                assert_eq!(inputs[0].name(), Some("g"));

                let (variable, key) = contributions[1];
                assert_eq!(variable.name(), Some("x"));
                assert_eq!(key, &ComponentKey::None);
                assert!(variable.transform().is_none());
                assert_eq!(
                    column.term().factors().collect::<Vec<_>>(),
                    vec![categorical, variable],
                );
                assert_eq!(
                    column.data().f64().unwrap().iter().collect::<Vec<_>>(),
                    expected
                );
            }
        }

        #[test]
        fn expanding_interactions_pair_levels_with_basis_indices() {
            let data =
                polars::df!("y" => [1.0, 2.0], "g" => ["a", "b"], "x" => [2.0, 3.0]).unwrap();
            let fitted = fit("y ~ C(g):poly(x, degree=2, raw=true) - 1", &data);
            let columns = fitted
                .columns(Role::Independent)
                .unwrap()
                .collect::<Vec<_>>();
            assert_eq!(columns.len(), 4);

            for (column, (level, basis, expected)) in columns.iter().zip([
                ("a", 1_i64, [Some(2.0), Some(0.0)]),
                ("a", 2, [Some(4.0), Some(0.0)]),
                ("b", 1, [Some(0.0), Some(3.0)]),
                ("b", 2, [Some(0.0), Some(9.0)]),
            ]) {
                let contributions = column.contributions().collect::<Vec<_>>();
                assert_eq!(contributions.len(), 2);
                assert_eq!(contributions[0].0.transform().unwrap().name(), "C");
                assert_eq!(
                    contributions[0].1,
                    &ComponentKey::Scalar(Scalar::from(PlSmallStr::from(level))),
                );
                assert_eq!(contributions[1].0.transform().unwrap().name(), "poly");
                assert_eq!(
                    contributions[1].1,
                    &ComponentKey::Scalar(Scalar::from(basis))
                );
                assert_eq!(
                    column.data().f64().unwrap().iter().collect::<Vec<_>>(),
                    expected
                );
            }
        }

        #[test]
        fn selection_and_prediction_preserve_contributions_and_fitted_order() {
            let training = polars::df!(
                "y" => [1.0, 2.0, 3.0], "g" => ["a", "b", "c"], "x" => [2.0, 3.0, 4.0], "z" => [5.0, 6.0, 7.0],
            )
            .unwrap();
            let fitted = fit("y ~ z + C(g, reference='a')*x", &training);
            let selected = fitted
                .select_terms("x:C(g, reference='a')", Role::Independent)
                .unwrap()
                .select_terms("C(g, reference='a'):x", Role::Independent)
                .unwrap();
            let input = polars::df!(
                "g" => ["c", "b", "a", "new"], "x" => [10.0, 20.0, 30.0, 40.0],
            )
            .unwrap();
            let predicted = selected.transform(&input).unwrap();

            for model in [&selected, &predicted] {
                let columns = model
                    .columns(Role::Independent)
                    .unwrap()
                    .collect::<Vec<_>>();
                assert_eq!(columns.len(), 2);
                for (column, level) in columns.iter().zip(["b", "c"]) {
                    let contributions = column.contributions().collect::<Vec<_>>();
                    assert_eq!(contributions.len(), 2);
                    assert_eq!(contributions[0].0.transform().unwrap().name(), "C");
                    assert_eq!(
                        contributions[0].1,
                        &ComponentKey::Scalar(Scalar::from(PlSmallStr::from(level))),
                    );
                    assert_eq!(contributions[1].0.name(), Some("x"));
                    assert_eq!(contributions[1].1, &ComponentKey::None);
                }
            }
            let columns = predicted
                .columns(Role::Independent)
                .unwrap()
                .collect::<Vec<_>>();
            assert_eq!(
                columns[0].data().f64().unwrap().iter().collect::<Vec<_>>(),
                [Some(0.0), Some(20.0), Some(0.0), None],
            );
            assert_eq!(
                columns[1].data().f64().unwrap().iter().collect::<Vec<_>>(),
                [Some(10.0), Some(0.0), Some(0.0), None],
            );
        }

        #[test]
        fn nested_transforms_expose_inputs_and_configuration() {
            let data = polars::df!("y" => [1.0, 2.0], "x" => [1.0, std::f64::consts::E]).unwrap();
            let fitted = fit("y ~ poly(log(x), degree=2, raw=true) - 1", &data);
            let columns = fitted
                .columns(Role::Independent)
                .unwrap()
                .collect::<Vec<_>>();
            assert_eq!(columns.len(), 2);

            for (column, basis) in columns.iter().zip([1_i64, 2]) {
                let contributions = column.contributions().collect::<Vec<_>>();
                assert_eq!(contributions.len(), 1);
                let (polynomial, key) = contributions[0];
                assert!(polynomial.name().is_none());
                assert_eq!(polynomial.transform().unwrap().name(), "poly");
                assert_eq!(key, &ComponentKey::Scalar(Scalar::from(basis)));
                let parameters = polynomial.transform().unwrap().parameters();
                assert_eq!(parameters.len(), 2);
                assert_eq!(parameters["degree"], Scalar::from(2_i64));
                assert_eq!(parameters["raw"], Scalar::from(true));

                let inputs = polynomial.transform().unwrap().inputs().collect::<Vec<_>>();
                assert_eq!(inputs.len(), 1);
                let logarithm = inputs[0];
                assert!(logarithm.name().is_none());
                assert_eq!(logarithm.transform().unwrap().name(), "log");
                assert!(logarithm.transform().unwrap().parameters().is_empty());
                let inputs = logarithm.transform().unwrap().inputs().collect::<Vec<_>>();
                assert_eq!(inputs.len(), 1);
                assert_eq!(inputs[0].name(), Some("x"));
                assert!(inputs[0].transform().is_none());

                for (actual, expected) in column.data().f64().unwrap().iter().zip([0.0, 1.0]) {
                    assert!((actual.unwrap() - expected).abs() < 1e-12);
                }
            }
        }

        #[test]
        fn literal_transform_inputs_preserve_values_and_types() {
            let data = polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0]).unwrap();
            for (source, expected) in [
                ("y ~ add(x, 2) - 1", Scalar::from(2_i64)),
                ("y ~ add(x, 2.0) - 1", Scalar::from(2.0_f64)),
            ] {
                let fitted = fit(source, &data);
                let column = fitted.columns(Role::Independent).unwrap().next().unwrap();
                let contributions = column.contributions().collect::<Vec<_>>();
                assert_eq!(contributions.len(), 1);
                let (factor, key) = contributions[0];
                assert!(factor.name().is_none());
                assert!(factor.value().is_none());
                assert_eq!(key, &ComponentKey::None);
                let call = factor.transform().unwrap();
                assert_eq!(call.name(), "add");
                assert!(call.parameters().is_empty());
                let inputs = call.inputs().collect::<Vec<_>>();
                assert_eq!(inputs.len(), 2);
                assert_eq!(inputs[0].name(), Some("x"));
                assert!(inputs[0].value().is_none());
                assert!(inputs[0].transform().is_none());
                assert!(inputs[1].name().is_none());
                assert_eq!(inputs[1].value(), Some(&expected));
                assert!(inputs[1].transform().is_none());
            }
        }

        #[test]
        fn intercepts_and_zero_column_terms_preserve_their_semantics() {
            let training = polars::df!("y" => [1.0, 2.0], "g" => ["a", "a"]).unwrap();
            let fitted = fit("y ~ C(g, reference='a')", &training);
            let columns = fitted
                .columns(Role::Independent)
                .unwrap()
                .collect::<Vec<_>>();
            assert_eq!(columns.len(), 1);
            assert!(columns[0].term().factors().next().is_none());
            assert!(columns[0].contributions().next().is_none());
            assert_eq!(
                columns[0].data().f64().unwrap().iter().collect::<Vec<_>>(),
                [Some(1.0), Some(1.0)],
            );

            let selected = fitted
                .select_terms("C(g, reference='a')", Role::Independent)
                .unwrap();
            let input = polars::df!("g" => ["a", "new", "a"]).unwrap();
            for data in [&input, &input.head(Some(0))] {
                let predicted = selected.transform(data).unwrap();
                assert_eq!(predicted.columns(Role::Independent).unwrap().count(), 0);
                assert_eq!(predicted.frame().unwrap().shape(), (data.height(), 0));
                assert!(
                    predicted
                        .select_terms("C(g, reference='a')", Role::Independent)
                        .is_ok()
                );
            }
        }

        #[test]
        fn columns_report_missing_roles() {
            let data = polars::df!("y" => [1.0, 2.0], "x" => [1.0, 2.0]).unwrap();
            let fitted = fit("y ~ x", &data);
            assert!(matches!(
                fitted.columns(Role::FixedEffect),
                Err(SelectionError::MissingRole(Role::FixedEffect)),
            ));
        }
    }

    mod term_selection {
        use super::*;
        use crate::FittedFormula;
        use polars::prelude::{DataFrame, DataType};

        fn fit(source: &str, data: &DataFrame) -> FittedFormula {
            crate::parse(source)
                .unwrap()
                .fit(data)
                .unwrap()
                .iter()
                .next()
                .unwrap()
                .clone()
        }

        #[test]
        fn expansion_preserves_fitted_order_and_deduplicates_outputs() {
            let data = polars::df!(
                "y" => [1.0, 2.0], "x" => [3.0, 4.0],
                "z" => [5.0, 6.0], "w" => [7.0, 8.0],
            )
            .unwrap();
            let model = fit("y ~ w + z + z:x + x", &data);
            let expected = polars::df!(
                "z" => [5.0, 6.0], "z:x" => [15.0, 24.0], "x" => [3.0, 4.0],
            )
            .unwrap();
            for query in ["x*z", "x + x:z + z + z:x", "(x*z + w) - w"] {
                let selected = model.select_terms(query, Role::Independent).unwrap();
                assert!(
                    selected.frame().unwrap().equals_missing(&expected),
                    "{query}"
                );
                assert_eq!(selected.formula(), model.formula());
                assert!(selected.expression(Role::Dependent).is_none());
            }
            let intercept = model.select_terms("1", Role::Independent).unwrap();
            assert_eq!(intercept.frame().unwrap().shape(), (2, 1));
            assert_eq!(
                intercept.frame().unwrap()[0].f64().unwrap().get(0),
                Some(1.0)
            );
        }

        #[test]
        fn multicolumn_selection_rebases_ranges_for_repeated_selection() {
            let data = polars::df!(
                "y" => [1.0, 2.0, 3.0], "x" => [4.0, 5.0, 6.0],
                "g" => ["a", "b", "c"], "z" => [7.0, 8.0, 9.0],
            )
            .unwrap();
            let model = fit("y ~ x + C(g, reference='a') + z", &data);
            let original = model.select(&[Role::Independent]).unwrap().frame().unwrap();
            let selected = model
                .select_terms("z + C(g, reference='a')", Role::Independent)
                .unwrap();
            assert!(
                selected
                    .frame()
                    .unwrap()
                    .equals_missing(&original.select(&original.get_column_names()[2..]).unwrap())
            );
            let categorical = selected
                .select_terms("C(g, reference='a')", Role::Independent)
                .unwrap();
            let frame = categorical.frame().unwrap();
            assert_eq!(frame.width(), 2);
            assert_eq!(
                frame[0].bool().unwrap().iter().collect::<Vec<_>>(),
                [Some(false), Some(true), Some(false)]
            );
            assert_eq!(
                frame[1].bool().unwrap().iter().collect::<Vec<_>>(),
                [Some(false), Some(false), Some(true)]
            );
            assert!(
                selected
                    .select_terms("z", Role::Independent)
                    .unwrap()
                    .frame()
                    .unwrap()
                    .equals_missing(&data.select(["z"]).unwrap())
            );
        }

        #[test]
        fn selection_retains_fitted_categories_and_only_selected_dependencies() {
            let data = polars::df!(
                "y" => [1.0, 2.0, 3.0], "x" => [4.0, 5.0, 6.0], "g" => ["a", "b", "c"],
            )
            .unwrap();
            let model = fit("y ~ x + C(g, reference='a')", &data);
            let selected = model
                .select_terms("C(g, reference='a')", Role::Independent)
                .unwrap();
            let input = polars::df!("g" => ["c", "b", "unseen"]).unwrap();
            let applied = selected.transform(&input).unwrap();
            let frame = applied.frame().unwrap();
            assert_eq!(
                frame.get_column_names(),
                selected.frame().unwrap().get_column_names()
            );
            assert_eq!(
                frame[0].bool().unwrap().iter().collect::<Vec<_>>(),
                [Some(false), Some(true), None]
            );
            assert_eq!(
                frame[1].bool().unwrap().iter().collect::<Vec<_>>(),
                [Some(true), Some(false), None]
            );
            assert!(
                applied
                    .select_terms("C(g, reference='a')", Role::Independent)
                    .unwrap()
                    .frame()
                    .unwrap()
                    .equals_missing(&frame)
            );
            assert_eq!(
                selected.frame().unwrap()[1].bool().unwrap().get(0),
                Some(false)
            );
        }

        #[test]
        fn selection_uses_role_specific_grammar_and_fitted_group_encoding() {
            let data = polars::df!(
                "y" => [1.0, 2.0, 3.0], "x" => [4.0, 5.0, 6.0],
                "f" => ["a", "a", "b"], "h" => ["u", "v", "u"],
            )
            .unwrap();
            let model = fit("y ~ x | f + f:h", &data);
            assert!(
                model
                    .select_terms("y", Role::Dependent)
                    .unwrap()
                    .frame()
                    .unwrap()
                    .equals_missing(&data.select(["y"]).unwrap())
            );
            let selected = model
                .select(&[Role::FixedEffect])
                .unwrap()
                .select_terms("f:h", Role::FixedEffect)
                .unwrap();
            assert!(selected.expression(Role::Independent).is_none());
            let input = polars::df!("f" => ["b", "a", "new"], "h" => ["u", "u", "u"]).unwrap();
            let frame = selected.transform(&input).unwrap().frame().unwrap();
            assert_eq!(frame.shape(), (3, 1));
            assert_eq!(frame.get_column_names(), ["G(f:h, intercept=true)[group]"]);
            assert_eq!(frame[0].dtype(), &DataType::UInt32);
            assert_eq!(
                frame[0].u32().unwrap().iter().collect::<Vec<_>>(),
                [Some(2), Some(0), None]
            );
        }

        #[test]
        fn empty_selections_and_zero_column_terms_preserve_rows() {
            let data = polars::df!(
                "y" => [1.0, 2.0], "x" => [3.0, 4.0], "g" => ["a", "a"],
            )
            .unwrap();
            let model = fit("y ~ x + C(g, reference='a')", &data);
            for query in ["x - x", "C(g, reference='a')"] {
                let selected = model.select_terms(query, Role::Independent).unwrap();
                assert_eq!(selected.frame().unwrap().shape(), (2, 0));
                for input in [&data, &data.head(Some(0))] {
                    assert_eq!(
                        selected.transform(input).unwrap().frame().unwrap().shape(),
                        (input.height(), 0)
                    );
                }
            }
            // A zero-width term remains present and selectable.
            let selected = model
                .select_terms("C(g, reference='a')", Role::Independent)
                .unwrap();
            assert!(
                selected
                    .select_terms("C(g, reference='a')", Role::Independent)
                    .is_ok()
            );
        }

        #[test]
        fn selection_errors_distinguish_absent_roles_and_terms() {
            let data = polars::df!(
                "y" => [1.0, 2.0], "x" => [3.0, 4.0], "g" => ["a", "b"],
            )
            .unwrap();
            let model = fit("y ~ x + C(g, reference='a') - 1", &data);
            assert!(matches!(
                model.select_terms("g", Role::FixedEffect),
                Err(SelectionError::MissingRole(Role::FixedEffect))
            ));
            for query in ["missing", "x + missing", "C(g, reference='b')", "1"] {
                assert!(
                    matches!(
                        model.select_terms(query, Role::Independent),
                        Err(SelectionError::MissingTerm(_))
                    ),
                    "{query}"
                );
            }
            let with_fe = fit("y ~ x | g", &data);
            assert!(matches!(
                with_fe.select_terms("1", Role::Independent),
                Err(SelectionError::MissingTerm(_))
            ));
        }

        #[test]
        fn queries_reject_unsupported_shapes_and_multiple_alternatives() {
            let data = polars::df!(
                "y" => [1.0, 2.0], "x" => [3.0, 4.0], "g" => ["a", "b"],
            )
            .unwrap();
            let model = fit("y ~ x | g", &data);
            for (query, role) in [
                ("x +", Role::Independent),
                ("[y ~ x]", Role::Independent),
                ("sw(x, x)", Role::Independent),
                ("x + (sw0(x))", Role::Independent),
                ("g*g", Role::FixedEffect),
                ("sw(g, g)", Role::FixedEffect),
            ] {
                assert!(
                    matches!(
                        model.select_terms(query, role),
                        Err(SelectionError::InvalidQuery(_))
                    ),
                    "{query}"
                );
            }
            // Until #92, single-alternative stepwise calls are accepted.
            let selected = model.select_terms("sw(x)", Role::Independent).unwrap();
            assert!(
                selected
                    .frame()
                    .unwrap()
                    .equals_missing(&data.select(["x"]).unwrap())
            );
        }
    }

    #[test]
    fn model_frame_combines_roles_and_preserves_prediction_values() {
        let data = polars::df!(
            "z" => [7.0, 8.0, 9.0],
            "x" => [4.0, 5.0, 6.0],
            "y" => [1.0, 2.0, 3.0],
        )
        .unwrap();
        let models = crate::parse("y ~ x + z - 1").unwrap().fit(&data).unwrap();
        let model = models.iter().next().unwrap();
        let expected = data.select(["y", "x", "z"]).unwrap();
        assert!(model.frame().unwrap().equals_missing(&expected));

        let selected = model.select(&[Role::Independent]).unwrap();
        assert!(
            selected
                .frame()
                .unwrap()
                .equals_missing(&data.select(["x", "z"]).unwrap())
        );

        let prediction = polars::df!(
            "z" => [30.0, 40.0], "x" => [10.0, 20.0], "y" => [50.0, 60.0],
        )
        .unwrap();
        assert!(
            model
                .transform(&prediction)
                .unwrap()
                .frame()
                .unwrap()
                .equals_missing(&prediction.select(["y", "x", "z"]).unwrap())
        );
        // Producing prediction frames must not alter the fitted training frame.
        assert!(model.frame().unwrap().equals_missing(&expected));
    }

    #[test]
    fn model_frame_preserves_rows_with_no_columns_and_empty_predictions() {
        let data = polars::df!("y" => [1.0, 2.0, 3.0], "x" => [4.0, 5.0, 6.0]).unwrap();
        let models = crate::parse("y ~ x - x - 1").unwrap().fit(&data).unwrap();
        let model = models.iter().next().unwrap();
        assert!(
            model
                .frame()
                .unwrap()
                .equals_missing(&data.select(["y"]).unwrap())
        );
        assert_eq!(
            model
                .transform(&data.head(Some(0)))
                .unwrap()
                .frame()
                .unwrap()
                .shape(),
            (0, 1)
        );

        // An empty formula still carries the input height through the public API.
        let models = crate::parse("y - y ~ x - x - 1")
            .unwrap()
            .fit(&data)
            .unwrap();
        let model = models.iter().next().unwrap();
        assert_eq!(model.frame().unwrap().shape(), (3, 0));
        assert_eq!(
            model
                .transform(&data.head(Some(2)))
                .unwrap()
                .frame()
                .unwrap()
                .shape(),
            (2, 0)
        );
    }

    #[test]
    fn models_select_preserves_formula_keys_order_and_fitted_state() {
        let data = polars::df!(
            "y" => [1.0, 2.0, 3.0], "x" => [4.0, 5.0, 6.0],
            "g" => ["a", "b", "a"],
        )
        .unwrap();
        let models = crate::parse_many(["y ~ C(g) - 1", "y ~ x - 1"])
            .unwrap()
            .fit(&data)
            .unwrap();
        let selected = models.select(&[Role::Independent]).unwrap();
        let input = data.select(["g", "x"]).unwrap().slice(1, 1);
        let transformed = selected.transform(&input).unwrap();

        assert_eq!(selected.len(), models.len());
        assert_eq!(transformed.len(), models.len());
        for ((original, selected_model), applied) in
            models.iter().zip(selected.iter()).zip(transformed.iter())
        {
            assert_eq!(selected_model.formula(), original.formula());
            assert_eq!(applied.formula(), original.formula());
            assert!(selected.get(original.formula()).is_some());
            assert!(selected_model.expression(Role::Dependent).is_none());
            assert!(applied.expression(Role::Dependent).is_none());
            let training = original.expression(Role::Independent).unwrap().frame();
            assert!(
                applied
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .equals_missing(&training.slice(1, 1))
            );
            assert!(
                original
                    .expression(Role::Dependent)
                    .unwrap()
                    .frame()
                    .equals_missing(&data.select(["y"]).unwrap())
            );
        }
    }

    #[test]
    fn select_rejects_missing_roles_and_empty_selections() {
        let data = polars::df!(
            "y" => [1.0, 2.0], "x" => [3.0, 4.0], "g" => ["a", "b"],
        )
        .unwrap();
        let models = crate::parse_many(["y ~ x | g", "y ~ x - 1"])
            .unwrap()
            .fit(&data)
            .unwrap();
        let model = models.iter().nth(1).unwrap();
        let roles = [Role::Independent, Role::FixedEffect];
        // The role exists in the first model, but must exist in every model.
        for error in [
            model.select(&roles).err().unwrap(),
            models.select(&roles).err().unwrap(),
        ] {
            assert!(matches!(
                error,
                SelectionError::MissingRole(Role::FixedEffect)
            ));
            assert_eq!(error.to_string(), "model does not contain role FixedEffect");
        }
        for error in [
            model.select(&[]).err().unwrap(),
            models.select(&[]).err().unwrap(),
        ] {
            assert!(matches!(error, SelectionError::EmptySelection));
            assert_eq!(error.to_string(), "select at least one role");
        }
        let empty = crate::parse_many([]).unwrap().fit(&data).unwrap();
        assert!(matches!(
            empty.select(&[]),
            Err(SelectionError::EmptySelection)
        ));
    }

    #[test]
    fn fixed_effect_pipe_fits_joint_groups_without_intercept() {
        let data = polars::df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "x" => [5.0, 6.0, 7.0, 8.0],
            "f1" => ["a", "b", "a", "a"],
            "f2" => ["u", "u", "v", "u"],
        )
        .unwrap();
        let models = crate::parse("y ~ 1 + x | f1:f2")
            .unwrap()
            .fit(&data)
            .unwrap();
        let model = models.iter().next().unwrap();

        assert_eq!(
            model
                .expression(Role::Independent)
                .unwrap()
                .frame()
                .get_column_names(),
            ["x"]
        );
        let fixed_effects = model.expression(Role::FixedEffect).unwrap().frame();
        assert_eq!(fixed_effects.shape(), (4, 1));
        assert_eq!(
            fixed_effects.get_column_names(),
            ["G(f1:f2, intercept=true)[group]"]
        );
        assert_eq!(fixed_effects[0].dtype(), &polars::prelude::DataType::UInt32);
        assert_eq!(fixed_effects[0].n_unique().unwrap(), 3);
    }

    #[test]
    fn fixed_effect_names_preserve_input_names_and_factor_order() {
        let data = polars::df!(
            "y" => [1.0, 2.0], "x" => [3.0, 4.0],
            "firm name" => ["a", "b"], "f2" => ["u", "v"],
        )
        .unwrap();
        let models = crate::parse_many([
            "y ~ x | `firm name` + `firm name` : f2",
            "y ~ x | f2 : `firm name`",
        ])
        .unwrap()
        .fit(&data)
        .unwrap();
        let input = data.select(["f2", "firm name", "x", "y"]).unwrap();
        let applied = models.transform(&input).unwrap();

        for models in [&models, &applied] {
            for (model, expected) in models.iter().zip([
                vec![
                    "G(firm name, intercept=true)[group]",
                    "G(firm name:f2, intercept=true)[group]",
                ],
                vec!["G(f2:firm name, intercept=true)[group]"],
            ]) {
                assert_eq!(
                    model
                        .expression(Role::FixedEffect)
                        .unwrap()
                        .frame()
                        .get_column_names(),
                    expected
                );
            }
        }
    }

    #[test]
    fn role_queries_distinguish_absence_from_zero_column_output() {
        let data = polars::df!("y" => [1.0, 2.0], "g" => ["a", "a"]).unwrap();
        let models = crate::parse("y ~ C(g, reference='a') - 1")
            .unwrap()
            .fit(&data)
            .unwrap();
        let model = models.iter().next().unwrap();

        assert!(model.expression(Role::FixedEffect).is_none());
        assert!(matches!(
            model.children(Role::FixedEffect),
            Err(SelectionError::MissingRole(Role::FixedEffect))
        ));
        assert!(model.children(Role::Independent).unwrap().is_empty());
        assert_eq!(
            model.expression(Role::Independent).unwrap().frame().shape(),
            (2, 0)
        );
    }

    #[test]
    fn models_iterate_in_expansion_order() {
        let data = polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0], "z" => [5.0, 6.0]).unwrap();
        let specification = crate::parse_many(["y ~ sw(x, z)", "y ~ x"]).unwrap();
        let models = specification.fit(&data).unwrap();
        drop(specification);

        let expected = crate::parse_many(["y ~ x", "y ~ z"]).unwrap();
        assert_eq!(models.len(), 2);
        assert!(!models.is_empty());
        for (model, expected) in models.iter().zip(&expected) {
            assert_eq!(model.formula(), expected);
            assert_eq!(
                model
                    .expression(Role::Dependent)
                    .unwrap()
                    .frame()
                    .get_column_names(),
                ["y"]
            );
        }
        // Borrowed iteration remains available alongside the explicit iterator.
        assert_eq!((&models).into_iter().count(), models.len());
    }

    #[test]
    fn models_lookup_by_formula() {
        let data = polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0], "z" => [5.0, 6.0]).unwrap();
        for source in [
            "y ~ sw(x, z)",
            "y ~ x + [x ~ sw(x, z)]",
            "y ~ [x ~ z - 1] - 1",
        ] {
            let specification = crate::parse(source).unwrap();
            let models = specification.fit(&data).unwrap();

            for (expected, formula) in models.iter().zip(&specification) {
                assert!(std::ptr::eq(models.get(formula).unwrap(), expected));
                let rendered = formula.to_string();
                assert!(std::ptr::eq(
                    models.get_by_str(&rendered).unwrap().unwrap(),
                    expected,
                ));
            }
            for source in ["y ~ x + z", "y ~ x - 1"] {
                let query = crate::parse(source).unwrap();
                assert!(models.get(query.iter().next().unwrap()).is_none());
            }
        }
    }

    #[test]
    fn equivalent_formulas_preserve_the_first_models_column_order() {
        let data = polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0], "z" => [5.0, 6.0]).unwrap();
        let cases: &[(&[&str], &[&str])] = &[
            (&["y ~ x + z", "y ~ z + x"], &["Intercept", "x", "z"]),
            (&["y ~ z + x", "y ~ x + z"], &["Intercept", "z", "x"]),
            (&["y ~ x - 1 + 1", "y ~ x + 1"], &["x", "Intercept"]),
            (&["y ~ x + 1", "y ~ x - 1 + 1"], &["Intercept", "x"]),
        ];

        for &(sources, expected_columns) in cases {
            let models = crate::parse_many(sources.iter().copied())
                .unwrap()
                .fit(&data)
                .unwrap();
            assert_eq!(models.len(), 1, "{sources:?}");
            let first = models.iter().next().unwrap();
            assert_eq!(
                first
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .get_column_names(),
                expected_columns,
                "{sources:?}"
            );

            for source in sources {
                let found = models.get_by_str(source).unwrap().unwrap();
                assert!(std::ptr::eq(found, first), "{source}");
            }
            let rendered = first.formula().to_string();
            assert!(std::ptr::eq(
                models.get_by_str(&rendered).unwrap().unwrap(),
                first,
            ));
        }
    }

    #[test]
    fn models_lookup_by_string() {
        let data = polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0], "z" => [5.0, 6.0]).unwrap();
        let specification = crate::parse("y ~ sw(x, z)").unwrap();
        let models = specification.fit(&data).unwrap();
        drop(specification);

        // A successful lookup borrows the collection, not the query string.
        let found = {
            let query = String::from(" y~ x ");
            models.get_by_str(&query).unwrap().unwrap()
        };
        let first = models.iter().next().unwrap();
        assert!(std::ptr::eq(found, first));
        assert!(models.get_by_str("y ~ z").unwrap().is_some());
        assert!(models.get_by_str("y ~ x + z").unwrap().is_none());
        assert!(models.get_by_str("y ~ x - 1").unwrap().is_none());
        assert!(models.get_by_str("y ~").is_err());
        let error = models.get_by_str("y ~ sw(x, z)").unwrap_err();
        assert!(error.to_string().contains("expected exactly one formula"));
        assert!(models.get_by_str("y ~ sw(x, x)").unwrap().is_some());
    }
}
