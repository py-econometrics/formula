use crate::Formula;
use crate::fit::FitError;
use crate::formula::{
    Expression, FittedExpression, FittedTerm, Role, ValidationError, parse_one, transform_terms,
};
use crate::parse::ParseResult;
use crate::transform::TransformError;
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
}

impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySelection => f.write_str("select at least one role"),
            Self::MissingRole(role) => {
                write!(f, "model does not contain role {role:?}")
            }
        }
    }
}

impl std::error::Error for SelectionError {}

#[cfg(test)]
mod tests {
    use super::{Role, SelectionError};

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
        assert_eq!(fixed_effects.get_column_names(), ["f1:f2"]);
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
            for (model, expected) in models
                .iter()
                .zip([vec!["firm name", "firm name:f2"], vec!["f2:firm name"]])
            {
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
