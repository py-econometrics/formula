use crate::Formula;
use crate::fit::FitError;
use crate::formula::{
    EvaluatedExpression, EvaluatedTerm, Expression, Role, ValidationError, evaluate_terms,
};
use crate::parse::{ParseResult, parse};
use crate::transform::TransformError;
use polars::prelude::DataFrame;
use std::collections::BTreeMap;
use winnow::Parser;
use winnow::combinator::fail;
use winnow::error::{ContextError, StrContext, StrContextValue};

#[derive(Clone, Debug)]
pub struct Model {
    formula: Formula,
    expressions: BTreeMap<Role, EvaluatedExpression>,
}

impl Model {
    pub(crate) fn new(formula: Formula, expressions: BTreeMap<Role, EvaluatedExpression>) -> Self {
        Self {
            formula,
            expressions,
        }
    }

    pub fn formula(&self) -> &Formula {
        &self.formula
    }

    fn assemble(&self, height: usize, terms: &[EvaluatedTerm]) -> Result<Self, EvaluationError> {
        let expressions = self
            .expressions
            .iter()
            .map(|(&role, evaluated)| {
                evaluated
                    .expression()
                    .assemble(height, terms)
                    .map(|expression| (role, expression))
            })
            .collect::<Result<BTreeMap<_, _>, EvaluationError>>()?;

        Ok(Self::new(self.formula.clone(), expressions))
    }

    pub fn evaluate(&self, data: &DataFrame) -> Result<Self, EvaluationError> {
        let expressions = self
            .expressions
            .values()
            .map(EvaluatedExpression::expression)
            .collect::<Vec<_>>();

        let terms = evaluate_terms(&expressions, data)?;
        self.assemble(data.height(), &terms)
    }

    pub fn expression(&self, role: Role) -> Option<&EvaluatedExpression> {
        self.expressions
            .get(&role)
            .filter(|evaluated| !evaluated.expression().is_empty())
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

        Ok(Self::new(self.formula.clone(), expressions))
    }
}

pub struct Models {
    entries: Vec<Model>,
}

impl Models {
    pub(crate) fn new(entries: Vec<Model>) -> Self {
        Self { entries }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Model> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn get(&self, formula: &Formula) -> Option<&Model> {
        self.entries.iter().find(|model| model.formula() == formula)
    }

    pub fn get_by_str<'a>(&self, source: &'a str) -> ParseResult<'a, Option<&Model>> {
        let specification = parse(source)?;

        if let [formula] = specification.formulas() {
            return Ok(self.get(formula));
        }

        // A formula string expanding to multiple formulas is not supported
        fail::<_, _, ContextError>
            .context(StrContext::Expected(StrContextValue::Description(
                "exactly one formula",
            )))
            .parse(source)
    }

    fn expressions(&self) -> impl Iterator<Item = &Expression> {
        self.iter().flat_map(|model| {
            model
                .expressions
                .values()
                .map(EvaluatedExpression::expression)
        })
    }

    pub fn evaluate(&self, data: &DataFrame) -> Result<Self, EvaluationError> {
        let expressions: Vec<&Expression> = self.expressions().collect();
        let terms = evaluate_terms(&expressions, data)?;

        let entries = self
            .iter()
            .map(|model| model.assemble(data.height(), &terms))
            .collect::<Result<Vec<_>, EvaluationError>>()?;

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

impl<'a> IntoIterator for &'a Models {
    type Item = &'a Model;
    type IntoIter = std::slice::Iter<'a, Model>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

#[derive(Debug)]
pub enum EvaluationError {
    Validation(ValidationError),
    Transform(TransformError),
    Fit(FitError<TransformError>),
    TransformCall(String),
}

impl From<ValidationError> for EvaluationError {
    fn from(error: ValidationError) -> Self {
        Self::Validation(error)
    }
}

impl From<TransformError> for EvaluationError {
    fn from(error: TransformError) -> Self {
        Self::Transform(error)
    }
}

impl From<FitError<TransformError>> for EvaluationError {
    fn from(error: FitError<TransformError>) -> Self {
        Self::Fit(error)
    }
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(error) => error.fmt(f),
            Self::Transform(error) => error.fmt(f),
            Self::Fit(error) => error.fmt(f),
            Self::TransformCall(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for EvaluationError {}

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
    fn model_select_evaluates_only_selected_roles() {
        let data = polars::df!(
            "y" => [1.0, 2.0, 3.0], "g" => ["a", "b", "a"],
        )
        .unwrap();
        let models = crate::parse("y ~ C(g) - 1")
            .unwrap()
            .materialize(&data)
            .unwrap();
        let model = models.iter().next().unwrap();

        for (role, column, excluded) in [
            (Role::Dependent, "y", Role::Independent),
            (Role::Independent, "g", Role::Dependent),
        ] {
            let training = model.expression(role).unwrap().frame().clone();
            let selected = model.select(&[role]).unwrap();
            assert_eq!(selected.formula(), model.formula());
            assert!(selected.expression(excluded).is_none());
            assert!(
                selected
                    .expression(role)
                    .unwrap()
                    .frame()
                    .equals_missing(&training)
            );

            let input = data.select([column]).unwrap().slice(1, 1);
            let evaluated = selected.evaluate(&input).unwrap();
            assert_eq!(evaluated.formula(), model.formula());
            assert!(evaluated.expression(excluded).is_none());
            assert!(
                evaluated
                    .expression(role)
                    .unwrap()
                    .frame()
                    .equals_missing(&training.slice(1, 1))
            );
            assert!(
                model
                    .expression(role)
                    .unwrap()
                    .frame()
                    .equals_missing(&training)
            );
            assert!(model.expression(excluded).is_some());
        }
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
            .materialize(&data)
            .unwrap();
        let selected = models.select(&[Role::Independent]).unwrap();
        let input = data.select(["g", "x"]).unwrap().slice(1, 1);
        let evaluated = selected.evaluate(&input).unwrap();

        assert_eq!(selected.len(), models.len());
        assert_eq!(evaluated.len(), models.len());
        for ((original, selected_model), applied) in
            models.iter().zip(selected.iter()).zip(evaluated.iter())
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
            .materialize(&data)
            .unwrap();
        let model = models.iter().nth(1).unwrap();
        let roles = [Role::Independent, Role::FixedEffect];
        assert!(matches!(
            model.select(&roles),
            Err(SelectionError::MissingRole(Role::FixedEffect))
        ));
        // The role exists in the first model, but must exist in every model.
        assert!(matches!(
            models.select(&roles),
            Err(SelectionError::MissingRole(Role::FixedEffect))
        ));
        assert!(matches!(
            model.select(&[]),
            Err(SelectionError::EmptySelection)
        ));
        assert!(matches!(
            models.select(&[]),
            Err(SelectionError::EmptySelection)
        ));

        let empty = crate::parse_many([]).unwrap().materialize(&data).unwrap();
        assert!(matches!(
            empty.select(&[]),
            Err(SelectionError::EmptySelection)
        ));
    }

    #[test]
    fn fixed_effect_pipe_materializes_joint_groups_without_intercept() {
        let data = polars::df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "x" => [5.0, 6.0, 7.0, 8.0],
            "f1" => ["a", "b", "a", "a"],
            "f2" => ["u", "u", "v", "u"],
        )
        .unwrap();
        let models = crate::parse("y ~ 1 + x | f1:f2")
            .unwrap()
            .materialize(&data)
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
        .materialize(&data)
        .unwrap();
        let input = data.select(["f2", "firm name", "x", "y"]).unwrap();
        let applied = models.evaluate(&input).unwrap();

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
            .materialize(&data)
            .unwrap();
        let model = models.iter().next().unwrap();

        assert!(model.expression(Role::FixedEffect).is_none());
        assert_eq!(
            model.expression(Role::Independent).unwrap().frame().shape(),
            (2, 0)
        );
    }

    #[test]
    fn models_iterate_in_expansion_order() {
        let data = polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0], "z" => [5.0, 6.0]).unwrap();
        let specification = crate::parse_many(["y ~ sw(x, z)", "y ~ x"]).unwrap();
        let models = specification.materialize(&data).unwrap();
        drop(specification);

        let expected = crate::parse_many(["y ~ x", "y ~ z"]).unwrap();
        assert_eq!(models.len(), 2);
        assert!(!models.is_empty());
        for (model, expected) in models.iter().zip(expected.formulas()) {
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
        let specification = crate::parse("y ~ sw(x, z)").unwrap();
        let models = specification.materialize(&data).unwrap();

        for (expected, formula) in models.iter().zip(specification.formulas()) {
            assert!(std::ptr::eq(models.get(formula).unwrap(), expected));
        }
        for source in ["y ~ x + z", "y ~ x - 1"] {
            let query = crate::parse(source).unwrap();
            assert!(models.get(&query.formulas()[0]).is_none());
        }
    }

    #[test]
    fn models_lookup_by_string() {
        let data = polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0], "z" => [5.0, 6.0]).unwrap();
        let specification = crate::parse("y ~ sw(x, z)").unwrap();
        let models = specification.materialize(&data).unwrap();
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
