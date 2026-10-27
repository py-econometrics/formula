use crate::Formula;
use crate::fit::FitError;
use crate::formula::{EvaluatedExpression, Expression, ValidationError, evaluate_terms};
use crate::parse::{ParseResult, parse};
use crate::transform::TransformError;
use polars::prelude::DataFrame;

#[derive(Clone, Debug)]
pub struct Model {
    left: EvaluatedExpression,
    right: EvaluatedExpression,
}

impl Model {
    pub(crate) fn new(left: EvaluatedExpression, right: EvaluatedExpression) -> Self {
        Self { left, right }
    }

    pub fn left(&self) -> &EvaluatedExpression {
        &self.left
    }

    pub fn right(&self) -> &EvaluatedExpression {
        &self.right
    }

    pub fn evaluate(&self, data: &DataFrame) -> Result<Self, EvaluationError> {
        let terms = evaluate_terms(&[self.left().expression(), self.right().expression()], data)?;
        let left = self.left().expression().assemble(data.height(), &terms)?;
        let right = self.right().expression().assemble(data.height(), &terms)?;

        Ok(Model::new(left, right))
    }
}

pub struct Models {
    entries: Vec<(Formula, Model)>,
}

impl Models {
    pub(crate) fn new(entries: Vec<(Formula, Model)>) -> Self {
        Self { entries }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, (Formula, Model)> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, formula: &Formula) -> Option<&Model> {
        self.entries
            .iter()
            .find_map(|(key, model)| (key == formula).then_some(model))
    }

    pub fn get_by_str<'a>(&self, source: &'a str) -> ParseResult<'a, Option<&Model>> {
        let specification = parse(source)?;
        Ok(match specification.formulas() {
            [formula] => self.get(formula),
            _ => None,
        })
    }

    fn expressions(&self) -> impl Iterator<Item = &Expression> {
        self.iter()
            .flat_map(|(_, model)| [model.left.expression(), model.right.expression()])
    }

    pub fn evaluate(&self, data: &DataFrame) -> Result<Self, EvaluationError> {
        let expressions: Vec<&Expression> = self.expressions().collect();
        let terms = evaluate_terms(&expressions, data)?;

        let entries = self
            .iter()
            .map(|(formula, model)| {
                let left = model.left().expression().assemble(data.height(), &terms)?;
                let right = model.right().expression().assemble(data.height(), &terms)?;
                Ok((formula.clone(), Model::new(left, right)))
            })
            .collect::<Result<Vec<_>, EvaluationError>>()?;
        Ok(Self { entries })
    }
}

impl<'a> IntoIterator for &'a Models {
    type Item = &'a (Formula, Model);
    type IntoIter = std::slice::Iter<'a, (Formula, Model)>;

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

#[cfg(test)]
mod tests {

    #[test]
    fn models_iterate_in_expansion_order() {
        let data = polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0], "z" => [5.0, 6.0]).unwrap();
        let specification = crate::parse_many(["y ~ sw(x, z)", "y ~ x"]).unwrap();
        let models = specification.materialize(&data).unwrap();
        drop(specification);

        let expected = crate::parse_many(["y ~ x", "y ~ z"]).unwrap();
        assert_eq!(models.len(), 2);
        assert!(!models.is_empty());
        for ((formula, model), expected) in models.iter().zip(expected.formulas()) {
            assert_eq!(formula, expected);
            assert_eq!(model.left().frame().get_column_names(), ["y"]);
        }
        // Borrowed iteration remains available alongside the explicit iterator.
        assert_eq!((&models).into_iter().count(), models.len());
    }

    #[test]
    fn models_lookup_by_formula() {
        let data = polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0], "z" => [5.0, 6.0]).unwrap();
        let specification = crate::parse("y ~ sw(x, z)").unwrap();
        let models = specification.materialize(&data).unwrap();

        for ((_, expected), formula) in models.iter().zip(specification.formulas()) {
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
        let (_, first) = models.iter().next().unwrap();
        assert!(std::ptr::eq(found, first));
        assert!(models.get_by_str("y ~ z").unwrap().is_some());
        assert!(models.get_by_str("y ~ x + z").unwrap().is_none());
        assert!(models.get_by_str("y ~ x - 1").unwrap().is_none());
        assert!(models.get_by_str("y ~").is_err());
        assert!(models.get_by_str("y ~ sw(x, z)").unwrap().is_none());
        assert!(models.get_by_str("y ~ sw(x, x)").unwrap().is_some());
    }
}
