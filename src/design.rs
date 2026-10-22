use crate::Formula;
use crate::formula::{Expression, ValidationError};
use crate::parse::{ParseResult, parse_one};
use crate::transform::TransformError;
use polars::prelude::DataFrame;

#[derive(Clone, Debug)]
pub struct Design {
    expression: Expression,
    frame: DataFrame,
}

impl Design {
    pub(crate) fn new(expression: Expression, frame: DataFrame) -> Self {
        Self { expression, frame }
    }
    pub fn evaluate(&self, data: &DataFrame) -> Result<Self, MaterializationError> {
        self.expression.apply(data)
    }
    pub fn expression(&self) -> &Expression {
        &self.expression
    }
    pub fn frame(&self) -> &DataFrame {
        &self.frame
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    dependent: Design,
    independent: Design,
}

impl Model {
    pub(crate) fn new(dependent: Design, independent: Design) -> Self {
        Self {
            dependent,
            independent,
        }
    }

    pub fn dependent(&self) -> &Design {
        &self.dependent
    }

    pub fn independent(&self) -> &Design {
        &self.independent
    }

    pub fn evaluate(&self, data: &DataFrame) -> Result<Self, MaterializationError> {
        crate::formula::apply_model(
            self.dependent.expression(),
            self.independent.expression(),
            data,
        )
    }
}

pub struct Models {
    entries: Vec<(Formula, Model)>,
}

impl Models {
    pub(crate) fn new(entries: Vec<(Formula, Model)>) -> Self {
        Self { entries }
    }

    pub fn get<'a>(&self, formula: &'a str) -> ParseResult<'a, Option<&Model>> {
        let formula = parse_one(formula)?;
        Ok(self
            .entries
            .iter()
            .find_map(|(key, model)| (key == &formula).then_some(model)))
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
pub enum MaterializationError {
    Validation(ValidationError),
    Transform(TransformError),
}

impl From<ValidationError> for MaterializationError {
    fn from(error: ValidationError) -> Self {
        Self::Validation(error)
    }
}

impl From<TransformError> for MaterializationError {
    fn from(error: TransformError) -> Self {
        Self::Transform(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_iterate_in_expansion_order() {
        let data = polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0], "z" => [5.0, 6.0]).unwrap();
        let specification = crate::parse(&["y ~ sw(x, z)", "y ~ x"]).unwrap();
        let models = specification.materialize(&data).unwrap();
        drop(specification);

        let expected = [parse_one("y ~ x").unwrap(), parse_one("y ~ z").unwrap()];
        for ((formula, model), expected) in (&models).into_iter().zip(&expected) {
            assert_eq!(formula, expected);
            assert_eq!(model.dependent().frame().get_column_names(), ["y"]);
        }
        assert_eq!((&models).into_iter().count(), 2);
    }

    #[test]
    fn models_lookup_by_formula() {
        let data = polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0], "z" => [5.0, 6.0]).unwrap();
        let specification = crate::parse("y ~ sw(x, z)").unwrap();
        let models = specification.materialize(&data).unwrap();
        drop(specification);

        // A successful lookup borrows the collection, not the query string.
        let found = {
            let query = String::from(" y~ x ");
            models.get(&query).unwrap().unwrap()
        };
        let (_, first) = (&models).into_iter().next().unwrap();
        assert!(std::ptr::eq(found, first));
        assert!(models.get("y ~ z").unwrap().is_some());
        assert!(models.get("y ~ x + z").unwrap().is_none());
        assert!(models.get("y ~ x - 1").unwrap().is_none());
        assert!(models.get("y ~").is_err());
        assert!(models.get("y ~ sw(x, z)").is_err());
    }
}
