use crate::Formula;
use crate::formula::{Expression, ValidationError};
use crate::parse::{ParseResult, parse};
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
            assert_eq!(model.dependent().frame().get_column_names(), ["y"]);
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
