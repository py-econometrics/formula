use crate::formula::{Expression, ValidationError};
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
