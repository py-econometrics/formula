use crate::formula::{Expression, ValidationError};
use crate::transform::TransformError;
use polars::error::{PolarsError, PolarsResult};
use polars::prelude::{DataFrame, Series};

#[derive(Clone, Copy, Debug)]
pub(crate) enum MaterializationMode {
    Fit,
    Apply,
}

#[derive(Clone, Debug)]
pub struct Design {
    expression: Expression,
    columns: Vec<Series>,
}

impl Design {
    pub(crate) fn new(expression: Expression, columns: Vec<Series>) -> PolarsResult<Self> {
        let design = Self {
            expression,
            columns,
        };

        let height = design.height();
        if design.columns.iter().any(|column| column.len() != height) {
            return Err(PolarsError::ShapeMismatch(
                "design columns must have equal lengths".into(),
            ));
        }

        Ok(design)
    }
    fn height(&self) -> usize {
        self.columns.first().map_or(0, |column| column.len())
    }
    pub fn materialize(&self, data: &DataFrame) -> Result<Self, MaterializationError> {
        self.expression
            .clone()
            .materialize_with_mode(data, MaterializationMode::Apply)
    }
    pub fn expression(&self) -> &Expression {
        &self.expression
    }
    pub fn columns(&self) -> &[Series] {
        &self.columns
    }

    pub fn to_frame(&self) -> PolarsResult<DataFrame> {
        DataFrame::new(
            self.height(),
            self.columns.iter().cloned().map(Into::into).collect(),
        )
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

    pub fn materialize(&self, data: &DataFrame) -> Result<Self, MaterializationError> {
        Ok(Self::new(
            self.dependent.materialize(data)?,
            self.independent.materialize(data)?,
        ))
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
